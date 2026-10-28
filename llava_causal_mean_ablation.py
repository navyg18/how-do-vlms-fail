#!/usr/bin/env python3
"""
llava_causal_mean_ablation.py  (LLaVA-1.5 port of causal_mean_ablation.py)
============================================================================
Same logic as the Qwen version; only architecture-related changes:
  - Uses HookedLLaVA instead of HookedQwenVL
  - LLaVA input format (no image embedded in chat template content dict)
  - Vision token identified by config.image_token_index (32000) not <|image_pad|>
  - Fixed 24x24 patch grid (336px / 14px) instead of dynamic image_grid_thw
  - n_layers from config.text_config.num_hidden_layers
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import kendalltau, spearmanr, mannwhitneyu
from tqdm import tqdm

from HookedLLaVA import HookedLLaVA
from causal_mean_ablation import load_samples

ONSET_FRAC = 0.10
MIN_DEG    = 1e-4   # minimum peak degradation to treat curve as valid signal

# LLaVA-1.5 vision grid constants (336px image, 14px CLIP patches)
LLAVA_IMG_SIZE   = 336
LLAVA_PATCH_SIZE = 14
LLAVA_GRID_SIZE  = LLAVA_IMG_SIZE // LLAVA_PATCH_SIZE  # 24


# ══════════════════════════════════════════════════════════════════
# BBOX → VISION TOKEN POSITIONS  (LLaVA-specific)
# ══════════════════════════════════════════════════════════════════

def bbox_to_token_positions(image, bbox, vision_idx, _grid_thw=None):
    """
    Map [x, y, w, h] bbox to (offsets_in_vision_span, absolute_seq_positions)
    for LLaVA-1.5.

    LLaVA resizes every image to 336×336 with 14×14 CLIP patches → 24×24 grid
    of 576 vision tokens laid out in raster order (row-major).
    _grid_thw is accepted but ignored (kept for call-site compatibility).
    """
    if bbox is None or not vision_idx:
        return [], []

    x, y, w, h = bbox
    orig_w, orig_h = image.size

    # Scale bbox to 336×336
    x1 = x / orig_w * LLAVA_IMG_SIZE
    y1 = y / orig_h * LLAVA_IMG_SIZE
    x2 = (x + w) / orig_w * LLAVA_IMG_SIZE
    y2 = (y + h) / orig_h * LLAVA_IMG_SIZE

    col_start = max(0, int(x1 / LLAVA_PATCH_SIZE))
    row_start = max(0, int(y1 / LLAVA_PATCH_SIZE))
    col_end   = min(LLAVA_GRID_SIZE, int(np.ceil(x2 / LLAVA_PATCH_SIZE)))
    row_end   = min(LLAVA_GRID_SIZE, int(np.ceil(y2 / LLAVA_PATCH_SIZE)))

    offsets = [
        r * LLAVA_GRID_SIZE + c
        for r in range(row_start, row_end)
        for c in range(col_start, col_end)
    ]

    if not offsets:
        return [], []

    vision_set = set(vision_idx)
    valid_offsets, abs_positions = [], []
    for off in offsets:
        if off < len(vision_idx):
            pos = vision_idx[off]
            if pos in vision_set:
                valid_offsets.append(off)
                abs_positions.append(pos)

    return valid_offsets, abs_positions


# ══════════════════════════════════════════════════════════════════
# MEAN ABLATION SOURCE
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def cache_layer_stats(vlm, inputs, vis_idx: List[int]):
    """
    One clean forward pass. At every layer compute mean and std of hidden
    states over all vision token positions.

    Returns (mean_acts, std_acts):
      mean_acts[L]: [seq_len, hidden] — every row = mean over vis tokens
      std_acts[L]:  [hidden]          — per-dim std over vis tokens
    """
    n_layers = vlm.model.config.text_config.num_hidden_layers
    captured  = {}

    def make_hook(L):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if h is not None and h.dim() == 3:
                captured[L] = h[0].detach().float().cpu()   # [seq, hidden]
        return hook

    layers  = vlm._get_text_layers()
    handles = [layers[L].register_forward_hook(make_hook(L))
               for L in range(n_layers)]
    try:
        vlm.model(**inputs)
    finally:
        for h in handles:
            h.remove()

    if not captured:
        print("  [WARN] cache_layer_stats: no layers captured")
        return None, None

    mean_acts, std_acts = {}, {}
    for L in range(n_layers):
        h_full   = captured[L]                              # [seq, hidden]
        h_vis    = h_full[vis_idx]                          # [n_vis, hidden]
        mean_v   = h_vis.mean(dim=0)                        # [hidden]
        std_v    = h_vis.std(dim=0).clamp(min=1e-6)         # [hidden]
        mean_acts[L] = mean_v.unsqueeze(0).expand(h_full.shape[0], -1).clone()
        std_acts[L]  = std_v
    return mean_acts, std_acts


def cache_mean_acts(vlm, inputs, vis_idx: List[int]) -> Optional[Dict[int, torch.Tensor]]:
    mean_acts, _ = cache_layer_stats(vlm, inputs, vis_idx)
    return mean_acts


# ══════════════════════════════════════════════════════════════════
# DEGRADATION CURVE ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyse_curve(deg: List[float]) -> dict:
    arr   = np.clip(np.array(deg, dtype=float), 0, None)
    peak  = float(arr.max())
    valid = peak > MIN_DEG
    pl    = int(np.argmax(arr)) if valid else None
    onset = None
    if valid:
        hits  = np.where(arr > ONSET_FRAC * peak)[0]
        onset = int(hits[0]) if len(hits) else None
    return {
        "peak_value":  peak,
        "auc":         float(arr.sum()),
        "peak_layer":  pl,
        "onset_layer": onset,
        "valid":       valid,
    }


# ══════════════════════════════════════════════════════════════════
# PROGRAM PARSING
# ══════════════════════════════════════════════════════════════════

def parse_obj_id(arg: str) -> Optional[str]:
    m = re.search(r'\((\d+)\)', arg)
    return m.group(1) if m else None

def base_op(s: str) -> str:
    return s.split()[0] if s else ""

def extract_program_objects(
    program: List[dict],
    image_objects: Dict[str, dict],
    sample_operation: str = "",
) -> List[dict]:
    op = sample_operation.split()[0] if sample_operation else ""

    def resolve_obj(arg: str) -> Optional[dict]:
        obj_id = parse_obj_id(arg)
        obj    = image_objects.get(obj_id) if obj_id else None
        if obj is None:
            clean_name = re.sub(r'\(\d+\)', '', arg).strip().split(',')[0].strip().lower()
            if clean_name:
                for o in image_objects.values():
                    if o.get("name", "").lower() == clean_name:
                        obj = o; break
        if obj is None or not obj.get("bbox"):
            return None
        return {
            "obj_id":   str(obj.get("obj_id", "")),
            "name":     obj.get("name", "unknown"),
            "bbox":     obj["bbox"],
        }

    if op in ("select", "common", "different", ""):
        objs = []
        for step in program:
            if base_op(step.get("operation","")) == "select":
                o = resolve_obj(step.get("argument",""))
                if o:
                    objs.append(o)
        if objs:
            return [{"objects": objs, "role": "select_joint",
                     "step_depth": len(objs)}]
        return []

    if op == "query":
        for step in program:
            if base_op(step.get("operation","")) == "select":
                o = resolve_obj(step.get("argument",""))
                if o:
                    return [{"objects": [o], "role": "query_target",
                             "step_depth": 0}]
        return []

    if op == "exist":
        groups = []
        for i, step in enumerate(program):
            if base_op(step.get("operation","")) == "select":
                next_steps = [s for s in program
                              if i in s.get("dependencies", [])
                              and base_op(s.get("operation","")) == "exist"]
                if next_steps:
                    o = resolve_obj(step.get("argument",""))
                    if o:
                        groups.append({"objects": [o], "role": "exist_branch",
                                       "step_depth": i})
        return groups if groups else []

    if op == "choose":
        for step in program:
            if base_op(step.get("operation","")) == "select":
                o = resolve_obj(step.get("argument",""))
                if o:
                    return [{"objects": [o], "role": "choose_target",
                             "step_depth": 0}]
        return []

    if op == "filter":
        for step in program:
            if base_op(step.get("operation","")) == "select":
                o = resolve_obj(step.get("argument",""))
                if o:
                    return [{"objects": [o], "role": "filter_target",
                             "step_depth": 0}]
        return []

    if op == "relate":
        for step in reversed(program):
            if base_op(step.get("operation","")) == "relate":
                arg = step.get("argument","")
                o = resolve_obj(arg)
                if o:
                    return [{"objects": [o], "role": "relate_terminal",
                             "step_depth": len(program)-1}]
        return []

    if op == "verify":
        primary_obj = None
        relation_obj = None
        for step in program:
            bop = base_op(step.get("operation",""))
            if bop == "select" and primary_obj is None:
                primary_obj = resolve_obj(step.get("argument",""))
            if bop == "verify":
                relation_obj = resolve_obj(step.get("argument",""))
        objs = [o for o in [primary_obj, relation_obj] if o is not None]
        if objs:
            return [{"objects": objs, "role": "verify_joint",
                     "step_depth": 1}]
        return []

    groups = []
    for i, step in enumerate(program):
        if base_op(step.get("operation","")) == "select":
            o = resolve_obj(step.get("argument",""))
            if o:
                groups.append({"objects": [o], "role": "fallback",
                               "step_depth": i})
    return groups


# ══════════════════════════════════════════════════════════════════
# INPUT / SCORING
# ══════════════════════════════════════════════════════════════════

def build_inputs(vlm, image, question):
    """LLaVA input format: image passed separately, not embedded in content dict."""
    conversation = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    text = vlm.processor.apply_chat_template(conversation, add_generation_prompt=True)
    return vlm.processor(images=image, text=text, return_tensors="pt").to(vlm.model.device)


def get_vision_indices(vlm, inputs) -> List[int]:
    """
    Return the 576 absolute positions that vision tokens occupy in the
    *expanded* sequence seen by language_model.model.layers.

    LLaVA input_ids has ONE placeholder token (id 32000) at position p.
    Inside model.forward(), that single token is replaced by 576 projected
    vision features before the LLaMA layers run.  The transformer hidden
    states therefore have shape [expanded_len, hidden] where:
        expanded_len = original_len - 1 + image_seq_length   (576)
    Vision tokens occupy positions p .. p+575 in that expanded sequence.
    """
    image_token_id  = vlm.model.config.image_token_index   # 32000
    image_seq_length = vlm.model.config.image_seq_length   # 576
    placeholder = torch.where(inputs["input_ids"][0] == image_token_id)[0].tolist()
    if not placeholder:
        return []
    p = placeholder[0]
    return list(range(p, p + image_seq_length))


def get_answer_tokens(vlm, answer: str) -> List[int]:
    toks = vlm.processor.tokenizer.encode(f" {answer}", add_special_tokens=False)
    return toks or vlm.processor.tokenizer.encode(answer, add_special_tokens=False)


def score_answer(vlm, inputs, ans_toks: List[int]) -> float:
    if not ans_toks:
        return float("-inf")
    with torch.no_grad():
        logits = vlm.model(**inputs).logits[0]
    # logits.shape[0] is the *expanded* sequence length (original - 1 + 576).
    # The last prompt position predicts the first answer token.
    prompt_len = logits.shape[0]
    lp = F.log_softmax(logits, dim=-1)
    total, count = 0.0, 0
    for i, tok in enumerate(ans_toks):
        idx = prompt_len - 1 + i
        if 0 <= idx < len(lp):
            total += lp[idx, tok].item(); count += 1
    return total / max(count, 1)


def model_correct(vlm, inputs, answer: str):
    with torch.no_grad():
        out = vlm.model.generate(**inputs, max_new_tokens=20, do_sample=False)
    decoded = vlm.processor.tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return answer.strip().lower() in decoded.lower(), decoded


# ══════════════════════════════════════════════════════════════════
# PER-OBJECT MEAN-ABLATION PATCHING
# ══════════════════════════════════════════════════════════════════

N_NOISE_SAMPLES = 5   # number of Gaussian noise draws for fallback averaging


@torch.no_grad()
def _gaussian_deg_curve(
    vlm, inputs, ans_toks,
    all_abs_pos, mean_acts, std_acts, n_layers, clean_lp,
) -> List[float]:
    """
    Gaussian-noise ablation fallback.  For each layer, replace bbox positions
    with  mean_vis + ε * std_vis  (ε ~ N(0,1) per feature dim), then average
    log-prob degradation over N_NOISE_SAMPLES independent draws.
    """
    rng = torch.Generator()
    rng.manual_seed(0)
    accum = [0.0] * n_layers
    for _ in range(N_NOISE_SAMPLES):
        for L in range(n_layers):
            mean_v = mean_acts[L][0]                         # [hidden]
            std_v  = std_acts[L]                             # [hidden]
            noise  = torch.randn(mean_v.shape, generator=rng)
            noise_rep = mean_acts[L].clone()                 # [seq, hidden]
            for pos in all_abs_pos:
                if pos < noise_rep.shape[0]:
                    noise_rep[pos] = mean_v + noise * std_v
            with vlm.patch_at_layer(L, all_abs_pos, noise_rep):
                patched_lp = score_answer(vlm, inputs, ans_toks)
            accum[L] += float(clean_lp - patched_lp)
    return [v / N_NOISE_SAMPLES for v in accum]


@torch.no_grad()
def run_per_object_patching(
    vlm, inputs, ans_toks, vis_idx, mean_acts, std_acts,
    n_layers, clean_lp, prog_groups, image, grid_thw,
) -> List[dict]:
    results = []

    for group in prog_groups:
        objs  = group["objects"]
        role  = group["role"]
        depth = group["step_depth"]

        all_abs_pos = []
        obj_names   = []
        for obj in objs:
            _, abs_pos = bbox_to_token_positions(image, obj["bbox"], vis_idx, grid_thw)
            all_abs_pos.extend(abs_pos)
            obj_names.append(obj["name"])

        all_abs_pos = sorted(set(all_abs_pos))
        if not all_abs_pos:
            continue

        with vlm.patch_at_layer(0, all_abs_pos, mean_acts[0]):
            test_lp = score_answer(vlm, inputs, ans_toks)
        if abs(clean_lp - test_lp) < 1e-7:
            print(f"  [WARN] group='{obj_names}': layer-0 patch zero effect "
                  f"(n_tokens={len(all_abs_pos)})")

        deg = []
        for L in range(n_layers):
            with vlm.patch_at_layer(L, all_abs_pos, mean_acts[L]):
                patched_lp = score_answer(vlm, inputs, ans_toks)
            deg.append(float(clean_lp - patched_lp))

        ablation_method = "mean"

        # Gaussian-noise fallback when mean gives no valid signal.
        # Mean ablation can be symmetric for comparison groups (both objects
        # replaced identically → net-zero logit change).
        # Gaussian noise breaks that symmetry while staying in-distribution.
        if not analyse_curve(deg)["valid"]:
            deg = _gaussian_deg_curve(
                vlm, inputs, ans_toks,
                all_abs_pos, mean_acts, std_acts, n_layers, clean_lp,
            )
            ablation_method = "gaussian_noise"

        results.append({
            "obj_id":          "+".join(o["obj_id"] for o in objs),
            "name":            "+".join(obj_names),
            "operation":       role,
            "step_depth":      depth,
            "n_tokens":        len(all_abs_pos),
            "ablation_method": ablation_method,
            "degradation":     deg,
            "curve_stats":     analyse_curve(deg),
        })

    return results

# ══════════════════════════════════════════════════════════════════
# PAIR ANALYSIS (secondary)
# ══════════════════════════════════════════════════════════════════

def compute_pair_metrics(obj_results, question_id, correct_flag):
    valid = [r for r in obj_results if r["curve_stats"]["valid"]]
    if len(valid) < 2:
        return []
    pairs = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            if a["step_depth"] == b["step_depth"]:
                continue
            if a["step_depth"] > b["step_depth"]:
                a, b = b, a
            pa = a["curve_stats"]["peak_layer"]
            pb = b["curve_stats"]["peak_layer"]
            if pa is None or pb is None:
                continue
            diff = pb - pa
            pairs.append({
                "question_id":        question_id,
                "model_correct":      correct_flag,
                "obj_name_early":     a["name"],
                "step_depth_early":   a["step_depth"],
                "operation_early":    a["operation"],
                "peak_value_early":   a["curve_stats"]["peak_value"],
                "peak_layer_early":   pa,
                "obj_name_late":      b["name"],
                "step_depth_late":    b["step_depth"],
                "peak_value_late":    b["curve_stats"]["peak_value"],
                "peak_layer_late":    pb,
                "step_depth_diff":    b["step_depth"] - a["step_depth"],
                "peak_layer_diff":    diff,
                "is_ordered":         diff > 0,
                "is_tied":            diff == 0,
            })
    return pairs


def compute_rank_corr(obj_results):
    valid = [r for r in obj_results if r["curve_stats"]["valid"]]
    if len(valid) < 3:
        return {"kendall_tau": None, "spearman_r": None, "n_valid": len(valid)}
    depths = [r["step_depth"]                  for r in valid]
    peaks  = [r["curve_stats"]["peak_layer"]   for r in valid]
    tau, _ = kendalltau(depths, peaks)
    rho, _ = spearmanr(depths, peaks)
    return {
        "kendall_tau": float(tau) if not np.isnan(tau) else None,
        "spearman_r":  float(rho) if not np.isnan(rho) else None,
        "n_valid":     len(valid),
    }


# ══════════════════════════════════════════════════════════════════
# MAIN PER-SAMPLE
# ══════════════════════════════════════════════════════════════════

def analyze_sample(vlm, sample, image_objects):
    question = sample["question_clean"]
    answer   = sample["answer"]
    program  = sample.get("semantic_program", [])
    if not program:
        return None

    img_path = sample.get("image_path", "")
    if not img_path or not os.path.exists(img_path):
        return None
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  [SKIP] {img_path}: {e}"); return None

    if answer.strip().lower() in ("yes", "no"):
        question = question + " Answer with yes or no."
    elif "common" in question.lower():
        question = question + " Name the shared attribute."
        
    inputs  = build_inputs(vlm, image, question)
    vis_idx = get_vision_indices(vlm, inputs)
    # LLaVA has no image_grid_thw — vision grid is always 24x24
    if not vis_idx:
        return None

    # grid_thw is None for LLaVA; bbox_to_token_positions ignores it
    grid_thw = None
    ans_toks = get_answer_tokens(vlm, answer)
    if not ans_toks:
        return None

    prog_groups = extract_program_objects(program, image_objects,
                                      sample_operation=sample.get("operation",""))
    if not prog_groups:
        return None

    clean_lp           = score_answer(vlm, inputs, ans_toks)
    correct, model_ans = model_correct(vlm, inputs, answer)
    mean_acts, std_acts = cache_layer_stats(vlm, inputs, vis_idx)
    if mean_acts is None:
        return None

    n_layers = vlm.model.config.text_config.num_hidden_layers

    obj_results = run_per_object_patching(
        vlm, inputs, ans_toks, vis_idx, mean_acts, std_acts,
        n_layers, clean_lp, prog_groups, image, grid_thw,
    )

    pair_metrics = compute_pair_metrics(obj_results, sample["question_id"], correct)
    rank_corr    = compute_rank_corr(obj_results)

    return {
        "question_id":     sample["question_id"],
        "image_id":        sample.get("image_id", ""),
        "image_path":      img_path,
        "question":        question,
        "answer":          answer,
        "model_answer":    model_ans,
        "operation":       sample.get("operation", ""),
        "n_steps":         len(program),
        "clean_logprob":   float(clean_lp),
        "model_correct":   correct,
        "n_objects":       len(prog_groups),
        "n_valid_objects": sum(1 for r in obj_results if r["curve_stats"]["valid"]),
        "obj_results":     obj_results,
        "pair_metrics":    pair_metrics,
        "rank_correlation":rank_corr,
    }


# ══════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════

def safe_mean(vals):
    v = [x for x in vals if x is not None and not np.isnan(float(x))]
    return float(np.mean(v)) if v else None

def safe_std(vals):
    v = [x for x in vals if x is not None and not np.isnan(float(x))]
    return float(np.std(v)) if v else None

def cohen_d(a, b):
    a = np.array([x for x in a if x is not None], dtype=float)
    b = np.array([x for x in b if x is not None], dtype=float)
    if len(a) < 2 or len(b) < 2: return None
    pool = np.sqrt((np.std(a,ddof=1)**2 + np.std(b,ddof=1)**2) / 2)
    return float((np.mean(a) - np.mean(b)) / pool) if pool > 0 else None


def aggregate_results(all_samples, all_pairs):
    by_op: Dict[str, Dict] = defaultdict(lambda: {"correct": [], "incorrect": []})

    for s in all_samples:
        op      = s["operation"]
        correct = s["model_correct"]
        valid   = [r for r in s.get("objects_summary", s.get("obj_results", []))
                   if r["curve_stats"]["valid"]]
        if not valid:
            continue
        crit = max(valid, key=lambda x: (x["step_depth"], x["curve_stats"]["peak_value"]))
        by_op[op]["correct" if correct else "incorrect"].append(
            crit["curve_stats"]["peak_value"]
        )

    op_summary = {}
    for op, grps in by_op.items():
        c, i = grps["correct"], grps["incorrect"]
        p = None
        if len(c) >= 3 and len(i) >= 3:
            _, p = mannwhitneyu(c, i, alternative="two-sided")
            p = float(p)
        op_summary[op] = {
            "n_correct":                   len(c),
            "n_incorrect":                 len(i),
            "mean_peak_value_correct":     safe_mean(c),
            "mean_peak_value_incorrect":   safe_mean(i),
            "cohen_d":                     cohen_d(c, i),
            "mannwhitney_p":               p,
        }

    depth_summary = defaultdict(list)
    for s in all_samples:
        for r in s.get("objects_summary", s.get("obj_results", [])):
            if r["curve_stats"]["valid"]:
                depth_summary[r["step_depth"]].append(r["curve_stats"]["peak_value"])

    depth_summary_out = {
        str(d): {"n": len(v), "mean_peak_value": safe_mean(v), "std_peak_value": safe_std(v)}
        for d, v in sorted(depth_summary.items())
    }

    n_pairs   = len(all_pairs)
    n_ordered = sum(1 for p in all_pairs if p["is_ordered"])
    n_tied    = sum(1 for p in all_pairs if p["is_tied"])

    taus = [s["rank_correlation"]["kendall_tau"] for s in all_samples
            if s["rank_correlation"]["kendall_tau"] is not None]
    rhos = [s["rank_correlation"]["spearman_r"]  for s in all_samples
            if s["rank_correlation"]["spearman_r"]  is not None]

    n_correct = sum(1 for s in all_samples if s["model_correct"])
    return {
        "dataset": {
            "n_samples": len(all_samples),
            "n_correct": n_correct,
            "accuracy":  n_correct / len(all_samples) if all_samples else None,
        },
        "peak_value_by_operation":  op_summary,
        "peak_value_by_step_depth": depth_summary_out,
        "pair_ordering": {
            "n_pairs": n_pairs, "n_ordered": n_ordered, "n_tied": n_tied,
            "ordering_rate": n_ordered / n_pairs if n_pairs else None,
        },
        "rank_correlation": {
            "mean_kendall_tau": safe_mean(taus),
            "mean_spearman_r":  safe_mean(rhos),
            "n_samples":        len(taus),
        },
    }


# ══════════════════════════════════════════════════════════════════
# JSON HELPER
# ══════════════════════════════════════════════════════════════════

def _json_default(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, np.ndarray):           return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def run(samples_path, image_dir, model_id, output_dir, device,
        quantize_type, max_samples, save_every, seed, min_objects):

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {samples_path} ...")
    all_samples, image_objects_map = load_samples(samples_path, image_dir)
    print(f"  {len(all_samples)} samples from {len(image_objects_map)} images")
    if max_samples:
        all_samples = all_samples[:max_samples]
        print(f"  Capped at {max_samples}")

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({
            "samples_path": samples_path, "model_id": model_id,
            "device": device, "quantize_type": quantize_type,
            "ablation": "mean", "metric": "peak_degradation",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)

    print(f"\nLoading {model_id} on {device} ...")
    quantize_type = {"int4": "4bit"}.get(quantize_type, quantize_type)
    vlm = HookedLLaVA(model_id=model_id, device=device,
                      quantize=True, quantize_type=quantize_type)
    vlm.model.eval()
    print(f"  {vlm.model.config.text_config.num_hidden_layers} layers")

    path_sample = os.path.join(output_dir, "per_sample_results.jsonl")
    path_obj    = os.path.join(output_dir, "per_object_results.jsonl")
    path_pairs  = os.path.join(output_dir, "pair_results.jsonl")

    f_s = open(path_sample, "w")
    f_o = open(path_obj,    "w")
    f_p = open(path_pairs,  "w")

    collected_samples, collected_pairs = [], []
    skipped = errors = 0

    try:
        for i, sample in enumerate(tqdm(all_samples, desc="degradation grounding")):
            img_obj = image_objects_map.get(sample.get("image_id", ""), {})
            try:
                result = analyze_sample(vlm, sample, img_obj)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); errors += 1; continue
            except Exception as e:
                errors += 1
                print(f"  [ERR] {sample.get('question_id')}: {e}")
                import traceback; traceback.print_exc(); continue

            if result is None:
                skipped += 1; continue

            s_rec = {k: v for k, v in result.items() if k != "obj_results"}
            s_rec["objects_summary"] = [{
                "obj_id":      r["obj_id"], "name": r["name"],
                "operation":   r["operation"], "step_depth": r["step_depth"],
                "n_tokens":    r["n_tokens"],
                "curve_stats": r["curve_stats"],
            } for r in result["obj_results"]]
            f_s.write(json.dumps(s_rec, default=_json_default) + "\n")

            for r in result["obj_results"]:
                f_o.write(json.dumps({
                    "question_id":   result["question_id"],
                    "image_id":      result["image_id"],
                    "model_correct": result["model_correct"],
                    "clean_logprob": result["clean_logprob"],
                    "operation_q":   result["operation"],
                    **{k: v for k, v in r.items() if k != "degradation"},
                    "degradation":   r["degradation"],
                }, default=_json_default) + "\n")

            for p in result["pair_metrics"]:
                f_p.write(json.dumps(p, default=_json_default) + "\n")
                collected_pairs.append(p)

            collected_samples.append(s_rec)
            f_s.flush(); f_o.flush(); f_p.flush()

            if (i + 1) % save_every == 0:
                interim = aggregate_results(collected_samples, collected_pairs)
                with open(os.path.join(output_dir, "aggregate_summary_interim.json"), "w") as fi:
                    json.dump(interim, fi, indent=2, default=_json_default)

                print(f"\n[{i+1}/{len(all_samples)}] acc={interim['dataset']['accuracy']:.3f}")
                print("  peak_value by op  (corr / incorr / d / p):")
                def _fmt(x): return f"{x:.3f}" if x is not None else "N/A"
                for op, d in interim["peak_value_by_operation"].items():
                    print(f"    {op:12s}  "
                          f"{_fmt(d['mean_peak_value_correct']):>6}  "
                          f"{_fmt(d['mean_peak_value_incorrect']):>6}  "
                          f"d={_fmt(d['cohen_d']):>6}  "
                          f"p={d['mannwhitney_p']}")
                torch.cuda.empty_cache()

    finally:
        f_s.close(); f_o.close(); f_p.close()

    print(f"\nProcessed {len(collected_samples)} samples "
          f"({skipped} skipped, {errors} errors)")

    summary = aggregate_results(collected_samples, collected_pairs)
    with open(os.path.join(output_dir, "aggregate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print("\n" + "="*65)
    ds = summary["dataset"]
    print(f"Samples: {ds['n_samples']}  |  accuracy: {ds['accuracy']:.3f}")
    print("\nPEAK DEGRADATION BY OPERATION  (PRIMARY FINDING)")
    print(f"  {'op':12s}  {'n_c':>5}  {'n_i':>5}  "
          f"{'pv_c':>7}  {'pv_i':>7}  {'cohen_d':>8}  {'MW_p':>10}")
    for op, d in sorted(summary["peak_value_by_operation"].items()):
        def f(x): return f"{x:.3f}" if x is not None else "N/A"
        print(f"  {op:12s}  {d['n_correct']:>5}  {d['n_incorrect']:>5}  "
              f"{f(d['mean_peak_value_correct']):>7}  "
              f"{f(d['mean_peak_value_incorrect']):>7}  "
              f"{f(d['cohen_d']):>8}  "
              f"{str(round(d['mannwhitney_p'],5) if d['mannwhitney_p'] else 'N/A'):>10}")

    po = summary["pair_ordering"]
    print(f"\nSECONDARY pair ordering: "
          f"n={po['n_pairs']}  rate={po['ordering_rate']}")
    print(f"Results → {output_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples",       required=True)
    ap.add_argument("--image_dir",     default="data/images",
                    help="Folder holding the images named in --samples")
    ap.add_argument("--model_id",      default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--output_dir",    default="results/llava_causal_mean_ablation")
    ap.add_argument("--device",        default="cuda:0")
    ap.add_argument("--quantize_type", default="fp16",
                    choices=["fp16","int8","int4","4bit"])
    ap.add_argument("--max_samples",   type=int, default=None)
    ap.add_argument("--save_every",    type=int, default=20)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--min_objects",   type=int, default=1)
    args = ap.parse_args()
    run(samples_path=args.samples, image_dir=args.image_dir, model_id=args.model_id,
        output_dir=args.output_dir, device=args.device,
        quantize_type=args.quantize_type, max_samples=args.max_samples,
        save_every=args.save_every, seed=args.seed, min_objects=args.min_objects)


if __name__ == "__main__":
    main()
