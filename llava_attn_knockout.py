#!/usr/bin/env python3
"""
llava_attn_knockout.py  (Attention Knockout — LLaVA-1.5)
========================================================
Mirror of attn_knockout.py for LLaVA-1.5-7B.

Analysis A — Attention Knockout  (gate: --knockout / --no-knockout)
  Stage A — Layer-level (32 passes per sample):
    At each layer L, zero out attention from ans_pos to all bbox tokens
    across ALL heads simultaneously.  ko_curve[L] = clean_logp - ko_logp.
  Stage B — Head-level (32 × 32 = 1024 passes per sample, gate: --head_level):
    For every layer L and every head H, knock out only that one head's
    attention to bbox tokens.  head_ko_matrix[L][H] = clean_logp - ko_logp.

Architecture (LLaVA-1.5-7B, LLaMA-2-7B backbone):
  Layers: model.language_model.model.layers  (len=32)
  GQA: 32 query heads, HEAD_DIM=128, hidden=4096
  Vision: CLIP ViT-L/14 @ 336px → 24×24 = 576 fixed vision tokens

Usage
-----
python llava_attn_knockout.py \\
    --samples   data/GQA_3500.json \\
    --image_dir /path/to/gqa/images \\
    --model_id  llava-hf/llava-1.5-7b-hf \\
    --device cuda:0 --max_per_op 500
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from HookedLLaVA import HookedLLaVA
from llava_causal_mean_ablation import (
    build_inputs,
    get_vision_indices,
    get_answer_tokens,
    score_answer,
    model_correct as compute_correct,
    extract_program_objects,
)

from attn_knockout import (
    load_samples,
    select_balanced,
    safe_mean,
    cohen_d,
    _bh_correct,
    text_dims,
)

# LLaVA-1.5-7B defaults (LLaMA-2-7B backbone); run() resets them from the config.
N_LAYERS = 32
N_HEADS  = 32
HEAD_DIM = 128   # 4096 / 32

# LLaVA-1.5 vision grid: CLIP ViT-L/14 @ 336px → 24×24 patches
_LLAVA_GRID = 24


def _per_layer_mw(c_raw, i_raw):
    """Mann-Whitney per layer. Uses this module's N_LAYERS (set from the LLaVA config)."""
    if not c_raw or not i_raw:
        return [None] * N_LAYERS

    def _nan(curve):
        return [np.nan if v is None else v for v in curve]

    c_arr = np.array([_nan(c) for c in c_raw])
    i_arr = np.array([_nan(c) for c in i_raw])
    pvals = []
    for L in range(N_LAYERS):
        c_l = c_arr[:, L]; c_l = c_l[~np.isnan(c_l)]
        i_l = i_arr[:, L]; i_l = i_l[~np.isnan(i_l)]
        if len(c_l) >= 3 and len(i_l) >= 3:
            _, p = mannwhitneyu(c_l, i_l, alternative="two-sided")
            pvals.append(float(p))
        else:
            pvals.append(None)
    return pvals


# ── bbox helpers ──────────────────────────────────────────────────

def _bbox_to_llava_positions(image: Image.Image, bbox, vis_idx: List[int]) -> List[int]:
    """
    Map [x, y, w, h] (GQA pixel coords) to absolute vision token positions
    in the LLaVA expanded sequence.

    LLaVA-1.5 resizes every image to 336×336 before CLIP encoding, producing
    a 24×24 = 576 patch grid.  We scale the bbox from the original image size
    to that grid, then look up the corresponding vis_idx entries.
    """
    if not vis_idx or bbox is None:
        return []
    img_w, img_h = image.size
    x, y, w, h = bbox
    PATCH = 336 / _LLAVA_GRID  # 14.0 px per patch
    x1 = x / img_w * 336;  y1 = y / img_h * 336
    x2 = (x + w) / img_w * 336;  y2 = (y + h) / img_h * 336
    c0 = max(0, int(x1 / PATCH))
    c1 = min(_LLAVA_GRID, int(np.ceil(x2 / PATCH)))   # exclusive end (matches ablation file)
    r0 = max(0, int(y1 / PATCH))
    r1 = min(_LLAVA_GRID, int(np.ceil(y2 / PATCH)))
    offsets = [r * _LLAVA_GRID + c for r in range(r0, r1) for c in range(c0, c1)]
    if not offsets or max(offsets) >= len(vis_idx):
        return []
    return sorted(set(vis_idx[o] for o in offsets))


def _get_bbox_positions(
    sample: dict,
    image: Image.Image,
    vis_idx: List[int],
    image_objects: Optional[dict],
) -> List[int]:
    if not image_objects:
        return []
    program = sample.get("semantic_program", [])
    if not program:
        return []
    groups = extract_program_objects(
        program, image_objects, sample_operation=sample.get("operation", "")
    )
    if not groups:
        return []
    all_pos: List[int] = []
    for group in groups:
        for obj in group["objects"]:
            abs_pos = _bbox_to_llava_positions(image, obj["bbox"], vis_idx)
            all_pos.extend(abs_pos)
    return sorted(set(all_pos))


# ── Attention Knockout ────────────────────────────────────────────

class HeadAblationHook:
    """
    Zeros head H's slice of the o_proj input at ans_pos.
    Compatible with eager, sdpa, and flash attention.
    """
    def __init__(self, head_idx: int, ans_pos: int):
        self.head_idx = head_idx
        self.ans_pos  = ans_pos
        self.lo = head_idx * HEAD_DIM
        self.hi = self.lo + HEAD_DIM

    def __call__(self, module, args):
        x = args[0]
        if x is None or x.shape[1] <= self.ans_pos:
            return args
        x = x.clone()
        x[:, self.ans_pos, self.lo : self.hi] = 0
        return (x,)


def _run_attention_knockout_stage_a_impl(
    vlm: HookedLLaVA,
    inputs: dict,
    ans_pos: int,
    bbox_positions: List[int],
    clean_logp: float,
    ans_toks: List[int],
) -> Optional[dict]:
    """
    32 forward passes (one per layer). True attention knockout via BlockAttentionHook.
    Requires attn_implementation="eager".
    """
    if not bbox_positions:
        return None

    pairs = [(ans_pos, bp) for bp in bbox_positions]
    ko_curve: List[Optional[float]] = []

    for L in range(N_LAYERS):
        try:
            with vlm.block_attention({L: pairs}):
                ko_lp = score_answer(vlm, inputs, ans_toks)
            ko_curve.append(float(clean_logp - ko_lp))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            ko_curve.append(None)
        except Exception as e:
            print(f"    [WARN] Stage A layer {L} failed: {e}")
            ko_curve.append(None)

    return {"ko_curve": ko_curve}


def _run_head_ablation_impl(
    vlm: HookedLLaVA,
    inputs: dict,
    ans_pos: int,
    clean_logp: float,
    ans_toks: List[int],
) -> dict:
    """
    Head ablation grid — 32 batched forward passes (batch_size=N_HEADS=32).
    Expands inputs to batch_size=32 and zeros head H's o_proj slice for
    batch item H, replacing 1024 sequential passes with 32 batched ones.
    """
    layers = vlm._get_text_layers()
    prompt_len = inputs["inputs_embeds"].shape[1]

    batch_inputs: dict = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == 1:
            batch_inputs[k] = v.expand(N_HEADS, *v.shape[1:]).contiguous()
        else:
            batch_inputs[k] = v

    def _batch_hook(_module, args):
        x = args[0].clone()
        for H in range(N_HEADS):
            x[H, ans_pos, H * HEAD_DIM : (H + 1) * HEAD_DIM] = 0
        return (x,)

    matrix: List[List[Optional[float]]] = []

    for L in range(N_LAYERS):
        handle = layers[L].self_attn.o_proj.register_forward_pre_hook(_batch_hook)
        try:
            with torch.no_grad():
                out = vlm.model(**batch_inputs)
            lp = F.log_softmax(out.logits.float(), dim=-1)
            row: List[Optional[float]] = []
            for H in range(N_HEADS):
                total, count = 0.0, 0
                for i, tok in enumerate(ans_toks):
                    idx = prompt_len - 1 + i
                    if 0 <= idx < lp.shape[1]:
                        total += lp[H, idx, tok].item()
                        count += 1
                row.append(float(clean_logp - total / max(count, 1)))
            matrix.append(row)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            matrix.append([None] * N_HEADS)
        except Exception as e:
            print(f"    [WARN] batched head-ablation L{L}: {e}")
            matrix.append([None] * N_HEADS)
        finally:
            handle.remove()

    return {"head_ko_matrix": matrix}


# ── Per-sample runner ─────────────────────────────────────────────

def analyze_sample(
    vlm: HookedLLaVA,
    sample: dict,
    image_objects: Optional[dict] = None,
    run_knockout: bool = True,
    head_level: bool = False,
) -> Optional[dict]:
    question = sample["question_clean"]
    answer   = sample["answer"]
    img_path = sample.get("image_path", "")
    if not img_path or not os.path.exists(img_path):
        return None

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        return None

    if answer.strip().lower() in ("yes", "no"):
        question = question + " Answer with yes or no."
    elif "have in common" in question.lower():
        question = question + " Name the shared attribute."

    inputs  = build_inputs(vlm, image, question)
    vis_idx = get_vision_indices(vlm, inputs)
    if not vis_idx:
        return None

    ans_toks = get_answer_tokens(vlm, answer)
    if not ans_toks:
        return None

    correct, model_response = compute_correct(vlm, inputs, answer)

    bbox_positions = _get_bbox_positions(sample, image, vis_idx, image_objects)

    # Clean pass — capture layer-0 hidden states to skip vision encoder on
    # subsequent passes. ans_pos is derived from the *actual* sequence length
    # seen by the transformer (which may differ from the formula depending on
    # how this transformers version handles LLaVA vision expansion).
    _emb: dict = {}

    def _cap(module, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is not None and "e" not in _emb:
            _emb["e"] = hs.detach().clone()

    _cap_h = vlm._get_text_layers()[0].register_forward_pre_hook(_cap, with_kwargs=True)
    clean_logp = score_answer(vlm, inputs, ans_toks)
    _cap_h.remove()

    if "e" in _emb:
        actual_seq_len = _emb["e"].shape[1]
        ans_pos = actual_seq_len - 1
        lean_inputs = {
            "inputs_embeds": _emb["e"],
            "attention_mask": torch.ones(
                1, actual_seq_len,
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            ),
        }
    else:
        ans_pos = inputs["input_ids"].shape[1] - 1
        lean_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}

    knockout = None
    if run_knockout and bbox_positions:
        stage_a = _run_attention_knockout_stage_a_impl(
            vlm, lean_inputs, ans_pos, bbox_positions, float(clean_logp), ans_toks
        )
        knockout = stage_a

        if head_level and stage_a is not None:
            stage_b = _run_head_ablation_impl(
                vlm, lean_inputs, ans_pos, float(clean_logp), ans_toks
            )
            knockout = {**stage_a, **stage_b}

    return {
        "question_id":    sample["question_id"],
        "operation":      sample.get("operation", ""),
        "question":       question,
        "answer":         answer,
        "model_response": model_response,
        "model_correct":  correct,
        "clean_logp":     float(clean_logp),
        "knockout":       knockout,
    }


# ── Aggregation ───────────────────────────────────────────────────

def _gss_curve(c_curves, i_curves):
    def layer_mean(curves, L):
        vals = [c[L] for c in curves if c[L] is not None]
        return float(np.mean(vals)) if vals else None
    return [
        (lambda mc, mi: mc - mi if mc is not None and mi is not None else None)(
            layer_mean(c_curves, L), layer_mean(i_curves, L)
        )
        for L in range(N_LAYERS)
    ]


def aggregate_knockout(results: List[dict]) -> dict:
    by_op: Dict[str, dict] = defaultdict(
        lambda: {
            "correct":   {"ko": [], "head_matrices": []},
            "incorrect": {"ko": [], "head_matrices": []},
        }
    )
    for r in results:
        ko = r.get("knockout")
        if not ko:
            continue
        op  = r["operation"]
        key = "correct" if r["model_correct"] else "incorrect"
        ko_curve = ko.get("ko_curve")
        if ko_curve:
            by_op[op][key]["ko"].append(ko_curve)
        head_mat = ko.get("head_ko_matrix")
        if head_mat and len(head_mat) == N_LAYERS and len(head_mat[0]) == N_HEADS:
            by_op[op][key]["head_matrices"].append(head_mat)

    def _to_nan(curve):
        return [np.nan if v is None else v for v in curve]

    def _curve_stats(raw_curves):
        if not raw_curves:
            return None, None
        arr = np.array([_to_nan(c) for c in raw_curves])
        return np.nanmean(arr, axis=0).tolist(), np.nanstd(arr, axis=0).tolist()

    def _peak_ko(raw_curves):
        vals = []
        for c in raw_curves:
            valid = [v for v in c if v is not None]
            if valid:
                vals.append(max(valid))
        return vals

    def _argmax_mean(mean_curve):
        if not mean_curve:
            return None
        valid = [(L, v) for L, v in enumerate(mean_curve) if v is not None and not np.isnan(v)]
        return max(valid, key=lambda x: x[1])[0] if valid else None

    out = {}
    for op, grps in by_op.items():
        op_out = {}
        for key in ("correct", "incorrect"):
            ko_mean, ko_std = _curve_stats(grps[key]["ko"])
            op_out[key] = {
                "n":             len(grps[key]["ko"]),
                "mean_ko_curve": ko_mean,
                "std_ko_curve":  ko_std,
            }

        c_peak = _peak_ko(grps["correct"]["ko"])
        i_peak = _peak_ko(grps["incorrect"]["ko"])
        p_peak = None
        if len(c_peak) >= 3 and len(i_peak) >= 3:
            _, p_peak = mannwhitneyu(c_peak, i_peak, alternative="two-sided")

        c_mean_curve, _ = _curve_stats(grps["correct"]["ko"])
        i_mean_curve, _ = _curve_stats(grps["incorrect"]["ko"])

        gss = _gss_curve(grps["correct"]["ko"], grps["incorrect"]["ko"])
        peak_gss_layer = None
        valid_gss = [(L, v) for L, v in enumerate(gss) if v is not None]
        if valid_gss:
            peak_gss_layer = max(valid_gss, key=lambda x: abs(x[1]))[0]

        pl_p_ko    = _per_layer_mw(grps["correct"]["ko"], grps["incorrect"]["ko"])
        pl_p_ko_bh = _bh_correct(pl_p_ko)
        n_sig_ko   = sum(1 for p in pl_p_ko_bh if p is not None and p < 0.05)

        head_agg = None
        all_mats_c = grps["correct"]["head_matrices"]
        all_mats_i = grps["incorrect"]["head_matrices"]
        if all_mats_c or all_mats_i:
            def _mat_mean(mats):
                if not mats:
                    return None
                arr = np.array(mats, dtype=float)
                return np.nanmean(arr, axis=0).tolist()

            mean_mat_c = _mat_mean(all_mats_c)
            mean_mat_i = _mat_mean(all_mats_i)

            def _top_heads(mat, n=10):
                if mat is None:
                    return []
                flat = [
                    (L, H, mat[L][H])
                    for L in range(N_LAYERS)
                    for H in range(N_HEADS)
                    if mat[L][H] is not None and not np.isnan(mat[L][H])
                ]
                flat.sort(key=lambda x: x[2], reverse=True)
                return [(L, H, round(v, 5)) for L, H, v in flat[:n]]

            def _per_head_cohen_d(mats_c, mats_i):
                if not mats_c or not mats_i:
                    return None
                arr_c = np.array(mats_c, dtype=float)
                arr_i = np.array(mats_i, dtype=float)
                peak_c = np.nanmax(arr_c, axis=1)
                peak_i = np.nanmax(arr_i, axis=1)
                return [cohen_d(peak_c[:, H].tolist(), peak_i[:, H].tolist())
                        for H in range(N_HEADS)]

            head_agg = {
                "n_correct":             len(all_mats_c),
                "n_incorrect":           len(all_mats_i),
                "mean_matrix_correct":   mean_mat_c,
                "mean_matrix_incorrect": mean_mat_i,
                "top10_heads_correct":   _top_heads(mean_mat_c),
                "top10_heads_incorrect": _top_heads(mean_mat_i),
                "per_head_cohen_d":      _per_head_cohen_d(all_mats_c, all_mats_i),
            }

        out[op] = {
            **op_out,
            "mean_peak_ko_correct":    safe_mean(c_peak),
            "mean_peak_ko_incorrect":  safe_mean(i_peak),
            "cohen_d_peak_ko":         cohen_d(c_peak, i_peak),
            "mannwhitney_p_peak_ko":   float(p_peak) if p_peak is not None else None,
            "peak_ko_layer_correct":   _argmax_mean(c_mean_curve),
            "peak_ko_layer_incorrect": _argmax_mean(i_mean_curve),
            "gss":                     gss,
            "peak_gss_layer":          peak_gss_layer,
            "per_layer_p_ko":          pl_p_ko,
            "per_layer_p_ko_bh":       pl_p_ko_bh,
            "n_sig_layers_ko":         n_sig_ko,
            "head_level":              head_agg,
        }
    return out


# ── Print summary ─────────────────────────────────────────────────

def print_summary(ko_agg: Optional[dict]):
    def _f(x):
        return f"{x:.4f}" if x is not None else "N/A"

    def _sig(p):
        return "***" if p and p < 0.001 else "**" if p and p < 0.01 \
               else "*" if p and p < 0.05 else "ns"

    print("\n" + "=" * 80)

    if ko_agg:
        print("Attention Knockout (Layer-Level):")
        print(f"  {'op':12s}  {'n_c':>5}  {'n_i':>5}  "
              f"{'KO_c':>8}  {'KO_i':>8}  {'cohen_d':>8}  "
              f"{'p_KO':>10}  {'peak_L_c':>9}  {'peak_L_i':>9}  "
              f"{'sig_L':>5}  {'GSS_dir':>7}")
        for op, d in sorted(ko_agg.items()):
            nc  = d.get("correct",   {}).get("n", 0)
            ni  = d.get("incorrect", {}).get("n", 0)
            p   = d.get("mannwhitney_p_peak_ko")
            cd  = d.get("cohen_d_peak_ko")
            gss = d.get("gss", [])
            peak_gss = d.get("peak_gss_layer")
            if peak_gss is not None and gss and gss[peak_gss] is not None:
                gss_dir = "↑" if gss[peak_gss] > 0 else ("↓" if gss[peak_gss] < 0 else "=")
            else:
                gss_dir = "="
            print(f"  {op:12s}  {nc:>5}  {ni:>5}  "
                  f"{_f(d.get('mean_peak_ko_correct')):>8}  "
                  f"{_f(d.get('mean_peak_ko_incorrect')):>8}  "
                  f"{_f(cd):>8}  "
                  f"{_f(p):>8} {_sig(p):3s}  "
                  f"{str(d.get('peak_ko_layer_correct', 'N/A')):>9}  "
                  f"{str(d.get('peak_ko_layer_incorrect', 'N/A')):>9}  "
                  f"{str(d.get('n_sig_layers_ko', 'N/A')):>5}  "
                  f"{gss_dir:>7}")

        has_head = any(d.get("head_level") for d in ko_agg.values())
        if has_head:
            print("\n" + "-" * 80)
            print("Head-Level Attribution (top-10 heads by mean degradation on correct samples):")
            print(f"  {'op':12s}  {'rank':>4}  {'L':>3}  {'H':>3}  {'mean_deg':>10}  {'cohen_d':>8}")
            for op, d in sorted(ko_agg.items()):
                hl = d.get("head_level")
                if not hl:
                    continue
                tops = hl.get("top10_heads_correct", [])
                cd_ph = hl.get("per_head_cohen_d") or []
                for rank, (L, H, v) in enumerate(tops, 1):
                    cd_val = cd_ph[H] if H < len(cd_ph) else None
                    print(f"  {op if rank==1 else '':12s}  {rank:>4}  {L:>3}  {H:>3}  "
                          f"{v:>10.5f}  {_f(cd_val):>8}")
                if cd_ph:
                    ranked_cd = sorted(
                        [(H, cd_ph[H]) for H in range(len(cd_ph)) if cd_ph[H] is not None],
                        key=lambda x: abs(x[1]) if x[1] is not None else 0, reverse=True
                    )[:5]
                    top_cd_str = "  ".join(f"H{H}(d={v:+.2f})" for H, v in ranked_cd)
                    print(f"  {'':12s}  Top discriminative heads: {top_cd_str}")


# ── Main ──────────────────────────────────────────────────────────

def run(
    samples_path: str,
    image_dir: Optional[str],
    model_id: str,
    output_dir: str,
    device: str,
    max_per_op: int,
    ops: List[str],
    seed: int,
    run_knockout: bool,
    head_level: bool,
):
    global N_LAYERS, N_HEADS, HEAD_DIM
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading samples from {samples_path} ...")
    samples, image_objects_map = load_samples(samples_path, image_dir)
    selected = select_balanced(samples, ops, max_per_op)
    print(f"  {len(selected)} samples  ({ops})")
    print(f"  run_knockout={run_knockout}  head_level={head_level}")

    print(f"\nLoading {model_id} on {device} ...")
    vlm = HookedLLaVA(model_id=model_id, device=device, quantize=False, attn_implementation="eager")
    vlm.model.eval()
    N_LAYERS, N_HEADS, HEAD_DIM = text_dims(vlm.model)
    print(f"  {N_LAYERS} layers, {N_HEADS} heads, head_dim {HEAD_DIM}")

    samples_path_out = os.path.join(output_dir, "per_sample.jsonl")

    def _default(o):
        if isinstance(o, np.ndarray):  return o.tolist()
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer):  return int(o)
        return None

    results = []
    with open(samples_path_out, "w") as jsonl_f:
        for s in tqdm(selected, desc="Analysing"):
            img_objects = image_objects_map.get(s.get("image_id", ""), {})
            try:
                r = analyze_sample(
                    vlm, s,
                    image_objects=img_objects,
                    run_knockout=run_knockout,
                    head_level=head_level,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            except Exception as e:
                print(f"  [ERR] {s.get('question_id')}: {e}")
                import traceback; traceback.print_exc()
                continue
            if r is None:
                continue
            results.append(r)
            jsonl_f.write(json.dumps(r, default=_default) + "\n")
            jsonl_f.flush()
            torch.cuda.empty_cache()

    print(f"\n{len(results)} samples collected")
    n_correct = sum(1 for r in results if r["model_correct"])
    print(f"  model accuracy: {n_correct}/{len(results)} = "
          f"{n_correct / max(len(results), 1):.3f}")

    by_op_acc: Dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        op = r["operation"]
        by_op_acc[op]["total"] += 1
        if r["model_correct"]:
            by_op_acc[op]["correct"] += 1
    print("\nPer-operation accuracy:")
    for op in sorted(by_op_acc):
        c = by_op_acc[op]["correct"]
        t = by_op_acc[op]["total"]
        print(f"  {op:12s}: {c}/{t} = {c/max(t,1):.3f}")

    ko_agg = aggregate_knockout(results) if run_knockout else None

    print_summary(ko_agg)

    out = {
        "config": {
            "model_id":     model_id,
            "run_knockout": run_knockout,
            "head_level":   head_level,
        },
        "knockout_by_op": ko_agg,
    }
    agg_path = os.path.join(output_dir, "attention_results.json")
    with open(agg_path, "w") as f:
        json.dump(out, f, indent=2, default=_default)
    print(f"\nSaved → {agg_path}")
    print(f"Saved → {samples_path_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples",    required=True)
    ap.add_argument("--image_dir",  default="data/images",
                    help="Folder holding the images named in --samples")
    ap.add_argument("--model_id",   default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--output_dir", default="results/llava_attn_knockout")
    ap.add_argument("--device",     default="cuda:0")
    ap.add_argument("--max_per_op", type=int, default=500)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--ops",        nargs="+",
                    default=["relate", "select", "exist", "verify", "filter", "choose", "query"])
    ap.add_argument("--knockout",    dest="run_knockout", action="store_true",  default=True)
    ap.add_argument("--no-knockout", dest="run_knockout", action="store_false")
    ap.add_argument("--head_level",  dest="head_level",   action="store_true",  default=False)
    args = ap.parse_args()

    run(
        samples_path = args.samples,
        image_dir    = args.image_dir,
        model_id     = args.model_id,
        output_dir   = args.output_dir,
        device       = args.device,
        max_per_op   = args.max_per_op,
        ops          = args.ops,
        seed         = args.seed,
        run_knockout = args.run_knockout,
        head_level   = args.head_level,
    )


if __name__ == "__main__":
    main()
