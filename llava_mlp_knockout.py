#!/usr/bin/env python3
"""
llava_mlp_knockout.py — MLP Knockout (LLaVA-1.5)
=================================================
Mirror of mlp_knockout.py for LLaVA-1.5-7B.

Two MLP knockout variants run together for all 32 layers.

Variant A — bbox MLP knockout:
  At layer L, replace MLP output at bbox token positions with the mean
  MLP output across all vision token positions at that layer.

Variant B — answer token MLP knockout:
  At layer L, replace MLP output at the answer token position with the mean
  MLP output across all text token positions (non-vision, non-pad).

Structure per sample:
  1 clean pass + 64 MLP passes (32 layers × 2 variants)

Architecture (LLaVA-1.5-7B, LLaMA-2-7B backbone):
  layers = model.language_model.model.layers  (len=32)
  layer.mlp                                    the MLP module
  Vision: CLIP ViT-L/14 @ 336px → 24×24 = 576 fixed tokens

Usage
-----
python llava_mlp_knockout.py \\
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

# Reuse LLaVA bbox helpers from llava_attn_knockout
from llava_attn_knockout import _get_bbox_positions

# LLaVA-1.5-7B default; run() resets it from the config.
N_LAYERS = 32


# ── local _per_layer_mw (uses this module's N_LAYERS) ──

def _per_layer_mw(c_raw, i_raw):
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


# ── index helpers ─────────────────────────────────────────────────

def _get_text_indices(vlm, inputs) -> List[int]:
    """
    All sequence positions that are neither the image placeholder nor padding.
    For LLaVA, the image placeholder token id is config.image_token_index (32000).
    These are used as the neutral baseline for Variant B.
    """
    input_ids  = inputs["input_ids"][0]
    img_tok_id = vlm.model.config.image_token_index   # 32000
    pad_id     = vlm.processor.tokenizer.pad_token_id
    mask = (input_ids != img_tok_id)
    if pad_id is not None:
        mask = mask & (input_ids != pad_id)
    return torch.where(mask)[0].tolist()


# ── MLP post-hooks ────────────────────────────────────────────────

def _make_bbox_mlp_hook(vision_indices: List[int], bbox_indices: List[int]):
    """
    Variant A: replace MLP output at bbox token positions with the mean MLP
    output across all vision token positions at that layer.
    """
    vis_t = torch.tensor(vision_indices, dtype=torch.long)

    def hook(module, input, output):
        out = output.clone()
        mlp_mean = out[0, vis_t, :].mean(dim=0)
        out[0, bbox_indices, :] = mlp_mean
        return out

    return hook


def _make_answer_mlp_hook(text_indices: List[int], answer_pos: int):
    """
    Variant B: replace MLP output at the answer token position with the mean
    MLP output across all text token positions (non-vision, non-pad).
    """
    txt_t = torch.tensor(text_indices, dtype=torch.long)

    def hook(module, input, output):
        out = output.clone()
        mlp_mean = out[0, txt_t, :].mean(dim=0)
        out[0, answer_pos, :] = mlp_mean
        return out

    return hook


# ── MLP knockout runner ───────────────────────────────────────────

def _run_mlp_knockout(
    vlm: HookedLLaVA,
    inputs: dict,
    vision_indices: List[int],
    bbox_indices: List[int],
    text_indices: List[int],
    answer_pos: int,
    ans_toks: List[int],
    clean_logp: float,
) -> dict:
    """
    64 forward passes: 32 layers × 2 variants (A=bbox, B=answer).

    Returns:
        mlp_ko_bbox:   list[Optional[float]] len=32  (variant A degradations)
        mlp_ko_answer: list[Optional[float]] len=32  (variant B degradations)
    """
    layers = vlm._get_text_layers()
    ko_bbox:   List[Optional[float]] = []
    ko_answer: List[Optional[float]] = []

    for L in range(N_LAYERS):
        # ── Variant A ──────────────────────────────────────────────
        if bbox_indices:
            hook_fn = _make_bbox_mlp_hook(vision_indices, bbox_indices)
            handle  = layers[L].mlp.register_forward_hook(hook_fn)
            try:
                ko_lp = score_answer(vlm, inputs, ans_toks)
                ko_bbox.append(float(clean_logp - ko_lp))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                ko_bbox.append(None)
            except Exception as e:
                print(f"    [WARN] MLP bbox KO L{L}: {e}")
                ko_bbox.append(None)
            finally:
                handle.remove()
        else:
            ko_bbox.append(None)

        # ── Variant B ──────────────────────────────────────────────
        if text_indices:
            hook_fn = _make_answer_mlp_hook(text_indices, answer_pos)
            handle  = layers[L].mlp.register_forward_hook(hook_fn)
            try:
                ko_lp = score_answer(vlm, inputs, ans_toks)
                ko_answer.append(float(clean_logp - ko_lp))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                ko_answer.append(None)
            except Exception as e:
                print(f"    [WARN] MLP answer KO L{L}: {e}")
                ko_answer.append(None)
            finally:
                handle.remove()
        else:
            ko_answer.append(None)

    return {"mlp_ko_bbox": ko_bbox, "mlp_ko_answer": ko_answer}


# ── Peak helpers ──────────────────────────────────────────────────

def _peak_val(curve: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in curve if v is not None]
    return float(max(valid)) if valid else None


def _peak_layer(curve: List[Optional[float]]) -> Optional[int]:
    valid = [(i, v) for i, v in enumerate(curve) if v is not None]
    return max(valid, key=lambda x: x[1])[0] if valid else None


# ── Per-sample runner ─────────────────────────────────────────────

def analyze_sample(
    vlm: HookedLLaVA,
    sample: dict,
    image_objects: Optional[dict] = None,
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
    text_indices   = _get_text_indices(vlm, inputs)

    # Clean pass — capture layer-0 hidden states to skip vision encoder on
    # subsequent passes. ans_pos derived from actual sequence length.
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

    # Filter indices to valid range for lean_inputs sequence length
    actual_seq_len = lean_inputs.get("inputs_embeds", inputs["attention_mask"]).shape[1]
    vis_idx_valid  = [i for i in vis_idx        if i < actual_seq_len]
    bbox_valid     = [i for i in bbox_positions  if i < actual_seq_len]
    text_valid     = [i for i in text_indices    if i < actual_seq_len]

    mlp_results = _run_mlp_knockout(
        vlm, lean_inputs,
        vision_indices=vis_idx_valid,
        bbox_indices=bbox_valid,
        text_indices=text_valid,
        answer_pos=ans_pos,
        ans_toks=ans_toks,
        clean_logp=float(clean_logp),
    )

    ko_bbox   = mlp_results["mlp_ko_bbox"]
    ko_answer = mlp_results["mlp_ko_answer"]

    return {
        "question_id":           sample["question_id"],
        "operation":             sample.get("operation", ""),
        "question":              question,
        "answer":                answer,
        "model_response":        model_response,
        "model_correct":         correct,
        "clean_logp":            float(clean_logp),
        "mlp_ko_bbox":           ko_bbox,
        "mlp_ko_answer":         ko_answer,
        "mlp_peak_bbox":         _peak_val(ko_bbox),
        "mlp_peak_answer":       _peak_val(ko_answer),
        "mlp_peak_layer_bbox":   _peak_layer(ko_bbox),
        "mlp_peak_layer_answer": _peak_layer(ko_answer),
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


def _peak_ko_per_sample(raw_curves: List[List[Optional[float]]]) -> List[float]:
    out = []
    for c in raw_curves:
        valid = [v for v in c if v is not None]
        if valid:
            out.append(max(valid))
    return out


def _argmax_mean(mean_curve: Optional[List]) -> Optional[int]:
    if not mean_curve:
        return None
    valid = [(L, v) for L, v in enumerate(mean_curve)
             if v is not None and not np.isnan(v)]
    return max(valid, key=lambda x: x[1])[0] if valid else None


def _curve_stats(raw_curves: List[List[Optional[float]]]):
    if not raw_curves:
        return None, None
    arr = np.array([[np.nan if v is None else v for v in c] for c in raw_curves])
    return np.nanmean(arr, axis=0).tolist(), np.nanstd(arr, axis=0).tolist()


def _mw_peaks(c_peak, i_peak):
    if len(c_peak) >= 3 and len(i_peak) >= 3:
        _, p = mannwhitneyu(c_peak, i_peak, alternative="two-sided")
        return float(p)
    return None


def aggregate_mlp(results: List[dict]) -> dict:
    by_op: Dict[str, dict] = defaultdict(lambda: {
        "correct":   {"bbox": [], "answer": []},
        "incorrect": {"bbox": [], "answer": []},
    })

    for r in results:
        op  = r["operation"]
        key = "correct" if r["model_correct"] else "incorrect"
        if r.get("mlp_ko_bbox"):
            by_op[op][key]["bbox"].append(r["mlp_ko_bbox"])
        if r.get("mlp_ko_answer"):
            by_op[op][key]["answer"].append(r["mlp_ko_answer"])

    out = {}
    for op, grps in by_op.items():
        op_out: dict = {}

        for variant, label in [("bbox", "bbox"), ("answer", "answer")]:
            c_curves = grps["correct"][variant]
            i_curves = grps["incorrect"][variant]

            c_mean, c_std = _curve_stats(c_curves)
            i_mean, i_std = _curve_stats(i_curves)

            c_peak = _peak_ko_per_sample(c_curves)
            i_peak = _peak_ko_per_sample(i_curves)
            p_peak = _mw_peaks(c_peak, i_peak)

            gss = _gss_curve(c_curves, i_curves)
            valid_gss = [(L, v) for L, v in enumerate(gss) if v is not None]
            peak_gss_layer = (
                max(valid_gss, key=lambda x: abs(x[1]))[0] if valid_gss else None
            )

            pl_p    = _per_layer_mw(c_curves, i_curves)
            pl_p_bh = _bh_correct(pl_p)
            n_sig   = sum(1 for p in pl_p_bh if p is not None and p < 0.05)

            op_out[f"variant_{label}"] = {
                "n_correct":               len(c_curves),
                "n_incorrect":             len(i_curves),
                "mean_ko_curve_correct":   c_mean,
                "std_ko_curve_correct":    c_std,
                "mean_ko_curve_incorrect": i_mean,
                "std_ko_curve_incorrect":  i_std,
                "mean_peak_correct":       safe_mean(c_peak),
                "mean_peak_incorrect":     safe_mean(i_peak),
                "cohen_d_peak":            cohen_d(c_peak, i_peak),
                "mannwhitney_p_peak":      p_peak,
                "peak_layer_correct":      _argmax_mean(c_mean),
                "peak_layer_incorrect":    _argmax_mean(i_mean),
                "gss_curve":               gss,
                "peak_gss_layer":          peak_gss_layer,
                "per_layer_p":             pl_p,
                "per_layer_p_bh":          pl_p_bh,
                "n_sig_layers":            n_sig,
            }

        out[op] = op_out

    return out


# ── Print summary ─────────────────────────────────────────────────

def print_summary(agg: Optional[dict]):
    def _f(x):
        return f"{x:.4f}" if x is not None else "N/A"

    def _sig(p):
        return "***" if p and p < 0.001 else "**" if p and p < 0.01 \
               else "*" if p and p < 0.05 else "ns"

    print("\n" + "=" * 90)
    print("MLP Knockout Summary  (LLaVA-1.5-7B)")
    print("=" * 90)

    for variant, label in [("bbox",   "Variant A — bbox MLP KO"),
                            ("answer", "Variant B — answer MLP KO")]:
        print(f"\n{label}:")
        print(f"  {'op':12s}  {'n_c':>5}  {'n_i':>5}  "
              f"{'peak_c':>8}  {'peak_i':>8}  {'cohen_d':>8}  "
              f"{'p_peak':>10}  {'peakL_c':>8}  {'peakL_i':>8}  "
              f"{'sig_L':>5}  {'GSS_dir':>7}")
        if not agg:
            continue
        for op, d in sorted(agg.items()):
            v = d.get(f"variant_{variant}", {})
            if not v:
                continue
            p        = v.get("mannwhitney_p_peak")
            cd       = v.get("cohen_d_peak")
            gss      = v.get("gss_curve", [])
            gss_peak = v.get("peak_gss_layer")
            if gss_peak is not None and gss and gss[gss_peak] is not None:
                gss_dir = "↑" if gss[gss_peak] > 0 else ("↓" if gss[gss_peak] < 0 else "=")
            else:
                gss_dir = "="
            print(f"  {op:12s}  "
                  f"{v.get('n_correct',   0):>5}  "
                  f"{v.get('n_incorrect', 0):>5}  "
                  f"{_f(v.get('mean_peak_correct')):>8}  "
                  f"{_f(v.get('mean_peak_incorrect')):>8}  "
                  f"{_f(cd):>8}  "
                  f"{_f(p):>8} {_sig(p):3s}  "
                  f"{str(v.get('peak_layer_correct',   'N/A')):>8}  "
                  f"{str(v.get('peak_layer_incorrect', 'N/A')):>8}  "
                  f"{str(v.get('n_sig_layers', 'N/A')):>5}  "
                  f"{gss_dir:>7}")


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
):
    global N_LAYERS
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading samples from {samples_path} ...")
    samples, image_objects_map = load_samples(samples_path, image_dir)
    selected = select_balanced(samples, ops, max_per_op)
    print(f"  {len(selected)} samples  ({ops})")

    print(f"\nLoading {model_id} on {device} ...")
    vlm = HookedLLaVA(model_id=model_id, device=device, quantize=False)
    vlm.model.eval()
    N_LAYERS = text_dims(vlm.model)[0]
    print(f"  {2 * N_LAYERS} MLP passes per sample ({N_LAYERS} layers × 2 variants)")

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
                r = analyze_sample(vlm, s, image_objects=img_objects)
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

    agg = aggregate_mlp(results)
    print_summary(agg)

    out = {
        "config": {
            "model_id":   model_id,
            "max_per_op": max_per_op,
            "ops":        ops,
            "seed":       seed,
        },
        "mlp_by_op": agg,
    }
    agg_path = os.path.join(output_dir, "mlp_results.json")
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
    ap.add_argument("--output_dir", default="results/llava_mlp_knockout")
    ap.add_argument("--device",     default="cuda:0")
    ap.add_argument("--max_per_op", type=int, default=500)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--ops",        nargs="+",
                    default=["relate", "select", "exist", "verify",
                             "filter", "choose", "query"])
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
    )


if __name__ == "__main__":
    main()
