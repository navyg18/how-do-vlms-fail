#!/usr/bin/env python3

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from HookedQwenVL import HookedQwenVL
from bbox_utils import bbox_to_token_positions
from causal_mean_ablation import (
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
    _get_bbox_positions,
)

# Qwen2.5-VL-3B default; run() resets it from the loaded model.
N_LAYERS = 36


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


# ── index helpers ───────────────────────────────────────────────

def _get_text_indices(vlm, inputs) -> List[int]:
    """Return token indices corresponding to text (non-vision, non-pad)."""
    # Kept exactly as in the paper's runs: Qwen's image tokens are <|image_pad|>,
    # not <|vision_pad|>, so this mean also covers the image positions.
    input_ids = inputs["input_ids"][0]
    pad_id = vlm.processor.tokenizer.pad_token_id
    vis_id = vlm.processor.tokenizer.convert_tokens_to_ids("<|vision_pad|>")

    mask = (input_ids != vis_id)
    if pad_id is not None:
        mask = mask & (input_ids != pad_id)

    return torch.where(mask)[0].tolist()


# ── MLP hooks ───────────────────────────────────────────────────

def _make_bbox_mlp_hook(vision_indices: List[int], bbox_indices: List[int]):
    """Replace bbox token MLP outputs with mean over vision tokens."""
    vis_t = torch.tensor(vision_indices, dtype=torch.long)

    def hook(module, input, output):
        out = output.clone()
        mean_vec = out[0, vis_t, :].mean(dim=0)
        out[0, bbox_indices, :] = mean_vec
        return out

    return hook


def _make_answer_mlp_hook(text_indices: List[int], answer_pos: int):
    """Replace answer token MLP output with mean over text tokens."""
    txt_t = torch.tensor(text_indices, dtype=torch.long)

    def hook(module, input, output):
        out = output.clone()
        mean_vec = out[0, txt_t, :].mean(dim=0)
        out[0, answer_pos, :] = mean_vec
        return out

    return hook


# ── knockout runner ──────────────────────────────────────────────

def _run_mlp_knockout(
    vlm,
    inputs,
    vision_indices,
    bbox_indices,
    text_indices,
    answer_pos,
    ans_toks,
    clean_logp,
):
    """Run both MLP knockout variants across all layers."""
    layers = vlm._get_text_layers()

    ko_bbox, ko_answer = [], []

    for L in range(len(layers)):
        # Variant A: bbox tokens
        if bbox_indices:
            handle = layers[L].mlp.register_forward_hook(
                _make_bbox_mlp_hook(vision_indices, bbox_indices)
            )
            try:
                lp = score_answer(vlm, inputs, ans_toks)
                ko_bbox.append(float(clean_logp - lp))
            except Exception:
                ko_bbox.append(None)
            finally:
                handle.remove()
        else:
            ko_bbox.append(None)

        # Variant B: answer token
        if text_indices:
            handle = layers[L].mlp.register_forward_hook(
                _make_answer_mlp_hook(text_indices, answer_pos)
            )
            try:
                lp = score_answer(vlm, inputs, ans_toks)
                ko_answer.append(float(clean_logp - lp))
            except Exception:
                ko_answer.append(None)
            finally:
                handle.remove()
        else:
            ko_answer.append(None)

    return {"mlp_ko_bbox": ko_bbox, "mlp_ko_answer": ko_answer}


# ── peak helpers ─────────────────────────────────────────────────

def _peak_val(curve):
    valid = [v for v in curve if v is not None]
    return float(max(valid)) if valid else None


def _peak_layer(curve):
    valid = [(i, v) for i, v in enumerate(curve) if v is not None]
    return max(valid, key=lambda x: x[1])[0] if valid else None


# ── per-sample ───────────────────────────────────────────────────

def analyze_sample(vlm, sample, image_objects=None):
    """Run MLP knockout analysis for one sample."""
    question = sample["question_clean"]
    answer = sample["answer"]
    img_path = sample.get("image_path", "")

    if not img_path or not os.path.exists(img_path):
        return None

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        return None

    # The question is used as is, as in the paper's Qwen runs
    # (the LLaVA scripts add " Answer with yes or no." to yes/no questions).
    inputs = build_inputs(vlm, image, question)
    vis_idx = get_vision_indices(vlm, inputs)

    if not vis_idx or inputs.get("image_grid_thw") is None:
        return None

    ans_toks = get_answer_tokens(vlm, answer)
    if not ans_toks:
        return None

    ans_pos = inputs["input_ids"].shape[1] - 1
    correct, _ = compute_correct(vlm, inputs, answer)

    grid_thw = inputs["image_grid_thw"][0].tolist()
    bbox_positions = _get_bbox_positions(sample, image, vis_idx, grid_thw, image_objects)
    text_indices = _get_text_indices(vlm, inputs)

    # Cache embeddings once to avoid repeated vision encoder calls
    captured = {}

    def _cap(module, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is not None and "e" not in captured:
            captured["e"] = hs.detach().clone()

    handle = vlm._get_text_layers()[0].register_forward_pre_hook(_cap, with_kwargs=True)
    clean_logp = score_answer(vlm, inputs, ans_toks)
    handle.remove()

    lean_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
    if "e" in captured:
        lean_inputs["inputs_embeds"] = captured["e"]

    mlp_results = _run_mlp_knockout(
        vlm,
        lean_inputs,
        vis_idx,
        bbox_positions,
        text_indices,
        ans_pos,
        ans_toks,
        float(clean_logp),
    )

    return {
        "question_id": sample["question_id"],
        "operation": sample.get("operation", ""),
        "model_correct": correct,
        "clean_logp": float(clean_logp),
        "mlp_ko_bbox": mlp_results["mlp_ko_bbox"],
        "mlp_ko_answer": mlp_results["mlp_ko_answer"],
        "mlp_peak_bbox": _peak_val(mlp_results["mlp_ko_bbox"]),
        "mlp_peak_answer": _peak_val(mlp_results["mlp_ko_answer"]),
        "mlp_peak_layer_bbox": _peak_layer(mlp_results["mlp_ko_bbox"]),
        "mlp_peak_layer_answer": _peak_layer(mlp_results["mlp_ko_answer"]),
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


def print_summary(agg: Optional[dict]):
    def _f(x):
        return f"{x:.4f}" if x is not None else "N/A"

    def _sig(p):
        return "***" if p and p < 0.001 else "**" if p and p < 0.01 \
               else "*" if p and p < 0.05 else "ns"

    print("\n" + "=" * 90)
    print("MLP Knockout Summary  (Qwen2.5-VL)")
    print("=" * 90)

    for variant, label in [("bbox",   "Variant A — bbox MLP KO"),
                            ("answer", "Variant B — answer MLP KO")]:
        print(f"\n{label}:")
        print(f"  {'op':12s}  {'n_c':>5}  {'n_i':>5}  "
              f"{'peak_c':>8}  {'peak_i':>8}  {'cohen_d':>8}  "
              f"{'p_peak':>10}  {'peakL_c':>8}  {'peakL_i':>8}  {'sig_L':>5}")
        if not agg:
            continue
        for op, d in sorted(agg.items()):
            v = d.get(f"variant_{variant}", {})
            if not v:
                continue
            p = v.get("mannwhitney_p_peak")
            print(f"  {op:12s}  "
                  f"{v.get('n_correct',   0):>5}  "
                  f"{v.get('n_incorrect', 0):>5}  "
                  f"{_f(v.get('mean_peak_correct')):>8}  "
                  f"{_f(v.get('mean_peak_incorrect')):>8}  "
                  f"{_f(v.get('cohen_d_peak')):>8}  "
                  f"{_f(p):>8} {_sig(p):3s}  "
                  f"{str(v.get('peak_layer_correct',   'N/A')):>8}  "
                  f"{str(v.get('peak_layer_incorrect', 'N/A')):>8}  "
                  f"{str(v.get('n_sig_layers', 'N/A')):>5}")


# ── main runner ──────────────────────────────────────────────────

def run(samples_path, image_dir, model_id, output_dir, device, max_per_op, ops, seed):
    global N_LAYERS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Resolve output directory (no hardcoded paths)
    output_dir = output_dir or str(Path.cwd() / "results" / "mlp_knockout")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading samples from {samples_path} ...")
    samples, image_objects_map = load_samples(samples_path, image_dir)
    selected = select_balanced(samples, ops, max_per_op)

    print(f"  {len(selected)} samples")

    vlm = HookedQwenVL(model_id=model_id, device=device, quantize=False)
    vlm.model.eval()
    N_LAYERS = len(vlm._get_text_layers())

    results = []
    samples_path_out = os.path.join(output_dir, "per_sample.jsonl")

    with open(samples_path_out, "w") as jsonl_f:
        for s in tqdm(selected):
            img_objects = image_objects_map.get(s.get("image_id", ""), {})
            try:
                r = analyze_sample(vlm, s, image_objects=img_objects)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  [ERR] {s.get('question_id')}: {e}")
                continue

            if r is None:
                continue

            results.append(r)
            jsonl_f.write(json.dumps(r) + "\n")
            jsonl_f.flush()
            torch.cuda.empty_cache()

    agg = aggregate_mlp(results)
    print_summary(agg)

    out = {
        "config": {"model_id": model_id, "max_per_op": max_per_op, "ops": ops, "seed": seed},
        "mlp_by_op": agg,
    }
    out_path = os.path.join(output_dir, "mlp_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved → {out_path}")
    print(f"Saved → {samples_path_out}")


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--image_dir", default="data/images", help="Folder holding the images named in --samples")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_per_op", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ops", nargs="+",
                        default=["relate", "select", "exist", "verify", "filter", "choose", "query"])

    args = parser.parse_args()

    run(
        samples_path=args.samples,
        image_dir=args.image_dir,
        model_id=args.model_id,
        output_dir=args.output_dir,
        device=args.device,
        max_per_op=args.max_per_op,
        ops=args.ops,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
