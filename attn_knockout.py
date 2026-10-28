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
    load_samples,
)

# Qwen2.5-VL-3B defaults (36 layers, 16 heads, head_dim 128).
# run() resets these from the loaded model's config, so other sizes work too.
N_LAYERS = 36
N_HEADS = 16
HEAD_DIM = 128


def text_dims(model) -> tuple[int, int, int]:
    """Return (n_layers, n_heads, head_dim) of the language model."""
    cfg = getattr(model.config, "text_config", model.config)
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    return cfg.num_hidden_layers, n_heads, head_dim


def select_balanced(samples, ops_of_interest, max_per_op):
    by_op = defaultdict(list)
    for sample in samples:
        op = sample.get("operation", "")
        if op in ops_of_interest:
            by_op[op].append(sample)

    out = []
    for op in ops_of_interest:
        pool = list(by_op.get(op, []))
        random.shuffle(pool)
        out.extend(pool[:max_per_op])

    random.shuffle(out)
    return out


def safe_mean(vals):
    v = [x for x in vals if x is not None and not np.isnan(float(x))]
    return float(np.mean(v)) if v else None


def cohen_d(a, b):
    a = np.array([x for x in a if x is not None], dtype=float)
    b = np.array([x for x in b if x is not None], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    pool = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return float((np.mean(a) - np.mean(b)) / pool) if pool > 0 else None


def _per_layer_mw(c_raw, i_raw):
    """Mann-Whitney test per layer from lists of knockout curves."""
    if not c_raw or not i_raw:
        return [None] * N_LAYERS

    def _nan(curve):
        return [np.nan if v is None else v for v in curve]

    c_arr = np.array([_nan(c) for c in c_raw])
    i_arr = np.array([_nan(c) for c in i_raw])
    pvals = []

    for layer_idx in range(N_LAYERS):
        c_l = c_arr[:, layer_idx]
        c_l = c_l[~np.isnan(c_l)]
        i_l = i_arr[:, layer_idx]
        i_l = i_l[~np.isnan(i_l)]
        if len(c_l) >= 3 and len(i_l) >= 3:
            _, p = mannwhitneyu(c_l, i_l, alternative="two-sided")
            pvals.append(float(p))
        else:
            pvals.append(None)
    return pvals


def _bh_correct(pvals):
    """Benjamini-Hochberg correction over per-layer p-values."""
    valid = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if not valid:
        return pvals[:]

    n = len(valid)
    sorted_v = sorted(valid, key=lambda x: x[1])
    adj = {}
    prev = 1.0
    for rank, (i, p) in enumerate(reversed(sorted_v)):
        a = min(prev, p * n / (n - rank))
        adj[i] = a
        prev = a
    return [adj.get(i) for i in range(len(pvals))]


def _get_bbox_positions(
    sample: dict,
    image,
    vis_idx: List[int],
    grid_thw,
    image_objects: Optional[dict],
) -> List[int]:
    """Collect unique bbox-mapped vision token positions for the sample."""
    if not image_objects:
        return []

    program = sample.get("semantic_program", [])
    if not program:
        return []

    groups = extract_program_objects(
        program,
        image_objects,
        sample_operation=sample.get("operation", ""),
    )
    if not groups:
        return []

    all_pos = []
    for group in groups:
        for obj in group["objects"]:
            _, abs_pos = bbox_to_token_positions(image, obj["bbox"], vis_idx, grid_thw)
            all_pos.extend(abs_pos)

    return sorted(set(all_pos))


class HeadAblationHook:
    """Zero one head's slice of the o_proj input at the answer position."""

    def __init__(self, head_idx: int, ans_pos: int):
        self.head_idx = head_idx
        self.ans_pos = ans_pos
        self.lo = head_idx * HEAD_DIM
        self.hi = self.lo + HEAD_DIM

    def __call__(self, module, args):
        x = args[0]
        if x is None or x.shape[1] <= self.ans_pos:
            return args
        x = x.clone()
        x[:, self.ans_pos, self.lo:self.hi] = 0
        return (x,)


def _run_attention_knockout_stage_a_impl(
    vlm: HookedQwenVL,
    inputs: dict,
    ans_pos: int,
    bbox_positions: List[int],
    clean_logp: float,
    ans_toks: List[int],
) -> Optional[dict]:
    """Layer-level attention knockout by blocking answer→bbox attention."""
    if not bbox_positions:
        return None

    pairs = [(ans_pos, bp) for bp in bbox_positions]
    ko_curve = []

    for layer_idx in range(N_LAYERS):
        try:
            with vlm.block_attention({layer_idx: pairs}):
                ko_lp = score_answer(vlm, inputs, ans_toks)
            ko_curve.append(float(clean_logp - ko_lp))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            ko_curve.append(None)
        except Exception as e:
            print(f"    [WARN] Stage A layer {layer_idx} failed: {e}")
            ko_curve.append(None)

    return {"ko_curve": ko_curve}


def _run_head_ablation_impl(
    vlm: HookedQwenVL,
    inputs: dict,
    ans_pos: int,
    clean_logp: float,
    ans_toks: List[int],
) -> dict:
    """Head-level ablation using one batched forward pass per layer."""
    layers = vlm._get_text_layers()
    prompt_len = inputs["input_ids"].shape[1]

    batch_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == 1:
            batch_inputs[key] = value.expand(N_HEADS, *value.shape[1:]).contiguous()
        else:
            batch_inputs[key] = value

    def _batch_hook(_module, args):
        x = args[0].clone()
        for head_idx in range(N_HEADS):
            x[head_idx, ans_pos, head_idx * HEAD_DIM:(head_idx + 1) * HEAD_DIM] = 0
        return (x,)

    matrix = []
    for layer_idx in range(N_LAYERS):
        handle = layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(_batch_hook)
        try:
            with torch.no_grad():
                out = vlm.model(**batch_inputs)
            lp = F.log_softmax(out.logits.float(), dim=-1)
            row = []
            for head_idx in range(N_HEADS):
                total, count = 0.0, 0
                for i, tok in enumerate(ans_toks):
                    idx = prompt_len - 1 + i
                    if 0 <= idx < lp.shape[1]:
                        total += lp[head_idx, idx, tok].item()
                        count += 1
                row.append(float(clean_logp - total / max(count, 1)))
            matrix.append(row)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            matrix.append([None] * N_HEADS)
        except Exception as e:
            print(f"    [WARN] batched head-ablation L{layer_idx}: {e}")
            matrix.append([None] * N_HEADS)
        finally:
            handle.remove()

    return {"head_ko_matrix": matrix}


def analyze_sample(
    vlm: HookedQwenVL,
    sample: dict,
    image_objects: Optional[dict] = None,
    run_knockout: bool = True,
    head_level: bool = False,
) -> Optional[dict]:
    """Run knockout analysis for one sample."""
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
    correct, model_response = compute_correct(vlm, inputs, answer)

    grid_thw = inputs["image_grid_thw"][0].tolist()
    bbox_positions = _get_bbox_positions(sample, image, vis_idx, grid_thw, image_objects)

    # Capture layer-0 inputs once so later knockout passes can reuse inputs_embeds.
    captured = {}

    def _cap(module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is not None and "e" not in captured:
            captured["e"] = hidden_states.detach().clone()

    handle = vlm._get_text_layers()[0].register_forward_pre_hook(_cap, with_kwargs=True)
    clean_logp = score_answer(vlm, inputs, ans_toks)
    handle.remove()

    # Reuse fused embeddings and skip pixel_values in repeated passes.
    lean_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
    if "e" in captured:
        lean_inputs["inputs_embeds"] = captured["e"]

    knockout = None
    if run_knockout and bbox_positions:
        stage_a = _run_attention_knockout_stage_a_impl(
            vlm,
            lean_inputs,
            ans_pos,
            bbox_positions,
            float(clean_logp),
            ans_toks,
        )
        knockout = stage_a

        if head_level and stage_a is not None:
            stage_b = _run_head_ablation_impl(
                vlm,
                lean_inputs,
                ans_pos,
                float(clean_logp),
                ans_toks,
            )
            knockout = {**stage_a, **stage_b}

    return {
        "question_id": sample["question_id"],
        "operation": sample.get("operation", ""),
        "question": question,
        "answer": answer,
        "model_response": model_response,
        "model_correct": correct,
        "clean_logp": float(clean_logp),
        "knockout": knockout,
    }


def _gss_curve(c_curves, i_curves):
    def layer_mean(curves, layer_idx):
        vals = [curve[layer_idx] for curve in curves if curve[layer_idx] is not None]
        return float(np.mean(vals)) if vals else None

    return [
        (lambda mc, mi: mc - mi if mc is not None and mi is not None else None)(
            layer_mean(c_curves, layer_idx),
            layer_mean(i_curves, layer_idx),
        )
        for layer_idx in range(N_LAYERS)
    ]


def aggregate_knockout(results: List[dict]) -> dict:
    """Aggregate layer-level and head-level knockout statistics by operation."""
    by_op: Dict[str, dict] = defaultdict(
        lambda: {
            "correct": {"ko": [], "head_matrices": []},
            "incorrect": {"ko": [], "head_matrices": []},
        }
    )

    for result in results:
        ko = result.get("knockout")
        if not ko:
            continue
        op = result["operation"]
        key = "correct" if result["model_correct"] is True else "incorrect"

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
        arr = np.array([_to_nan(curve) for curve in raw_curves])
        return np.nanmean(arr, axis=0).tolist(), np.nanstd(arr, axis=0).tolist()

    def _peak_ko(raw_curves):
        vals = []
        for curve in raw_curves:
            valid = [v for v in curve if v is not None]
            if valid:
                vals.append(max(valid))
        return vals

    def _argmax_mean(mean_curve):
        if not mean_curve:
            return None
        valid = [
            (layer_idx, value)
            for layer_idx, value in enumerate(mean_curve)
            if value is not None and not np.isnan(value)
        ]
        return max(valid, key=lambda x: x[1])[0] if valid else None

    out = {}
    for op, grps in by_op.items():
        op_out = {}
        for key in ("correct", "incorrect"):
            ko_mean, ko_std = _curve_stats(grps[key]["ko"])
            op_out[key] = {
                "n": len(grps[key]["ko"]),
                "mean_ko_curve": ko_mean,
                "std_ko_curve": ko_std,
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
        valid_gss = [(layer_idx, value) for layer_idx, value in enumerate(gss) if value is not None]
        if valid_gss:
            peak_gss_layer = max(valid_gss, key=lambda x: abs(x[1]))[0]

        pl_p_ko = _per_layer_mw(grps["correct"]["ko"], grps["incorrect"]["ko"])
        pl_p_ko_bh = _bh_correct(pl_p_ko)
        n_sig_ko = sum(1 for p in pl_p_ko_bh if p is not None and p < 0.05)

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
                    (layer_idx, head_idx, mat[layer_idx][head_idx])
                    for layer_idx in range(N_LAYERS)
                    for head_idx in range(N_HEADS)
                    if mat[layer_idx][head_idx] is not None and not np.isnan(mat[layer_idx][head_idx])
                ]
                flat.sort(key=lambda x: x[2], reverse=True)
                return [(layer_idx, head_idx, round(value, 5)) for layer_idx, head_idx, value in flat[:n]]

            def _per_head_cohen_d(mats_c, mats_i):
                if not mats_c or not mats_i:
                    return None
                arr_c = np.array(mats_c, dtype=float)
                arr_i = np.array(mats_i, dtype=float)
                peak_c = np.nanmax(arr_c, axis=1)
                peak_i = np.nanmax(arr_i, axis=1)
                return [
                    cohen_d(peak_c[:, head_idx].tolist(), peak_i[:, head_idx].tolist())
                    for head_idx in range(N_HEADS)
                ]

            head_agg = {
                "n_correct": len(all_mats_c),
                "n_incorrect": len(all_mats_i),
                "mean_matrix_correct": mean_mat_c,
                "mean_matrix_incorrect": mean_mat_i,
                "top10_heads_correct": _top_heads(mean_mat_c),
                "top10_heads_incorrect": _top_heads(mean_mat_i),
                "per_head_cohen_d": _per_head_cohen_d(all_mats_c, all_mats_i),
            }

        out[op] = {
            **op_out,
            "mean_peak_ko_correct": safe_mean(c_peak),
            "mean_peak_ko_incorrect": safe_mean(i_peak),
            "cohen_d_peak_ko": cohen_d(c_peak, i_peak),
            "mannwhitney_p_peak_ko": float(p_peak) if p_peak is not None else None,
            "peak_ko_layer_correct": _argmax_mean(c_mean_curve),
            "peak_ko_layer_incorrect": _argmax_mean(i_mean_curve),
            "gss": gss,
            "peak_gss_layer": peak_gss_layer,
            "per_layer_p_ko": pl_p_ko,
            "per_layer_p_ko_bh": pl_p_ko_bh,
            "n_sig_layers_ko": n_sig_ko,
            "head_level": head_agg,
        }
    return out


def print_summary(ko_agg: Optional[dict]):
    """Print a compact console summary."""
    def _f(x):
        return f"{x:.4f}" if x is not None else "N/A"

    def _sig(p):
        return "***" if p and p < 0.001 else "**" if p and p < 0.01 else "*" if p and p < 0.05 else "ns"

    print("\n" + "=" * 80)

    if ko_agg:
        print("Attention Knockout (Layer-Level):")
        print(
            f"  {'op':12s}  {'n_c':>5}  {'n_i':>5}  {'KO_c':>8}  {'KO_i':>8}  {'cohen_d':>8}  "
            f"{'p_KO':>10}  {'peak_L_c':>9}  {'peak_L_i':>9}  {'sig_L':>5}  {'GSS_dir':>7}"
        )
        for op, data in sorted(ko_agg.items()):
            nc = data.get("correct", {}).get("n", 0)
            ni = data.get("incorrect", {}).get("n", 0)
            p = data.get("mannwhitney_p_peak_ko")
            cd = data.get("cohen_d_peak_ko")
            gss = data.get("gss", [])
            peak_gss = data.get("peak_gss_layer")

            if peak_gss is not None and gss and gss[peak_gss] is not None:
                gss_dir = "↑" if gss[peak_gss] > 0 else ("↓" if gss[peak_gss] < 0 else "=")
            else:
                gss_dir = "="

            print(
                f"  {op:12s}  {nc:>5}  {ni:>5}  "
                f"{_f(data.get('mean_peak_ko_correct')):>8}  "
                f"{_f(data.get('mean_peak_ko_incorrect')):>8}  "
                f"{_f(cd):>8}  "
                f"{_f(p):>8} {_sig(p):3s}  "
                f"{str(data.get('peak_ko_layer_correct', 'N/A')):>9}  "
                f"{str(data.get('peak_ko_layer_incorrect', 'N/A')):>9}  "
                f"{str(data.get('n_sig_layers_ko', 'N/A')):>5}  "
                f"{gss_dir:>7}"
            )

        has_head = any(data.get("head_level") for data in ko_agg.values())
        if has_head:
            print("\n" + "-" * 80)
            print("Head-Level Attribution (top-10 heads by mean degradation on correct samples):")
            print(f"  {'op':12s}  {'rank':>4}  {'L':>3}  {'H':>3}  {'mean_deg':>10}  {'cohen_d':>8}")
            for op, data in sorted(ko_agg.items()):
                head_level = data.get("head_level")
                if not head_level:
                    continue
                tops = head_level.get("top10_heads_correct", [])
                cd_ph = head_level.get("per_head_cohen_d") or []
                for rank, (layer_idx, head_idx, value) in enumerate(tops, 1):
                    cd_val = cd_ph[head_idx] if head_idx < len(cd_ph) else None
                    print(
                        f"  {op if rank == 1 else '':12s}  {rank:>4}  {layer_idx:>3}  {head_idx:>3}  "
                        f"{value:>10.5f}  {_f(cd_val):>8}"
                    )
                if cd_ph:
                    ranked_cd = sorted(
                        [(head_idx, cd_ph[head_idx]) for head_idx in range(len(cd_ph)) if cd_ph[head_idx] is not None],
                        key=lambda x: abs(x[1]) if x[1] is not None else 0,
                        reverse=True,
                    )[:5]
                    top_cd_str = "  ".join(f"H{head_idx}(d={value:+.2f})" for head_idx, value in ranked_cd)
                    print(f"  {'':12s}  Top discriminative heads: {top_cd_str}")


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
    """Run the analysis and save per-sample plus aggregate outputs."""
    global N_LAYERS, N_HEADS, HEAD_DIM
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading samples from {samples_path} ...")
    samples, image_objects_map = load_samples(samples_path, image_dir)
    selected = select_balanced(samples, ops, max_per_op)
    print(f"  {len(selected)} samples  ({ops})")
    print(f"  run_knockout={run_knockout}  head_level={head_level}")

    print(f"\nLoading {model_id} on {device} ...")
    # fp16 as in the paper's run. In bf16, log-probs near -20 are rounded to steps of 1/8,
    # which adds about 0.1 to every peak knockout value.
    vlm = HookedQwenVL(
        model_id=model_id,
        device=device,
        quantize=True,
        quantize_type="fp16",
        attn_implementation="eager",
    )
    vlm.model.eval()
    N_LAYERS, N_HEADS, HEAD_DIM = text_dims(vlm.model)
    print(f"  {N_LAYERS} layers, {N_HEADS} heads, head_dim {HEAD_DIM}")

    samples_path_out = os.path.join(output_dir, "per_sample.jsonl")

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return None

    results = []
    with open(samples_path_out, "w") as jsonl_f:
        for sample in tqdm(selected, desc="Analysing"):
            img_objects = image_objects_map.get(sample.get("image_id", ""), {})
            try:
                result = analyze_sample(
                    vlm,
                    sample,
                    image_objects=img_objects,
                    run_knockout=run_knockout,
                    head_level=head_level,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  [ERR] {sample.get('question_id')}: {e}")
                import traceback
                traceback.print_exc()
                continue

            if result is None:
                continue

            results.append(result)
            jsonl_f.write(json.dumps(result, default=_default) + "\n")
            jsonl_f.flush()
            torch.cuda.empty_cache()

    print(f"\n{len(results)} samples collected")
    n_correct = sum(1 for result in results if result["model_correct"])
    print(f"  model accuracy: {n_correct}/{len(results)} = {n_correct / max(len(results), 1):.3f}")

    by_op_acc: Dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        op = result["operation"]
        by_op_acc[op]["total"] += 1
        if result["model_correct"]:
            by_op_acc[op]["correct"] += 1

    print("\nPer-operation accuracy:")
    for op in sorted(by_op_acc):
        correct = by_op_acc[op]["correct"]
        total = by_op_acc[op]["total"]
        print(f"  {op:12s}: {correct}/{total} = {correct / max(total, 1):.3f}")

    ko_agg = aggregate_knockout(results) if run_knockout else None
    print_summary(ko_agg)

    out = {
        "config": {
            "run_knockout": run_knockout,
            "head_level": head_level,
        },
        "knockout_by_op": ko_agg,
    }

    agg_path = os.path.join(output_dir, "attention_results.json")
    with open(agg_path, "w") as f:
        json.dump(out, f, indent=2, default=_default)

    print(f"\nSaved → {agg_path}")
    print(f"Saved → {samples_path_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--image_dir", default="data/images", help="Folder holding the images named in --samples")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_per_op", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["relate", "select", "exist", "verify", "filter", "choose", "query"],
    )
    parser.add_argument("--knockout", dest="run_knockout", action="store_true", default=True)
    parser.add_argument("--no-knockout", dest="run_knockout", action="store_false")
    parser.add_argument("--head_level", dest="head_level", action="store_true", default=False)
    args = parser.parse_args()

    # Keep output_dir explicit if passed; otherwise use a local results folder.
    output_dir = args.output_dir or str(Path.cwd() / "results" / "attention_knockout")

    run(
        samples_path=args.samples,
        image_dir=args.image_dir,
        model_id=args.model_id,
        output_dir=output_dir,
        device=args.device,
        max_per_op=args.max_per_op,
        ops=args.ops,
        seed=args.seed,
        run_knockout=args.run_knockout,
        head_level=args.head_level,
    )


if __name__ == "__main__":
    main()
