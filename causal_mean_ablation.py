#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import kendalltau, mannwhitneyu, spearmanr
from tqdm import tqdm

from HookedQwenVL import HookedQwenVL
from bbox_utils import bbox_to_token_positions

IMAGE_PAD_TOKEN = "<|image_pad|>"
ONSET_FRAC = 0.10
MIN_DEG = 1e-4


@torch.no_grad()
def cache_layer_stats(vlm, inputs, vis_idx: List[int]):
    """Cache per-layer mean/std over vision-token hidden states."""
    n_layers = vlm.model.config.num_hidden_layers
    captured = {}

    def make_hook(layer_idx):
        def hook(module, inputs_, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden is not None and hidden.dim() == 3:
                captured[layer_idx] = hidden[0].detach().float().cpu()
        return hook

    layers = vlm._get_text_layers()
    handles = [layers[layer_idx].register_forward_hook(make_hook(layer_idx)) for layer_idx in range(n_layers)]

    try:
        vlm.model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    if not captured:
        print("  [WARN] cache_layer_stats: no layers captured")
        return None, None

    mean_acts, std_acts = {}, {}
    for layer_idx in range(n_layers):
        hidden_full = captured[layer_idx]
        hidden_vis = hidden_full[vis_idx]
        mean_vec = hidden_vis.mean(dim=0)
        std_vec = hidden_vis.std(dim=0).clamp(min=1e-6)

        # Expand mean to [seq, hidden] so patch_at_layer can reuse it directly.
        mean_acts[layer_idx] = mean_vec.unsqueeze(0).expand(hidden_full.shape[0], -1).clone()
        std_acts[layer_idx] = std_vec

    return mean_acts, std_acts


def cache_mean_acts(vlm, inputs, vis_idx):
    """Helper when only per-layer means are needed."""
    mean_acts, _ = cache_layer_stats(vlm, inputs, vis_idx)
    return mean_acts


def analyse_curve(deg: List[float]) -> dict:
    """Summarize a degradation curve using a few scalar stats."""
    arr = np.clip(np.array(deg, dtype=float), 0, None)
    peak = float(arr.max())
    valid = peak > MIN_DEG
    peak_layer = int(np.argmax(arr)) if valid else None
    onset_layer = None

    if valid:
        hits = np.where(arr > ONSET_FRAC * peak)[0]
        onset_layer = int(hits[0]) if len(hits) else None

    return {
        "peak_value": peak,
        "auc": float(arr.sum()),
        "peak_layer": peak_layer,
        "onset_layer": onset_layer,
        "valid": valid,
    }


def parse_obj_id(arg: str) -> Optional[str]:
    """Extract an object id from strings like 'chair (12345)'."""
    match = re.search(r"\((\d+)\)", arg)
    return match.group(1) if match else None


def base_op(op_name: str) -> str:
    """Return the base operation name."""
    return op_name.split()[0] if op_name else ""


def extract_program_objects(
    program: List[dict],
    image_objects: Dict[str, dict],
    sample_operation: str = "",
) -> List[dict]:
    """Build object groups to patch, based on program structure."""
    op = sample_operation.split()[0] if sample_operation else ""

    def resolve_obj(arg: str) -> Optional[dict]:
        obj_id = parse_obj_id(arg)
        obj = image_objects.get(obj_id) if obj_id else None

        if obj is None:
            clean_name = re.sub(r"\(\d+\)", "", arg).strip().split(",")[0].strip().lower()
            if clean_name:
                for candidate in image_objects.values():
                    if candidate.get("name", "").lower() == clean_name:
                        obj = candidate
                        break

        if obj is None or not obj.get("bbox"):
            return None

        return {
            "obj_id": str(obj.get("obj_id", "")),
            "name": obj.get("name", "unknown"),
            "bbox": obj["bbox"],
        }

    if op in ("select", "common", "different", ""):
        objs = []
        for step in program:
            if base_op(step.get("operation", "")) == "select":
                obj = resolve_obj(step.get("argument", ""))
                if obj:
                    objs.append(obj)
        return [{"objects": objs, "role": "select_joint", "step_depth": len(objs)}] if objs else []

    if op == "query":
        for step in program:
            if base_op(step.get("operation", "")) == "select":
                obj = resolve_obj(step.get("argument", ""))
                if obj:
                    return [{"objects": [obj], "role": "query_target", "step_depth": 0}]
        return []

    if op == "exist":
        groups = []
        for idx, step in enumerate(program):
            if base_op(step.get("operation", "")) != "select":
                continue

            next_steps = [
                s for s in program
                if idx in s.get("dependencies", []) and base_op(s.get("operation", "")) == "exist"
            ]
            if not next_steps:
                continue

            obj = resolve_obj(step.get("argument", ""))
            if obj:
                groups.append({"objects": [obj], "role": "exist_branch", "step_depth": idx})
        return groups

    if op == "choose":
        for step in program:
            if base_op(step.get("operation", "")) == "select":
                obj = resolve_obj(step.get("argument", ""))
                if obj:
                    return [{"objects": [obj], "role": "choose_target", "step_depth": 0}]
        return []

    if op == "filter":
        for step in program:
            if base_op(step.get("operation", "")) == "select":
                obj = resolve_obj(step.get("argument", ""))
                if obj:
                    return [{"objects": [obj], "role": "filter_target", "step_depth": 0}]
        return []

    if op == "relate":
        for step in reversed(program):
            if base_op(step.get("operation", "")) == "relate":
                obj = resolve_obj(step.get("argument", ""))
                if obj:
                    return [{"objects": [obj], "role": "relate_terminal", "step_depth": len(program) - 1}]
        return []

    if op == "verify":
        primary_obj = None
        relation_obj = None
        for step in program:
            step_op = base_op(step.get("operation", ""))
            if step_op == "select" and primary_obj is None:
                primary_obj = resolve_obj(step.get("argument", ""))
            if step_op == "verify":
                relation_obj = resolve_obj(step.get("argument", ""))

        objs = [obj for obj in [primary_obj, relation_obj] if obj is not None]
        return [{"objects": objs, "role": "verify_joint", "step_depth": 1}] if objs else []

    groups = []
    for idx, step in enumerate(program):
        if base_op(step.get("operation", "")) == "select":
            obj = resolve_obj(step.get("argument", ""))
            if obj:
                groups.append({"objects": [obj], "role": "fallback", "step_depth": idx})
    return groups


def load_samples(path: str, image_dir: Optional[str] = None):
    """Load image-grouped samples and attach a full image path to each one."""
    with open(path) as f:
        data = json.load(f)

    samples, image_objects_map = [], {}
    for img_id, img_data in data["images"].items():
        image_objects_map[img_id] = img_data.get("objects", {})
        for sample in img_data.get("samples", []):
            # Image file names are stored once per image; join them onto --image_dir.
            img_file = sample.get("image_path") or img_data.get("image_path", "")
            sample["image_path"] = os.path.join(image_dir, img_file) if image_dir else img_file
            sample["image_id"] = img_id
            samples.append(sample)

    missing = sum(1 for p in {s["image_path"] for s in samples} if not os.path.exists(p))
    if missing:
        print(f"  [WARN] {missing} image files not found (check --image_dir); their samples will be skipped")
    return samples, image_objects_map


def build_inputs(vlm, image, question):
    """Convert an image-question pair into model inputs."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return vlm.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(vlm.model.device)


def get_vision_indices(vlm, inputs) -> List[int]:
    """Find token positions corresponding to image padding tokens."""
    pad_id = vlm.processor.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    return torch.where(inputs["input_ids"][0] == pad_id)[0].tolist()


def get_answer_tokens(vlm, answer: str) -> List[int]:
    """Tokenize the gold answer, preferring a leading-space variant."""
    toks = vlm.processor.tokenizer.encode(f" {answer}", add_special_tokens=False)
    return toks or vlm.processor.tokenizer.encode(answer, add_special_tokens=False)


def score_answer(vlm, inputs, ans_toks: List[int]) -> float:
    """Average teacher-forced log-prob of the gold answer tokens."""
    if not ans_toks:
        return float("-inf")

    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        logits = vlm.model(**inputs).logits[0]

    log_probs = F.log_softmax(logits, dim=-1)
    total, count = 0.0, 0
    for offset, tok in enumerate(ans_toks):
        idx = prompt_len - 1 + offset
        if 0 <= idx < len(log_probs):
            total += log_probs[idx, tok].item()
            count += 1

    return total / max(count, 1)


def model_correct(vlm, inputs, answer: str):
    """Generate a response and check whether it contains the gold answer."""
    with torch.no_grad():
        output = vlm.model.generate(**inputs, max_new_tokens=20, do_sample=False)

    decoded = vlm.processor.tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return answer.strip().lower() in decoded.lower(), decoded


@torch.no_grad()
def run_per_object_patching(
    vlm,
    inputs,
    ans_toks,
    vis_idx,
    mean_acts,
    n_layers,
    clean_lp,
    prog_groups,
    image,
    grid_thw,
) -> List[dict]:
    """Patch each object group across layers and record degradation curves."""
    results = []

    for group in prog_groups:
        objs = group["objects"]
        role = group["role"]
        depth = group["step_depth"]

        all_abs_pos = []
        obj_names = []
        for obj in objs:
            _, abs_pos = bbox_to_token_positions(image, obj["bbox"], vis_idx, grid_thw)
            all_abs_pos.extend(abs_pos)
            obj_names.append(obj["name"])

        # Overlapping boxes may map to the same visual tokens.
        all_abs_pos = sorted(set(all_abs_pos))
        if not all_abs_pos:
            continue

        with vlm.patch_at_layer(0, all_abs_pos, mean_acts[0]):
            test_lp = score_answer(vlm, inputs, ans_toks)
        if abs(clean_lp - test_lp) < 1e-7:
            print(f"  [WARN] group='{obj_names}': layer-0 patch zero effect (n_tokens={len(all_abs_pos)})")

        deg = []
        for layer_idx in range(n_layers):
            with vlm.patch_at_layer(layer_idx, all_abs_pos, mean_acts[layer_idx]):
                patched_lp = score_answer(vlm, inputs, ans_toks)
            deg.append(float(clean_lp - patched_lp))

        # Plain mean ablation for every object, as in the paper's Qwen run
        # (the LLaVA script falls back to mean + Gaussian noise when this curve is flat).
        ablation_method = "mean"

        results.append({
            "obj_id": "+".join(obj["obj_id"] for obj in objs),
            "name": "+".join(obj_names),
            "operation": role,
            "step_depth": depth,
            "n_tokens": len(all_abs_pos),
            "ablation_method": ablation_method,
            "degradation": deg,
            "curve_stats": analyse_curve(deg),
        })

    return results


def compute_pair_metrics(obj_results, question_id, correct_flag):
    """Compare peak-layer ordering between valid object pairs."""
    valid = [result for result in obj_results if result["curve_stats"]["valid"]]
    if len(valid) < 2:
        return []

    pairs = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            early, late = valid[i], valid[j]
            if early["step_depth"] == late["step_depth"]:
                continue
            if early["step_depth"] > late["step_depth"]:
                early, late = late, early

            peak_early = early["curve_stats"]["peak_layer"]
            peak_late = late["curve_stats"]["peak_layer"]
            if peak_early is None or peak_late is None:
                continue

            diff = peak_late - peak_early
            pairs.append({
                "question_id": question_id,
                "model_correct": correct_flag,
                "obj_name_early": early["name"],
                "step_depth_early": early["step_depth"],
                "operation_early": early["operation"],
                "peak_value_early": early["curve_stats"]["peak_value"],
                "peak_layer_early": peak_early,
                "obj_name_late": late["name"],
                "step_depth_late": late["step_depth"],
                "peak_value_late": late["curve_stats"]["peak_value"],
                "peak_layer_late": peak_late,
                "step_depth_diff": late["step_depth"] - early["step_depth"],
                "peak_layer_diff": diff,
                "is_ordered": diff > 0,
                "is_tied": diff == 0,
            })
    return pairs


def compute_rank_corr(obj_results):
    """Compute depth-to-peak-layer rank correlation within a sample."""
    valid = [result for result in obj_results if result["curve_stats"]["valid"]]
    if len(valid) < 3:
        return {"kendall_tau": None, "spearman_r": None, "n_valid": len(valid)}

    depths = [result["step_depth"] for result in valid]
    peaks = [result["curve_stats"]["peak_layer"] for result in valid]
    tau, _ = kendalltau(depths, peaks)
    rho, _ = spearmanr(depths, peaks)
    return {
        "kendall_tau": float(tau) if not np.isnan(tau) else None,
        "spearman_r": float(rho) if not np.isnan(rho) else None,
        "n_valid": len(valid),
    }


def analyze_sample(vlm, sample, image_objects):
    """Run the full grounding analysis for one sample."""
    question = sample["question_clean"]
    answer = sample["answer"]
    program = sample.get("semantic_program", [])
    if not program:
        return None

    img_path = sample.get("image_path", "")
    if not img_path or not os.path.exists(img_path):
        return None

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as exc:
        print(f"  [SKIP] {img_path}: {exc}")
        return None

    # The question is used as is, as in the paper's Qwen runs
    # (the LLaVA scripts add " Answer with yes or no." to yes/no questions).
    inputs = build_inputs(vlm, image, question)
    vis_idx = get_vision_indices(vlm, inputs)
    if not vis_idx or inputs.get("image_grid_thw") is None:
        return None

    grid_thw = inputs["image_grid_thw"][0].tolist()
    ans_toks = get_answer_tokens(vlm, answer)
    if not ans_toks:
        return None

    prog_groups = extract_program_objects(
        program,
        image_objects,
        sample_operation=sample.get("operation", ""),
    )
    if not prog_groups:
        return None

    clean_lp = score_answer(vlm, inputs, ans_toks)
    correct, model_ans = model_correct(vlm, inputs, answer)
    print("Model answer(per sample):", model_ans)

    mean_acts, _ = cache_layer_stats(vlm, inputs, vis_idx)
    if mean_acts is None:
        return None

    n_layers = vlm.model.config.num_hidden_layers
    obj_results = run_per_object_patching(
        vlm,
        inputs,
        ans_toks,
        vis_idx,
        mean_acts,
        n_layers,
        clean_lp,
        prog_groups,
        image,
        grid_thw,
    )

    pair_metrics = compute_pair_metrics(obj_results, sample["question_id"], correct)
    rank_corr = compute_rank_corr(obj_results)

    return {
        "question_id": sample["question_id"],
        "image_id": sample.get("image_id", ""),
        "image_path": img_path,
        "question": question,
        "answer": answer,
        "model_answer": model_ans,
        "operation": sample.get("operation", ""),
        "n_steps": len(program),
        "clean_logprob": float(clean_lp),
        "model_correct": correct,
        "n_objects": len(prog_groups),
        "n_valid_objects": sum(1 for result in obj_results if result["curve_stats"]["valid"]),
        "obj_results": obj_results,
        "pair_metrics": pair_metrics,
        "rank_correlation": rank_corr,
    }


def safe_mean(vals):
    """Mean that ignores None and NaN values."""
    cleaned = [x for x in vals if x is not None and not np.isnan(float(x))]
    return float(np.mean(cleaned)) if cleaned else None


def safe_std(vals):
    """Std that ignores None and NaN values."""
    cleaned = [x for x in vals if x is not None and not np.isnan(float(x))]
    return float(np.std(cleaned)) if cleaned else None


def cohen_d(a, b):
    """Effect size between two groups."""
    a = np.array([x for x in a if x is not None], dtype=float)
    b = np.array([x for x in b if x is not None], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None

    pooled = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else None


def aggregate_results(all_samples, all_pairs):
    """Aggregate primary and secondary metrics across samples."""
    by_op: Dict[str, Dict] = defaultdict(lambda: {"correct": [], "incorrect": []})

    for sample in all_samples:
        op = sample["operation"]
        valid = [
            result for result in sample.get("objects_summary", sample.get("obj_results", []))
            if result["curve_stats"]["valid"]
        ]
        if not valid:
            continue

        # Use the deepest valid object; break ties by larger peak effect.
        critical = max(valid, key=lambda x: (x["step_depth"], x["curve_stats"]["peak_value"]))
        group = "correct" if sample["model_correct"] else "incorrect"
        by_op[op][group].append(critical["curve_stats"]["peak_value"])

    op_summary = {}
    for op, groups in by_op.items():
        correct_vals = groups["correct"]
        incorrect_vals = groups["incorrect"]
        p_val = None
        if len(correct_vals) >= 3 and len(incorrect_vals) >= 3:
            _, p_val = mannwhitneyu(correct_vals, incorrect_vals, alternative="two-sided")
            p_val = float(p_val)

        op_summary[op] = {
            "n_correct": len(correct_vals),
            "n_incorrect": len(incorrect_vals),
            "mean_peak_value_correct": safe_mean(correct_vals),
            "mean_peak_value_incorrect": safe_mean(incorrect_vals),
            "cohen_d": cohen_d(correct_vals, incorrect_vals),
            "mannwhitney_p": p_val,
        }

    depth_summary = defaultdict(list)
    for sample in all_samples:
        for result in sample.get("objects_summary", sample.get("obj_results", [])):
            if result["curve_stats"]["valid"]:
                depth_summary[result["step_depth"]].append(result["curve_stats"]["peak_value"])

    depth_summary_out = {
        str(depth): {
            "n": len(values),
            "mean_peak_value": safe_mean(values),
            "std_peak_value": safe_std(values),
        }
        for depth, values in sorted(depth_summary.items())
    }

    n_pairs = len(all_pairs)
    n_ordered = sum(1 for pair in all_pairs if pair["is_ordered"])
    n_tied = sum(1 for pair in all_pairs if pair["is_tied"])

    taus = [sample["rank_correlation"]["kendall_tau"] for sample in all_samples if sample["rank_correlation"]["kendall_tau"] is not None]
    rhos = [sample["rank_correlation"]["spearman_r"] for sample in all_samples if sample["rank_correlation"]["spearman_r"] is not None]

    n_correct = sum(1 for sample in all_samples if sample["model_correct"])
    return {
        "dataset": {
            "n_samples": len(all_samples),
            "n_correct": n_correct,
            "accuracy": n_correct / len(all_samples) if all_samples else None,
        },
        "peak_value_by_operation": op_summary,
        "peak_value_by_step_depth": depth_summary_out,
        "pair_ordering": {
            "n_pairs": n_pairs,
            "n_ordered": n_ordered,
            "n_tied": n_tied,
            "ordering_rate": n_ordered / n_pairs if n_pairs else None,
        },
        "rank_correlation": {
            "mean_kendall_tau": safe_mean(taus),
            "mean_spearman_r": safe_mean(rhos),
            "n_samples": len(taus),
        },
    }


def _json_default(obj):
    """Make NumPy values JSON serializable."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def run(samples_path, image_dir, model_id, output_dir, device, quantize_type, max_samples, save_every, seed, min_objects):
    """Run the full experiment and write sample/object/pair outputs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {samples_path} ...")
    all_samples, image_objects_map = load_samples(samples_path, image_dir)
    print(f"  {len(all_samples)} samples from {len(image_objects_map)} images")
    if max_samples:
        all_samples = all_samples[:max_samples]
        print(f"  Capped at {max_samples}")

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "samples_path": samples_path,
                "model_id": model_id,
                "device": device,
                "quantize_type": quantize_type,
                "ablation": "mean",
                "metric": "peak_degradation",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "min_objects": min_objects,
            },
            f,
            indent=2,
        )

    # Keep backward compatibility with old CLI values.
    quantize_aliases = {"int4": "4bit", "fp16": "fp16", "int8": "int8", "4bit": "4bit"}
    resolved_quantize_type = quantize_aliases[quantize_type]

    print(f"\nLoading {model_id} on {device} ...")
    vlm = HookedQwenVL(
        model_id=model_id,
        device=device,
        quantize=True,
        quantize_type=resolved_quantize_type,
    )
    vlm.model.eval()
    print(f"  {vlm.model.config.num_hidden_layers} layers")

    path_sample = os.path.join(output_dir, "per_sample_results.jsonl")
    path_obj = os.path.join(output_dir, "per_object_results.jsonl")
    path_pairs = os.path.join(output_dir, "pair_results.jsonl")

    f_s = open(path_sample, "w")
    f_o = open(path_obj, "w")
    f_p = open(path_pairs, "w")

    collected_samples, collected_pairs = [], []
    skipped = 0
    errors = 0

    try:
        for idx, sample in enumerate(tqdm(all_samples, desc="degradation grounding")):
            img_obj = image_objects_map.get(sample.get("image_id", ""), {})
            try:
                result = analyze_sample(vlm, sample, img_obj)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                errors += 1
                continue
            except Exception as exc:
                errors += 1
                print(f"  [ERR] {sample.get('question_id')}: {exc}")
                import traceback
                traceback.print_exc()
                continue

            if result is None:
                skipped += 1
                continue

            # Per-sample file keeps summaries, not full curves.
            sample_record = {k: v for k, v in result.items() if k != "obj_results"}
            sample_record["objects_summary"] = [
                {
                    "obj_id": item["obj_id"],
                    "name": item["name"],
                    "operation": item["operation"],
                    "step_depth": item["step_depth"],
                    "n_tokens": item["n_tokens"],
                    "ablation_method": item.get("ablation_method", "mean"),
                    "curve_stats": item["curve_stats"],
                }
                for item in result["obj_results"]
            ]
            f_s.write(json.dumps(sample_record, default=_json_default) + "\n")

            for item in result["obj_results"]:
                f_o.write(json.dumps({
                    "question_id": result["question_id"],
                    "image_id": result["image_id"],
                    "model_correct": result["model_correct"],
                    "clean_logprob": result["clean_logprob"],
                    "operation_q": result["operation"],
                    **{k: v for k, v in item.items() if k != "degradation"},
                    "degradation": item["degradation"],
                }, default=_json_default) + "\n")

            for pair in result["pair_metrics"]:
                f_p.write(json.dumps(pair, default=_json_default) + "\n")
                collected_pairs.append(pair)

            collected_samples.append(sample_record)
            f_s.flush()
            f_o.flush()
            f_p.flush()

            if (idx + 1) % save_every == 0:
                interim = aggregate_results(collected_samples, collected_pairs)
                with open(os.path.join(output_dir, "aggregate_summary_interim.json"), "w") as f_interim:
                    json.dump(interim, f_interim, indent=2, default=_json_default)

                print(f"\n[{idx + 1}/{len(all_samples)}] acc={interim['dataset']['accuracy']:.3f}")
                print("  peak_value by op  (corr / incorr / d / p):")

                def _fmt(x):
                    return f"{x:.3f}" if x is not None else "N/A"

                for op, stats in interim["peak_value_by_operation"].items():
                    print(
                        f"    {op:12s}  "
                        f"{_fmt(stats['mean_peak_value_correct']):>6}  "
                        f"{_fmt(stats['mean_peak_value_incorrect']):>6}  "
                        f"d={_fmt(stats['cohen_d']):>6}  "
                        f"p={stats['mannwhitney_p']}"
                    )
                torch.cuda.empty_cache()

    finally:
        f_s.close()
        f_o.close()
        f_p.close()

    print(f"\nProcessed {len(collected_samples)} samples ({skipped} skipped, {errors} errors)")

    summary = aggregate_results(collected_samples, collected_pairs)
    with open(os.path.join(output_dir, "aggregate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print("\n" + "=" * 65)
    dataset_stats = summary["dataset"]
    print(f"Samples: {dataset_stats['n_samples']}  |  accuracy: {dataset_stats['accuracy']:.3f}")
    print("\nPEAK DEGRADATION BY OPERATION  (PRIMARY FINDING)")
    print(f"  {'op':12s}  {'n_c':>5}  {'n_i':>5}  {'pv_c':>7}  {'pv_i':>7}  {'cohen_d':>8}  {'MW_p':>10}")

    for op, stats in sorted(summary["peak_value_by_operation"].items()):
        def fmt(x):
            return f"{x:.3f}" if x is not None else "N/A"

        mw_p = stats["mannwhitney_p"]
        mw_p_str = str(round(mw_p, 5)) if mw_p is not None else "N/A"
        print(
            f"  {op:12s}  {stats['n_correct']:>5}  {stats['n_incorrect']:>5}  "
            f"{fmt(stats['mean_peak_value_correct']):>7}  "
            f"{fmt(stats['mean_peak_value_incorrect']):>7}  "
            f"{fmt(stats['cohen_d']):>8}  "
            f"{mw_p_str:>10}"
        )

    pair_ordering = summary["pair_ordering"]
    print(f"\nSECONDARY pair ordering: n={pair_ordering['n_pairs']}  rate={pair_ordering['ordering_rate']}")
    print(f"Results → {output_dir}")
    return summary


def main():
    """Parse CLI args and launch the experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--image_dir", default="data/images", help="Folder holding the images named in --samples")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", default="results/causal_mean_ablation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quantize_type", default="fp16", choices=["fp16", "int8", "int4", "4bit"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_objects", type=int, default=1)
    args = parser.parse_args()

    run(
        samples_path=args.samples,
        image_dir=args.image_dir,
        model_id=args.model_id,
        output_dir=args.output_dir,
        device=args.device,
        quantize_type=args.quantize_type,
        max_samples=args.max_samples,
        save_every=args.save_every,
        seed=args.seed,
        min_objects=args.min_objects,
    )


if __name__ == "__main__":
    main()