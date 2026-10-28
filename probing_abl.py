#!/usr/bin/env python3

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from HookedQwenVL import HookedQwenVL


class LinearProbe(nn.Module):
    """Single linear layer probe."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class HiddenStateCollector:
    """Collect hidden states and short generations from Qwen-VL."""

    def __init__(self, model_id: str, device: str = "cuda:0"):
        print(f"Loading {model_id} ...")
        self.vlm = HookedQwenVL(
            model_id=model_id,
            device=device,
            quantize=True,
            quantize_type="fp16",
        )
        self.vlm.model.eval()
        self.device = device
        self.n_layers = self.vlm.model.config.num_hidden_layers
        self.hidden_dim = self.vlm.model.config.hidden_size
        print(f"  {self.n_layers} layers  hidden_dim={self.hidden_dim}")

    def _inputs_with_image(self, image_path: str, question: str):
        img = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ],
        }]
        text = self.vlm.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.vlm.processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    def _inputs_text_only(self, question: str):
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        text = self.vlm.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.vlm.processor(text=[text], padding=True, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def hidden_states(self, image_path: Optional[str], question: str, position: int = -1) -> Optional[List[torch.Tensor]]:
        """Return one residual vector per layer at the requested token position."""
        try:
            inputs = self._inputs_with_image(image_path, question) if image_path else self._inputs_text_only(question)
            outputs = self.vlm.model(**inputs, output_hidden_states=True)
            states = [
                outputs.hidden_states[layer_idx + 1][0, position, :].detach().cpu()
                for layer_idx in range(self.n_layers)
            ]
            del outputs
            torch.cuda.empty_cache()
            return states
        except Exception as exc:
            print(f"  [ERR] {exc}")
            return None

    @torch.no_grad()
    def generate_first_word(self, image_path: Optional[str], question: str) -> Optional[str]:
        """Generate a short answer and return only the first decoded word."""
        try:
            inputs = self._inputs_with_image(image_path, question) if image_path else self._inputs_text_only(question)
            output = self.vlm.model.generate(**inputs, max_new_tokens=5, do_sample=False)
            decoded = self.vlm.processor.tokenizer.decode(
                output[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip().lower()
            return decoded.split()[0] if decoded else None
        except Exception:
            return None


def _train_one_probe(
    X_tr,
    y_tr,
    X_va,
    y_va,
    hidden_dim: int,
    num_classes: int,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
) -> dict:
    """Train one linear probe and return train/validation metrics."""
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = torch.tensor(X_va, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.long)

    probe = LinearProbe(hidden_dim, num_classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    best_val = 0.0
    patience = 0
    probe.train()

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            criterion(probe(batch_x), batch_y).backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            probe.eval()
            with torch.no_grad():
                val_acc = (probe(Xv).argmax(1) == yv).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                patience = 0
            else:
                patience += 1
            if patience >= 5:
                break
            probe.train()

    probe.eval()
    with torch.no_grad():
        train_pred = probe(Xt).argmax(1).numpy()
        val_pred = probe(Xv).argmax(1).numpy()

    return {
        "train_acc": float(accuracy_score(y_tr, train_pred)),
        "val_acc": float(accuracy_score(y_va, val_pred)),
        "val_f1": float(f1_score(y_va, val_pred, average="weighted")),
    }


def _pair_split(pair_ids: List[str], labels: List[int], test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Split at pair level so the same question pair never appears in both splits."""
    unique_pairs = list(dict.fromkeys(pair_ids))
    pair_labels = [labels[pair_ids.index(pair_id)] for pair_id in unique_pairs]

    try:
        train_pairs, val_pairs = train_test_split(
            unique_pairs,
            test_size=test_size,
            random_state=42,
            stratify=pair_labels,
        )
    except ValueError:
        train_pairs, val_pairs = train_test_split(
            unique_pairs,
            test_size=test_size,
            random_state=42,
        )

    train_pairs = set(train_pairs)
    val_pairs = set(val_pairs)
    idx_tr = np.array([i for i, pair_id in enumerate(pair_ids) if pair_id in train_pairs])
    idx_va = np.array([i for i, pair_id in enumerate(pair_ids) if pair_id in val_pairs])
    return idx_tr, idx_va


def train_all_layers(
    hidden_by_layer: Dict[int, List[torch.Tensor]],
    labels: List[int],
    n_layers: int,
    hidden_dim: int,
    num_classes: int,
    pair_ids: Optional[List[str]] = None,
) -> Dict[int, dict]:
    """Train one probe per layer."""
    n = len(labels)
    if n < 30:
        print(f"  [SKIP] only {n} samples")
        return {}

    y = np.array(labels)
    if pair_ids:
        idx_tr, idx_va = _pair_split(pair_ids, labels)
    else:
        try:
            idx_tr, idx_va = train_test_split(np.arange(n), test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            idx_tr, idx_va = train_test_split(np.arange(n), test_size=0.2, random_state=42)

    y_tr, y_va = y[idx_tr], y[idx_va]
    results = {}

    for layer_idx in tqdm(range(n_layers), desc="  probes", leave=False):
        X = torch.stack(hidden_by_layer[layer_idx]).numpy()
        results[layer_idx] = _train_one_probe(
            X[idx_tr],
            y_tr,
            X[idx_va],
            y_va,
            hidden_dim,
            num_classes,
        )

    return results


def collect_contrastive(
    samples: List[dict],
    collector: HiddenStateCollector,
    op_filter: Optional[str],
    use_image: bool,
    max_n: int,
) -> Tuple[Dict, List[int], List[str], List[dict]]:
    """Collect clean/corrupted hidden states and mark each pair as VS or NV."""
    pool = [
        sample for sample in samples
        if sample.get("question_corrupted")
        and (op_filter is None or sample["operation"].split()[0] == op_filter)
    ]
    random.shuffle(pool)
    pool = pool[:max_n]

    hidden_by_layer: Dict[int, list] = defaultdict(list)
    labels: List[int] = []
    pair_ids: List[str] = []
    metadata: List[dict] = []

    desc = f"  {'img' if use_image else 'txt'} ({op_filter or 'all'})"
    for sample in tqdm(pool, desc=desc, leave=False):
        image_path = sample["image_path"] if use_image else None
        question_id = sample["question_id"]

        states_clean = collector.hidden_states(image_path, sample["question_clean"])
        states_corrupt = collector.hidden_states(image_path, sample["question_corrupted"])
        if states_clean is None or states_corrupt is None:
            continue

        ans_clean = collector.generate_first_word(image_path, sample["question_clean"])
        ans_corrupt = collector.generate_first_word(image_path, sample["question_corrupted"])
        vision_sensitive = ans_clean is not None and ans_corrupt is not None and ans_clean != ans_corrupt
        correct_clean = (ans_clean == sample["answer"].strip().lower()) if ans_clean else None

        for layer_idx in range(collector.n_layers):
            hidden_by_layer[layer_idx].append(states_clean[layer_idx])
            hidden_by_layer[layer_idx].append(states_corrupt[layer_idx])

        labels.extend([0, 1])
        pair_ids.extend([question_id, question_id])
        metadata.append({
            "question_id": question_id,
            "ans_clean": ans_clean,
            "ans_corrupted": ans_corrupt,
            "correct_clean": correct_clean,
            "vision_sensitive": vision_sensitive,
        })

    n_vs = sum(item["vision_sensitive"] for item in metadata)
    print(
        f"    {len(metadata)} pairs  vision-sensitive: "
        f"{n_vs}/{len(metadata)} ({100 * n_vs / max(len(metadata), 1):.1f}%)"
    )
    return dict(hidden_by_layer), labels, pair_ids, metadata


def split_by_sensitivity(
    hidden_by_layer: Dict,
    labels: List[int],
    pair_ids: List[str],
    metadata: List[dict],
) -> Tuple[Dict, List, List, Dict, List, List]:
    """Split contrastive examples into vision-sensitive and non-sensitive subsets."""
    vs_set = {item["question_id"] for item in metadata if item["vision_sensitive"]}
    nv_set = {item["question_id"] for item in metadata if not item["vision_sensitive"]}

    def subset(keep: set):
        idx = [i for i, pair_id in enumerate(pair_ids) if pair_id in keep]
        return (
            {layer_idx: [hidden_by_layer[layer_idx][i] for i in idx] for layer_idx in hidden_by_layer},
            [labels[i] for i in idx],
            [pair_ids[i] for i in idx],
        )

    return (*subset(vs_set), *subset(nv_set))


def analyze_curve(results: Dict[int, dict], num_classes: int, task: str) -> dict:
    """Summarize layerwise probe results."""
    if not results:
        return {"task": task, "error": "no results"}

    chance = 1.0 / num_classes
    layers = sorted(results.keys())
    accs = [results[layer_idx]["val_acc"] for layer_idx in layers]
    best_idx = int(np.argmax(accs))
    best_layer = layers[best_idx]
    best_acc = accs[best_idx]
    emergence_layer = next((layer_idx for layer_idx, acc in zip(layers, accs) if acc > chance + 0.10), None)

    return {
        "task": task,
        "chance": round(chance, 4),
        "best_layer": best_layer,
        "best_acc": round(best_acc, 4),
        "above_chance": round(best_acc - chance, 4),
        "emergence_layer": emergence_layer,
        "acc_by_layer": {layer_idx: round(results[layer_idx]["val_acc"], 4) for layer_idx in layers},
        "train_acc_by_layer": {layer_idx: round(results[layer_idx]["train_acc"], 4) for layer_idx in layers},
    }


def delta_visual(img_curve: dict, txt_curve: dict) -> dict:
    """Compute image minus text-only probe accuracy at each layer."""
    layers = sorted(img_curve["acc_by_layer"].keys())
    delta = {
        layer_idx: round(img_curve["acc_by_layer"][layer_idx] - txt_curve["acc_by_layer"].get(layer_idx, 0.5), 4)
        for layer_idx in layers
    }
    max_layer = max(delta, key=delta.get)
    emergence = next((layer_idx for layer_idx in layers if delta[layer_idx] > 0.05), None)
    return {
        "delta_by_layer": delta,
        "max_delta": round(delta[max_layer], 4),
        "max_delta_layer": max_layer,
        "vision_emergence": emergence,
    }


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_sensitivity_comparison(a_all: dict, a_vs: dict, a_nv: dict, op: str, path: str):
    """Plot all/VS/NV probe curves for one operation."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(13, 5))

    for label, analysis, color, linestyle in [
        ("All pairs", a_all, "#1f77b4", "-"),
        ("Vision-sensitive", a_vs, "#2ca02c", "-"),
        ("Vision-insensitive", a_nv, "#d62728", "--"),
    ]:
        if not analysis or "acc_by_layer" not in analysis:
            continue
        layers = sorted(analysis["acc_by_layer"].keys())
        accs = [analysis["acc_by_layer"][layer_idx] for layer_idx in layers]
        ax.plot(layers, accs, f"{linestyle}o", ms=3, lw=2, color=color, label=f"{label} (best={analysis.get('best_acc', 0):.3f})")

    ax.axhline(0.5, color="gray", ls=":", lw=1.5, label="Chance 0.50")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Val Accuracy")
    ax.set_title(
        f"{op}: Counterfactual Sensitivity — Vision-Sensitive Split\n"
        "VS > NV → probe reads visual computation not surface text",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_sensitivity_summary(summary: Dict[str, dict], path: str):
    """Bar chart of best accuracy for all/VS/NV per operation."""
    plt = _plt()
    ops = sorted(summary.keys())
    x = np.arange(len(ops))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    for offset, key, color, label in [
        (-width, "best_all", "#1f77b4", "All pairs"),
        (0, "best_vs", "#2ca02c", "Vision-sensitive"),
        (+width, "best_nv", "#d62728", "Vision-insensitive"),
    ]:
        vals = [summary[op].get(key, 0.5) for op in ops]
        ax.bar(x + offset, vals, width, color=color, alpha=0.85, label=label)

    ax.axhline(0.5, color="gray", ls="--", lw=1.5, label="Chance 0.50")
    ax.set_xticks(x)
    ax.set_xticklabels(ops, fontsize=11)
    ax.set_ylabel("Best val accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(
        "Counterfactual Sensitivity: Vision-Sensitive vs Insensitive Pairs\n"
        "VS > NV confirms probe reads visual computation",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_delta_validation(deltas: Dict[str, dict], path: str):
    """Plot Δ_visual curves for each operation."""
    plt = _plt()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    fig, ax = plt.subplots(figsize=(14, 5))

    for idx, (op, delta_info) in enumerate(deltas.items()):
        layers = sorted(delta_info["delta_by_layer"].keys())
        vals = [delta_info["delta_by_layer"][layer_idx] for layer_idx in layers]
        emergence = delta_info.get("vision_emergence")
        label = f"{op} (emerge L{emergence})" if emergence else f"{op} (no signal)"
        ax.plot(layers, vals, "-o", ms=3, lw=2, color=colors[idx % len(colors)], label=label)

    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.05, color="orange", lw=1, ls=":", label="Δ=0.05 threshold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ_visual (img − text-only)")
    ax.set_title("Probe Validation: Layer at Which Vision Signal Exceeds Text Signal", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def load_samples(path: str, image_dir: Optional[str] = None) -> List[dict]:
    """Load samples from either flat-list or image-grouped JSON format."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    elif "images" in data:
        samples = []
        for img_data in data["images"].values():
            for sample in img_data.get("samples", []):
                sample.setdefault("image_path", img_data.get("image_path", ""))
                samples.append(sample)
    else:
        raise ValueError(f"Unknown format: {path}")

    # Image file names are stored without a folder; join them onto --image_dir.
    if image_dir:
        for sample in samples:
            sample["image_path"] = os.path.join(image_dir, sample["image_path"])
    missing = sum(1 for p in {s["image_path"] for s in samples} if not os.path.exists(p))
    if missing:
        print(f"  [WARN] {missing} image files not found (check --image_dir)")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="Path to probing_operations.json (must have question_corrupted)")
    parser.add_argument("--image_dir", default="data/images", help="Folder holding the images named in --samples")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_per_op", type=int, default=500)
    parser.add_argument("--skip", nargs="*", default=[], help="Skip experiments: sensitivity, answer")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    output_dir = args.output_dir or str(Path.cwd() / "results" / "probing")
    os.makedirs(output_dir, exist_ok=True)

    samples = load_samples(args.samples, args.image_dir)
    op_counts = Counter(sample["operation"].split()[0] for sample in samples)
    operations = sorted(op_counts.keys())
    has_corrupt = sum(1 for sample in samples if sample.get("question_corrupted"))

    print(f"Loaded {len(samples)} samples  ops={dict(op_counts)}")
    print(f"With corrupted questions: {has_corrupt}/{len(samples)}")
    if has_corrupt < 50:
        print(
            "[FATAL] This experiment requires question_corrupted field.\n"
            "        Use data/probing_operations.json, not data/GQA_3500.json."
        )
        return

    collector = HiddenStateCollector(args.model_id, args.device)
    n_layers = collector.n_layers
    hidden_dim = collector.hidden_dim
    all_results = {}

    if "sensitivity" not in args.skip:
        print("\n" + "=" * 70)
        print("EXP 1: Counterfactual Sensitivity + Vision-Sensitive Split")
        print("=" * 70)

        sensitivity_summary = {}
        deltas = {}

        for op in operations:
            print(f"\n── {op} ──")

            h_img, l_img, p_img, metadata = collect_contrastive(
                samples,
                collector,
                op,
                use_image=True,
                max_n=args.max_per_op,
            )
            if len(l_img) < 40:
                print(f"  [SKIP] too few samples ({len(l_img)})")
                continue

            res_all = train_all_layers(h_img, l_img, n_layers, hidden_dim, 2, p_img)
            a_all = analyze_curve(res_all, 2, f"{op} all-pairs")
            print(f"  All pairs:  best L{a_all['best_layer']}={a_all['best_acc']:.3f}  emerge L{a_all.get('emergence_layer', '—')}")

            h_vs, l_vs, p_vs, h_nv, l_nv, p_nv = split_by_sensitivity(h_img, l_img, p_img, metadata)
            a_vs, a_nv = {}, {}

            if len(l_vs) >= 30:
                res_vs = train_all_layers(h_vs, l_vs, n_layers, hidden_dim, 2, p_vs)
                a_vs = analyze_curve(res_vs, 2, f"{op} vision-sensitive")
                print(f"  VS pairs:   best L{a_vs['best_layer']}={a_vs['best_acc']:.3f}  (n={len(l_vs) // 2} pairs)")
            else:
                print(f"  VS pairs:   [SKIP] only {len(l_vs) // 2} pairs")

            if len(l_nv) >= 30:
                res_nv = train_all_layers(h_nv, l_nv, n_layers, hidden_dim, 2, p_nv)
                a_nv = analyze_curve(res_nv, 2, f"{op} vision-insensitive")
                print(f"  NV pairs:   best L{a_nv['best_layer']}={a_nv['best_acc']:.3f}  (n={len(l_nv) // 2} pairs)")
            else:
                print(f"  NV pairs:   [SKIP] only {len(l_nv) // 2} pairs")

            vs_acc = a_vs.get("best_acc", 0.5) if a_vs else 0.5
            nv_acc = a_nv.get("best_acc", 0.5) if a_nv else 0.5
            gap = vs_acc - nv_acc
            verdict = "probe reads VISION ✓" if gap > 0.05 else "probe reads LANGUAGE — artifact risk"
            print(f"  VS − NV gap: {gap:+.3f}  ({verdict})")

            text_hidden: Dict[int, list] = defaultdict(list)
            l_txt, p_txt = [], []
            txt_pool = [
                sample for sample in samples
                if sample.get("question_corrupted") and sample["operation"].split()[0] == op
            ]
            random.shuffle(txt_pool)

            for sample in tqdm(txt_pool[:args.max_per_op], desc=f"  txt-only ({op})", leave=False):
                states_clean = collector.hidden_states(None, sample["question_clean"])
                states_corrupt = collector.hidden_states(None, sample["question_corrupted"])
                if states_clean is None or states_corrupt is None:
                    continue
                for layer_idx in range(n_layers):
                    text_hidden[layer_idx].append(states_clean[layer_idx])
                    text_hidden[layer_idx].append(states_corrupt[layer_idx])
                l_txt.extend([0, 1])
                p_txt.extend([sample["question_id"]] * 2)

            if len(l_txt) >= 40:
                res_txt = train_all_layers(dict(text_hidden), l_txt, n_layers, hidden_dim, 2, p_txt)
                a_txt = analyze_curve(res_txt, 2, f"{op} text-only")
                d = delta_visual(a_all, a_txt)
                deltas[op] = d
                all_results[f"delta_{op}"] = d
                print(
                    f"  Text-only:  best L{a_txt['best_layer']}={a_txt['best_acc']:.3f}  "
                    f"Δ_max={d['max_delta']:.3f} @L{d['max_delta_layer']}  "
                    f"vision emerge L{d.get('vision_emergence', '—')}"
                )

            sensitivity_summary[op] = {
                "best_all": a_all.get("best_acc", 0.5),
                "best_vs": vs_acc,
                "best_nv": nv_acc,
                "vs_nv_gap": round(gap, 4),
                "emerge_all": a_all.get("emergence_layer"),
                "emerge_vs": a_vs.get("emergence_layer") if a_vs else None,
            }
            all_results[f"sensitivity_{op}_all"] = a_all
            all_results[f"sensitivity_{op}_vs"] = a_vs
            all_results[f"sensitivity_{op}_nv"] = a_nv

            plot_sensitivity_comparison(
                a_all,
                a_vs,
                a_nv,
                op,
                os.path.join(output_dir, f"{op}_sensitivity_split.png"),
            )

        if sensitivity_summary:
            plot_sensitivity_summary(sensitivity_summary, os.path.join(output_dir, "sensitivity_summary.png"))
        if deltas:
            plot_delta_validation(deltas, os.path.join(output_dir, "delta_visual_validation.png"))

        print("\n" + "=" * 70)
        print("COUNTERFACTUAL SENSITIVITY SUMMARY")
        print("=" * 70)
        print(f"  {'op':10s}  {'all':>6}  {'VS':>6}  {'NV':>6}  {'VS-NV':>7}  {'emerge_all':>11}  {'emerge_VS':>10}  interpretation")
        print("  " + "─" * 75)
        for op, summary in sensitivity_summary.items():
            flag = "vision ✓" if summary["vs_nv_gap"] > 0.05 else "language artifact"
            print(
                f"  {op:10s}  {summary['best_all']:>6.3f}  {summary['best_vs']:>6.3f}  "
                f"{summary['best_nv']:>6.3f}  {summary['vs_nv_gap']:>+7.3f}  "
                f"{str(summary['emerge_all']):>11}  {str(summary['emerge_vs']):>10}  {flag}"
            )
        all_results["sensitivity_summary"] = sensitivity_summary

    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results → {output_dir}/")


if __name__ == "__main__":
    main()
