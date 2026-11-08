"""Analysis of the saved results. Needs no GPU and no model.

Produces:
  - accuracy vs. number of samples (results/voting_plot.png)
  - accuracy by difficulty, single sample vs. vote (results/difficulty_plot.png)
  - fixed/broken counts, consistent vs. uncertain errors, and pass@k

Usage:  python src/analyze.py --voting results/voting_200.json \
                              --baseline results/baseline_200.json
(--baseline is optional; it only adds the greedy line to the first plot.)
"""

import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset


def majority(preds):
    valid = [p for p in preds if p is not None]
    return Counter(valid).most_common(1)[0][0] if valid else None


def vote_acc(voted, k):
    """Accuracy if only the first k samples had been drawn."""
    return sum(majority(v["preds"][:k]) == v["gold"] for v in voted) / len(voted)


def n_steps(ex):
    """Difficulty proxy: lines in GSM8K's reference solution (one per step)."""
    solution = ex["answer"].split("####")[0].strip()
    return len([line for line in solution.split("\n") if line.strip()])


def bucket(s):
    if s <= 3:
        return "easy (≤3 steps)"
    return "medium (4 steps)" if s == 4 else "hard (5+ steps)"


def plot_samples(voted, greedy_acc, out):
    k_max = len(voted[0]["preds"])
    ks = list(range(1, k_max + 1))
    accs = [100 * vote_acc(voted, k) for k in ks]
    print("\nAccuracy vs. number of samples")
    for k, a in zip(ks, accs):
        print(f"  k={k}: {a:.1f}%")

    plt.figure()
    plt.plot(ks, accs, marker="o", label="Majority vote (temperature 0.7)")
    if greedy_acc is not None:
        plt.axhline(100 * greedy_acc, linestyle="--", color="gray",
                    label=f"Single greedy answer ({100 * greedy_acc:.0f}%)")
    plt.xlabel("Number of sampled answers")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Qwen2.5-1.5B-Instruct on {len(voted)} GSM8K test problems")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_difficulty(voted, steps, out):
    names = ["easy (≤3 steps)", "medium (4 steps)", "hard (5+ steps)"]
    single_acc, vote_acc_ = [], []
    print("\nAccuracy by difficulty (single sample vs. majority vote)")
    for name in names:
        g = [v for v in voted if bucket(steps[v["i"]]) == name]
        if not g:
            single_acc.append(0); vote_acc_.append(0)
            continue
        single = 100 * sum(v["preds"][0] == v["gold"] for v in g) / len(g)
        vote = 100 * sum(v["ok"] for v in g) / len(g)
        single_acc.append(single); vote_acc_.append(vote)
        print(f"  {name:18} n={len(g):3}  single {single:5.1f}%  vote {vote:5.1f}%  "
              f"gain {vote - single:+5.1f}")

    x = np.arange(len(names))
    plt.figure()
    plt.bar(x - 0.2, single_acc, 0.4, label="Single sampled answer")
    plt.bar(x + 0.2, vote_acc_, 0.4, label=f"Majority vote ({len(voted[0]['preds'])} samples)")
    plt.xticks(x, names)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Does voting help more on harder problems?")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def error_analysis(voted):
    k = len(voted[0]["preds"])
    fixed = sum(v["ok"] and v["preds"][0] != v["gold"] for v in voted)
    broken = sum(v["preds"][0] == v["gold"] and not v["ok"] for v in voted)
    print(f"\nCompared with a single sample, voting fixed {fixed} and broke {broken}")

    wrong = [v for v in voted if not v["ok"]]
    threshold = int(0.7 * k)
    consistent = 0
    for v in wrong:
        valid = [p for p in v["preds"] if p is not None]
        if valid and Counter(valid).most_common(1)[0][1] >= threshold:
            consistent += 1
    print(f"{len(wrong)} still wrong after voting; in {consistent} of them, "
          f"{threshold}+ of {k} samples agree on the same wrong answer")

    pass_k = sum(v["gold"] in v["preds"] for v in voted)
    print(f"pass@{k} (correct answer among the samples): {pass_k}/{len(voted)} "
          f"= {100 * pass_k / len(voted):.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voting", default="results/voting_200.json")
    ap.add_argument("--baseline", default="results/baseline_200.json")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    voted = json.load(open(args.voting))
    greedy_acc = None
    if os.path.exists(args.baseline):
        baseline = json.load(open(args.baseline))
        greedy_acc = sum(r["ok"] for r in baseline) / len(baseline)
        print(f"Greedy baseline: {100 * greedy_acc:.1f}%")
    print(f"Majority vote: {100 * vote_acc(voted, len(voted[0]['preds'])):.1f}%")

    ds = load_dataset("openai/gsm8k", "main", split="test")
    steps = {v["i"]: n_steps(ds[v["i"]]) for v in voted}
    print("Problems per step count:", sorted(Counter(steps.values()).items()))

    os.makedirs(args.outdir, exist_ok=True)
    plot_samples(voted, greedy_acc, os.path.join(args.outdir, "voting_plot.png"))
    plot_difficulty(voted, steps, os.path.join(args.outdir, "difficulty_plot.png"))
    error_analysis(voted)


if __name__ == "__main__":
    main()
