"""Greedy baseline: one deterministic answer per problem.

Usage:  python src/run_baseline.py --n 200 --out results/baseline_200.json
"""

import argparse
import json
import os

from common import extract, generate, gold_answer, load_gsm8k, load_model, to_number


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of test problems")
    ap.add_argument("--out", default="results/baseline_200.json")
    args = ap.parse_args()

    tok, model = load_model()
    ds = load_gsm8k()

    # Resume if a partial file exists (Colab can disconnect).
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    print(f"Starting from problem {len(results)}")

    for i in range(len(results), args.n):
        ex = ds[i]
        reply = generate(tok, model, ex["question"], do_sample=False)[0]
        pred, gold = extract(reply), gold_answer(ex)
        ok = to_number(pred) is not None and to_number(pred) == float(gold)
        results.append({"i": i, "reply": reply, "pred": pred, "gold": gold, "ok": ok})
        if (i + 1) % 10 == 0:
            json.dump(results, open(args.out, "w"))
            acc = sum(r["ok"] for r in results) / len(results)
            print(f"{i + 1}/{args.n} done, accuracy so far {acc:.1%}")

    json.dump(results, open(args.out, "w"))
    print(f"Final greedy accuracy: {sum(r['ok'] for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()
