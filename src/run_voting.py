"""Self-consistency: sample k answers per problem and take a majority vote.

Saves every extracted answer, so accuracy for any smaller k can be
computed later without rerunning the model (see analyze.py).

Usage:  python src/run_voting.py --n 200 --k 10 --out results/voting_200.json
"""

import argparse
import json
import os
from collections import Counter

import torch

from common import extract, generate, gold_answer, load_gsm8k, load_model, to_number


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of test problems")
    ap.add_argument("--k", type=int, default=10, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/voting_200.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok, model = load_model()
    ds = load_gsm8k()

    # Resume if a partial file exists (Colab can disconnect).
    voted = json.load(open(args.out)) if os.path.exists(args.out) else []
    print(f"Starting from problem {len(voted)}")

    for i in range(len(voted), args.n):
        ex = ds[i]
        replies = generate(tok, model, ex["question"], n=args.k, do_sample=True,
                           temperature=args.temperature)
        preds = [to_number(extract(r)) for r in replies]
        gold = float(gold_answer(ex))
        valid = [p for p in preds if p is not None]
        vote = Counter(valid).most_common(1)[0][0] if valid else None
        voted.append({"i": i, "preds": preds, "gold": gold, "vote": vote, "ok": vote == gold})
        if (i + 1) % 10 == 0:
            json.dump(voted, open(args.out, "w"))
            acc = sum(v["ok"] for v in voted) / len(voted)
            print(f"{i + 1}/{args.n}, voting accuracy so far {acc:.1%}")

    json.dump(voted, open(args.out, "w"))
    print(f"Final voting accuracy: {sum(v['ok'] for v in voted)}/{len(voted)}")


if __name__ == "__main__":
    main()
