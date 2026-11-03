"""LLM-as-judge calibration: compare judge verdicts with human labels on 20 eval runs.

    python -m eval.calibration sample            # pick 20 runs from reports/eval/latest/results.json
    python -m eval.calibration label             # label them yourself (y/n), judge scores are hidden
    python -m eval.calibration report            # agreement per metric, flags unreliable metrics

Human labels are binary: is the answer faithful to the context? relevant to the
question? is its behaviour (answer / partial / decline) correct? The judge (or
deterministic metric) passes a run when its score clears the metric threshold.
A metric is flagged unreliable when Cohen's kappa < 0.6 or accuracy < 0.8, or
when fewer than 10 labelled items cover it.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from app.config import ROOT
from eval.config import REPORTS_DIR, load_eval_config
from eval.suite_report import md_table, write_suite_report

LABELS_PATH = ROOT / "eval" / "calibration" / "labels.json"
# metric name -> human label field
PAIRS = {"faithfulness": "faithful", "answer_relevancy": "relevant", "refusal_correctness": "behaviour_correct"}
MIN_KAPPA, MIN_ACCURACY, MIN_ITEMS = 0.6, 0.8, 10


def sample(results_path: Path, n: int = 20, seed: int = 13) -> list[dict]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    pool = []
    for case in data["cases"]:
        for run in case["runs"]:
            if run["reply"]["error"] is None:
                pool.append({
                    "item_id": f"{case['case']['id']}#run{run['run_index'] + 1}",
                    "category": case["case"]["category"],
                    "expected_behavior": case["case"]["expected_behavior"],
                    "question": case["case"]["question"],
                    "retrieved": [p["passage"] for p in run["reply"]["retrieved"]],
                    "answer": run["answer"],
                    "judge_scores": run["scores"],
                    "human": {field: None for field in PAIRS.values()},
                })
    rng = random.Random(seed)
    by_cat: dict[str, list] = {}
    for item in pool:
        by_cat.setdefault(item["category"], []).append(item)
    picked = []
    while len(picked) < n and any(by_cat.values()):  # round-robin over categories for coverage
        for items in by_cat.values():
            if items and len(picked) < n:
                picked.append(items.pop(rng.randrange(len(items))))
    return picked


def label(items: list[dict], ask=input) -> list[dict]:
    for i, item in enumerate(items, 1):
        if all(v is not None for v in item["human"].values()):
            continue
        print("\n" + "=" * 80 + f"\n[{i}/{len(items)}] {item['item_id']} ({item['category']}, expected: {item['expected_behavior']})")
        print(f"\nQUESTION: {item['question']}\n\nRETRIEVED CONTEXT:")
        for p in item["retrieved"] or ["(nothing retrieved)"]:
            print(f"  - {p[:400]}")
        print(f"\nANSWER: {item['answer']}\n")
        for field, prompt in (("faithful", "Every claim supported by the context?"),
                              ("relevant", "Answers what was asked?"),
                              ("behaviour_correct", f"Correct behaviour for '{item['expected_behavior']}'?")):
            reply = ""
            while reply not in ("y", "n", "s"):
                reply = ask(f"{prompt} [y/n/s=skip] ").strip().lower()
            item["human"][field] = {"y": True, "n": False, "s": None}[reply]
    return items


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(items: list[dict], thresholds: dict[str, float]) -> dict[str, dict]:
    out = {}
    for metric, field in PAIRS.items():
        pairs = [(item["judge_scores"][metric] >= thresholds[metric], item["human"][field])
                 for item in items
                 if item["human"].get(field) is not None and item["judge_scores"].get(metric) is not None]
        judge, human = [p[0] for p in pairs], [p[1] for p in pairs]
        n = len(pairs)
        acc = sum(j == h for j, h in pairs) / n if n else 0.0
        kappa = cohen_kappa(judge, human)
        reasons = []
        if n < MIN_ITEMS:
            reasons.append(f"only {n} labelled items")
        if n and kappa < MIN_KAPPA:
            reasons.append(f"kappa {kappa:.2f} < {MIN_KAPPA}")
        if n and acc < MIN_ACCURACY:
            reasons.append(f"accuracy {acc:.0%} < {MIN_ACCURACY:.0%}")
        out[metric] = {
            "n": n, "accuracy": acc, "kappa": kappa,
            "judge_pass_human_fail": sum(j and not h for j, h in pairs),
            "judge_fail_human_pass": sum(h and not j for j, h in pairs),
            "reliable": not reasons, "reasons": reasons,
        }
    return out


def render(result: dict) -> str:
    lines = ["# Judge calibration: judge vs human labels", "",
             "Judge = DeepEval metric (or the deterministic refusal detector) passing at its threshold.", ""]
    lines += md_table(
        ["Metric", "Labelled", "Accuracy", "Cohen's kappa", "Judge too lenient", "Judge too strict", "Reliable?"],
        [[m, r["n"], f"{r['accuracy']:.0%}", f"{r['kappa']:.2f}", r["judge_pass_human_fail"], r["judge_fail_human_pass"],
          "yes" if r["reliable"] else "**no**: " + "; ".join(r["reasons"])] for m, r in result.items()],
    )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sample", "label", "report"])
    parser.add_argument("--results", type=Path, default=REPORTS_DIR / "eval" / "latest" / "results.json")
    parser.add_argument("-n", type=int, default=20)
    args = parser.parse_args(argv)

    if args.command == "sample":
        if not args.results.exists():
            print(f"No eval results at {args.results}. Run `python -m eval.run --profile full` first.", file=sys.stderr)
            return 2
        items = sample(args.results, args.n)
        LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LABELS_PATH.write_text(json.dumps({"source": str(args.results), "items": items}, indent=2), encoding="utf-8")
        print(f"Wrote {len(items)} items to {LABELS_PATH}. Next: python -m eval.calibration label")
        return 0

    data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if args.command == "label":
        label(data["items"])
        LABELS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved labels to {LABELS_PATH}")
        return 0

    result = agreement(data["items"], load_eval_config("full").thresholds)
    out = write_suite_report("calibration", render(result), {"agreement": result})
    print(render(result))
    print(f"Report: {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
