import json
import time

from agent import triage

DATASET_PATH = "dataset.json"
OUTPUT_PATH = "eval_results_v0.json"
CATEGORIES = ["bug", "billing", "feature_request", "how_to", "churn_risk"]


def run_eval():
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    results = []
    for i, case in enumerate(dataset, 1):
        print(f"  [{i:02d}/{len(dataset)}] case {case['id']} ...", end=" ", flush=True)
        try:
            prediction = triage(case["message"])
            cat_correct = prediction.get("category") == case["label"]["category"]
            pri_correct = prediction.get("priority") == case["label"]["priority"]
            print(
                f"cat={'✓' if cat_correct else '✗'} pri={'✓' if pri_correct else '✗'}"
                f"  predicted={prediction.get('category')}/{prediction.get('priority')}"
                f"  expected={case['label']['category']}/{case['label']['priority']}"
            )
        except Exception as e:
            prediction = {"error": str(e)}
            cat_correct = False
            pri_correct = False
            print(f"ERROR: {e}")

        results.append(
            {
                "id": case["id"],
                "case_type": case.get("case_type"),
                "message": case["message"],
                "label": case["label"],
                "prediction": prediction,
                "category_correct": cat_correct,
                "priority_correct": pri_correct,
            }
        )

        if i < len(dataset):
            time.sleep(0.3)  # avoid rate-limit bursts

    return results


def compute_stats(results):
    total = len(results)
    cat_correct = sum(r["category_correct"] for r in results)
    pri_correct = sum(r["priority_correct"] for r in results)

    per_category = {}
    for cat in CATEGORIES:
        cases = [r for r in results if r["label"]["category"] == cat]
        if cases:
            correct = sum(r["category_correct"] for r in cases)
            per_category[cat] = {
                "total": len(cases),
                "correct": correct,
                "accuracy": round(correct / len(cases), 3),
            }

    return {
        "total": total,
        "category_accuracy": round(cat_correct / total, 3),
        "priority_accuracy": round(pri_correct / total, 3),
        "per_category": per_category,
    }


def print_summary(summary):
    print()
    print("=" * 52)
    print("EVAL RESULTS  v0")
    print("=" * 52)
    print(f"Overall category accuracy : {summary['category_accuracy']:.0%}  ({int(summary['category_accuracy']*summary['total'])}/{summary['total']})")
    print(f"Overall priority accuracy : {summary['priority_accuracy']:.0%}  ({int(summary['priority_accuracy']*summary['total'])}/{summary['total']})")
    print()
    print(f"{'Category':<20} {'Correct':>7} {'Total':>7} {'Accuracy':>9}")
    print("-" * 47)
    for cat, s in summary["per_category"].items():
        print(f"{cat:<20} {s['correct']:>7} {s['total']:>7} {s['accuracy']:>8.0%}")
    print()
    print(f"Full results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    print(f"Running eval on {DATASET_PATH} …\n")
    results = run_eval()
    summary = compute_stats(results)

    output = {"summary": summary, "cases": results}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print_summary(summary)
