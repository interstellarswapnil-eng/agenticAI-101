import json
import os
import sys
import time

from agent import triage
from llm_judge import run_judge, DIMENSIONS

DATASET_PATH = "dataset.json"
CALIBRATION_PATH = "calibration_human_scores.json"
CALIBRATION_OUTPUT_PATH = "calibration_results.json"
JUDGE_OUTPUT_PATH = "judge_results.json"

AGREEMENT_THRESHOLD = 1  # |human - judge| <= 1 counts as agreement

# 10 cases chosen for calibration: 3 clear, 4 ambiguous, 3 adversarial
CALIBRATION_IDS = ["001", "003", "005", "011", "014", "016", "018", "021", "023", "027"]


def _load_dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


def generate_calibration_template():
    """Run triage on the 10 calibration cases and write a scoreable template.

    The user hand-scores each response_draft, then runs --calibrate.
    The judge will score the same response_draft the human scored.
    """
    dataset = _load_dataset()
    cases_by_id = {c["id"]: c for c in dataset}

    template = []
    total = len(CALIBRATION_IDS)
    for i, cid in enumerate(CALIBRATION_IDS, 1):
        case = cases_by_id[cid]
        print(f"  [{i:02d}/{total}] case {cid} ({case['case_type']}) ...", end=" ", flush=True)
        try:
            prediction = triage(case["message"])
            response_draft = prediction.get("response_draft", "")
            print("ok")
        except Exception as e:
            response_draft = ""
            print(f"ERROR: {e}")

        template.append({
            "case_id": cid,
            "case_type": case["case_type"],
            "message": case["message"],
            "response_draft": response_draft,
            "human_scores": {
                "tone_appropriateness": None,
                "issue_addressed": None,
                "actionability": None,
            },
            "notes": "",
        })

        if i < total:
            time.sleep(0.3)

    with open(CALIBRATION_PATH, "w") as f:
        json.dump(template, f, indent=2)

    print(f"\nTemplate written to {CALIBRATION_PATH}")
    print("Next: open the file, read each response_draft, fill in human_scores (1-5), then run --calibrate.")


def run_calibration():
    """Compare judge scores against human scores for the 10 calibration cases."""
    with open(CALIBRATION_PATH) as f:
        calibration_cases = json.load(f)

    unfilled = [
        c["case_id"] for c in calibration_cases
        if any(v is None for v in c["human_scores"].values())
    ]
    if unfilled:
        print(f"Human scores missing for cases: {unfilled}")
        print("Fill in all human_scores in calibration_human_scores.json first.")
        sys.exit(1)

    results = []
    total = len(calibration_cases)
    for i, case in enumerate(calibration_cases, 1):
        cid = case["case_id"]
        print(f"  [{i:02d}/{total}] case {cid} ...", end=" ", flush=True)

        try:
            scores = run_judge(case["message"], case["response_draft"])
        except Exception as e:
            print(f"ERROR (judge): {e}")
            continue

        judge_scores = {r["name"]: r["score"] for r in scores}
        human_scores = case["human_scores"]

        agreements = {}
        divergences = []
        for dim in DIMENSIONS:
            h = human_scores[dim]
            j = judge_scores[dim]
            agrees = abs(h - j) <= AGREEMENT_THRESHOLD
            agreements[dim] = agrees
            if not agrees:
                divergences.append({
                    "case_id": cid,
                    "dimension": dim,
                    "human": h,
                    "judge": j,
                    "gap": j - h,
                    "judge_reason": next(r["reason"] for r in scores if r["name"] == dim),
                })

        agreed = sum(agreements.values())
        print(f"agreement {agreed}/{len(DIMENSIONS)}")

        results.append({
            "case_id": cid,
            "case_type": case.get("case_type"),
            "message": case["message"],
            "response_draft": case["response_draft"],
            "human_scores": human_scores,
            "judge_scores": judge_scores,
            "judge_reasons": {r["name"]: r["reason"] for r in scores},
            "agreements": agreements,
            "divergences": divergences,
        })

        if i < total:
            time.sleep(0.3)

    # Aggregate
    per_dim = {}
    for dim in DIMENSIONS:
        agreed_count = sum(1 for r in results if r["agreements"].get(dim))
        per_dim[dim] = {
            "agreed": agreed_count,
            "total": len(results),
            "agreement_rate": round(agreed_count / len(results), 2),
        }

    all_divergences = []
    for r in results:
        all_divergences.extend(r["divergences"])

    total_scored = sum(v["total"] for v in per_dim.values())
    total_agreed = sum(v["agreed"] for v in per_dim.values())
    overall = round(total_agreed / total_scored, 2) if total_scored else None

    output = {
        "per_dimension_agreement": per_dim,
        "overall_agreement_rate": overall,
        "divergences": all_divergences,
        "per_case": results,
    }

    with open(CALIBRATION_OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"   {'Dimension':<26} Agreement")
    for dim in DIMENSIONS:
        s = per_dim[dim]
        print(f"   {dim:<26} {s['agreed']}/{s['total']} ({s['agreement_rate']:.0%})")
    print(f"\n   Overall within ±1: {total_agreed}/{total_scored} ({overall:.0%})")
    print(f"\nFull calibration results written to {CALIBRATION_OUTPUT_PATH}")

    if all_divergences:
        print(f"\n{len(all_divergences)} divergence(s) to review — read {CALIBRATION_OUTPUT_PATH} for details.")


def run_judge_eval():
    """Run the judge on all 30 dataset cases and write judge_results.json."""
    dataset = _load_dataset()
    total = len(dataset)
    per_case = []
    dim_scores = {dim: [] for dim in DIMENSIONS}

    for i, case in enumerate(dataset, 1):
        print(f"  [{i:02d}/{total}] case {case['id']} ...", end=" ", flush=True)

        try:
            prediction = triage(case["message"])
            response_draft = prediction.get("response_draft", "")
        except Exception as e:
            print(f"ERROR (agent): {e}")
            continue

        try:
            scores = run_judge(case["message"], response_draft)
        except Exception as e:
            print(f"ERROR (judge): {e}")
            continue

        total_score = sum(r["score"] for r in scores)
        print(f"total {total_score}/15  " + "  ".join(f"{r['name'][:4]}={r['score']}" for r in scores))

        for r in scores:
            dim_scores[r["name"]].append(r["score"])

        per_case.append({
            "case_id": case["id"],
            "case_type": case.get("case_type"),
            "message": case["message"],
            "response_draft": response_draft,
            "scores": scores,
            "total_score": total_score,
        })

        if i < total:
            time.sleep(0.3)

    summary = {
        dim: {
            "mean": round(sum(vals) / len(vals), 2),
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }
        for dim, vals in dim_scores.items() if vals
    }

    print()
    print(f"   {'Dimension':<26} Mean   Min  Max")
    for dim in DIMENSIONS:
        if dim in summary:
            s = summary[dim]
            print(f"   {dim:<26} {s['mean']:.2f}   {s['min']}    {s['max']}")

    output = {"summary": summary, "per_case": per_case}
    with open(JUDGE_OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results written to {JUDGE_OUTPUT_PATH}")


if __name__ == "__main__":
    print("Ledgr LLM-as-judge harness")
    print("--------------------------")
    print("1. Generate calibration template (run agent on 10 cases, ready for hand-scoring)")
    print("2. Run calibration     (compare your hand scores vs judge — fill template first)")
    print("3. Run full judge eval (all 30 cases)")
    print()
    choice = input("Select an option [1/2/3]: ").strip()

    if choice == "1":
        print(f"\nGenerating calibration template …\n")
        generate_calibration_template()
    elif choice == "2":
        if not os.path.exists(CALIBRATION_PATH):
            print(f"\nCalibration file not found. Run option 1 first.")
            sys.exit(1)
        print(f"\nRunning calibration …\n")
        run_calibration()
    elif choice == "3":
        print(f"\nRunning judge eval on all {DATASET_PATH} cases …\n")
        run_judge_eval()
    else:
        print("Invalid selection.")
        sys.exit(1)
