import json
import time

from agent import triage
from agent_v2 import triage_with_memory

DATASET_PATH = "dataset_multiturn.json"
OUTPUT_PATH = "eval_memory_results.json"
SCENARIO_TYPES = ["memory_required", "irrelevant_context", "consistency_test", "context_exhaustion"]

_STOP_WORDS = {
    "to", "the", "a", "an", "of", "in", "or", "and", "for", "with",
    "from", "your", "you", "that", "this", "is", "was", "it", "be",
    "by", "at", "as", "on", "my", "i", "me", "we", "our",
    "their", "they", "its", "are", "have", "has", "had", "do", "did",
    "not", "but", "if", "so", "no", "up", "out", "about",
}


def check_context_awareness(response_draft, must_reference, must_not_repeat):
    """
    Deterministic check for whether response_draft uses prior context appropriately.

    Returns {"name": "context_awareness", "pass": bool or None, "reason": str}
    pass=None means not applicable (turn 1, no prior context to check).

    must_not_repeat: exact phrase substring check — fail if phrase appears in response.
    must_reference:  keyword check — fail if fewer than 2 content words from the hint
                     appear as substrings in the response. Conservative lower bound by
                     design (consistent with rubric.py's heuristic philosophy).
    """
    if must_reference is None and must_not_repeat is None:
        return {"name": "context_awareness", "pass": None, "reason": "not applicable for this turn"}

    response_lower = response_draft.lower()

    if must_not_repeat is not None:
        if must_not_repeat.lower() in response_lower:
            return {
                "name": "context_awareness",
                "pass": False,
                "reason": f"response repeats a suggestion the customer already tried (found '{must_not_repeat}')",
            }

    if must_reference is not None:
        ref_words = [
            w.lower().strip(".,!?;:'\"")
            for w in must_reference.split()
            if w.lower().strip(".,!?;:'\"") not in _STOP_WORDS
            and w.lower().strip(".,!?;:'\"")
        ]
        matches = sum(1 for w in ref_words if w in response_lower)
        if matches < 2:
            return {
                "name": "context_awareness",
                "pass": False,
                "reason": (
                    f"response does not reference prior context '{must_reference}' "
                    f"({matches}/{len(ref_words)} keywords matched, need ≥2)"
                ),
            }

    return {"name": "context_awareness", "pass": True, "reason": "response correctly incorporates prior context"}


def run_memory_eval(dataset):
    results = []
    for scenario in dataset:
        sid = scenario["scenario_id"]
        stype = scenario["scenario_type"]
        print(f"\n  {sid} ({stype})")
        session = []
        turn_results = []

        for i, turn in enumerate(scenario["turns"]):
            try:
                result, session = triage_with_memory(turn["message"], session)
                cat_correct = result.get("category") == turn["label"]["category"]
                pri_correct = result.get("priority") == turn["label"]["priority"]
                ctx = check_context_awareness(
                    result.get("response_draft", ""),
                    turn["label"].get("response_must_reference"),
                    turn["label"].get("response_must_not_repeat"),
                )
            except Exception as e:
                result = {"error": str(e)}
                cat_correct = False
                pri_correct = False
                ctx = {"name": "context_awareness", "pass": False, "reason": f"agent error: {e}"}

            ctx_label = "n/a" if ctx["pass"] is None else ("PASS" if ctx["pass"] else "FAIL")
            print(
                f"    turn {turn['turn']}: "
                f"cat={'✓' if cat_correct else '✗'} pri={'✓' if pri_correct else '✗'}  "
                f"memory={result.get('category', '?')}/{result.get('priority', '?')}  "
                f"ctx={ctx_label}",
                flush=True,
            )

            turn_results.append({
                "turn": turn["turn"],
                "message": turn["message"],
                "label": turn["label"],
                "prediction": result,
                "category_correct": cat_correct,
                "priority_correct": pri_correct,
                "context_awareness": ctx,
            })

            if i < len(scenario["turns"]) - 1:
                time.sleep(0.3)

        results.append({
            "scenario_id": sid,
            "scenario_type": stype,
            "failure_mode": scenario["failure_mode"],
            "turns": turn_results,
        })
        time.sleep(1.0)

    return results


def run_baseline_eval(dataset):
    results = []
    for scenario in dataset:
        sid = scenario["scenario_id"]
        turn_results = []

        for i, turn in enumerate(scenario["turns"]):
            try:
                result = triage(turn["message"])
                cat_correct = result.get("category") == turn["label"]["category"]
                pri_correct = result.get("priority") == turn["label"]["priority"]
            except Exception as e:
                result = {"error": str(e)}
                cat_correct = False
                pri_correct = False

            turn_results.append({
                "turn": turn["turn"],
                "category_correct": cat_correct,
                "priority_correct": pri_correct,
                "prediction": result,
            })

            if i < len(scenario["turns"]) - 1:
                time.sleep(0.3)

        results.append({"scenario_id": sid, "turns": turn_results})
        time.sleep(1.0)

    return results


def compute_stats(memory_results, baseline_results):
    mem_turns = [t for s in memory_results for t in s["turns"]]
    base_turns = [t for s in baseline_results for t in s["turns"]]
    total = len(mem_turns)

    mem_cat = sum(t["category_correct"] for t in mem_turns) / total
    mem_pri = sum(t["priority_correct"] for t in mem_turns) / total
    base_cat = sum(t["category_correct"] for t in base_turns) / total
    base_pri = sum(t["priority_correct"] for t in base_turns) / total

    applicable = [t for t in mem_turns if t["context_awareness"]["pass"] is not None]
    ctx_pass_rate = (
        sum(1 for t in applicable if t["context_awareness"]["pass"]) / len(applicable)
        if applicable else 0
    )

    baseline_by_id = {s["scenario_id"]: s["turns"] for s in baseline_results}

    per_type = {}
    for stype in SCENARIO_TYPES:
        mem_s = [s for s in memory_results if s["scenario_type"] == stype]
        if not mem_s:
            continue

        m_turns = [t for s in mem_s for t in s["turns"]]
        b_turns = []
        for s in mem_s:
            b_turns.extend(baseline_by_id.get(s["scenario_id"], []))

        m_cat = sum(t["category_correct"] for t in m_turns) / len(m_turns)
        b_cat = sum(t["category_correct"] for t in b_turns) / len(b_turns) if b_turns else 0
        applicable_type = [t for t in m_turns if t["context_awareness"]["pass"] is not None]
        ctx_type = (
            sum(1 for t in applicable_type if t["context_awareness"]["pass"]) / len(applicable_type)
            if applicable_type else None
        )

        per_type[stype] = {
            "memory_cat_accuracy": round(m_cat, 3),
            "baseline_cat_accuracy": round(b_cat, 3),
            "context_awareness_pass_rate": round(ctx_type, 3) if ctx_type is not None else None,
        }

    return {
        "total_scenarios": len(memory_results),
        "total_turns": total,
        "memory_agent": {
            "category_accuracy": round(mem_cat, 3),
            "priority_accuracy": round(mem_pri, 3),
            "context_awareness_pass_rate": round(ctx_pass_rate, 3),
            "context_awareness_applicable_turns": len(applicable),
        },
        "baseline_agent": {
            "category_accuracy": round(base_cat, 3),
            "priority_accuracy": round(base_pri, 3),
        },
        "regression": {
            "category_accuracy_delta": round(mem_cat - base_cat, 3),
            "priority_accuracy_delta": round(mem_pri - base_pri, 3),
            "interpretation": "positive = memory agent better; negative = memory agent regression",
        },
        "per_scenario_type": per_type,
    }


def print_summary(summary):
    m = summary["memory_agent"]
    b = summary["baseline_agent"]
    r = summary["regression"]
    delta_cat = r["category_accuracy_delta"]
    delta_pri = r["priority_accuracy_delta"]

    print()
    print("=" * 58)
    print("MEMORY EVAL RESULTS")
    print("=" * 58)
    print(
        f"Memory agent    category: {m['category_accuracy']:.0%}  "
        f"priority: {m['priority_accuracy']:.0%}  "
        f"ctx_awareness: {m['context_awareness_pass_rate']:.0%}"
    )
    print(
        f"Baseline agent  category: {b['category_accuracy']:.0%}  "
        f"priority: {b['priority_accuracy']:.0%}"
    )
    print(
        f"Regression      category: {delta_cat:+.0%}  priority: {delta_pri:+.0%}  "
        f"(positive = memory helps)"
    )
    print()
    print(f"{'Scenario type':<25} {'Mem cat':>8} {'Base cat':>9} {'Ctx aware':>10}")
    print("-" * 55)
    for stype in SCENARIO_TYPES:
        s = summary["per_scenario_type"].get(stype)
        if s is None:
            continue
        ctx = f"{s['context_awareness_pass_rate']:.0%}" if s["context_awareness_pass_rate"] is not None else "n/a"
        print(
            f"{stype:<25} {s['memory_cat_accuracy']:>7.0%} "
            f"{s['baseline_cat_accuracy']:>9.0%} {ctx:>10}"
        )
    print()
    print(f"Full results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    print("Ledgr memory eval harness")
    print("-------------------------")

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    print(f"\nRunning memory eval ({len(dataset)} scenarios, {len(dataset) * 3} turns) …")
    memory_results = run_memory_eval(dataset)

    print(f"\nRunning baseline eval (stateless agent, {len(dataset) * 3} turns) …")
    baseline_results = run_baseline_eval(dataset)

    summary = compute_stats(memory_results, baseline_results)

    baseline_by_id = {s["scenario_id"]: s["turns"] for s in baseline_results}
    scenarios_output = []
    for scenario in memory_results:
        sid = scenario["scenario_id"]
        base_turns = baseline_by_id.get(sid, [])
        merged_turns = []
        for i, t in enumerate(scenario["turns"]):
            base_t = base_turns[i] if i < len(base_turns) else {}
            merged_turns.append({
                **t,
                "baseline_prediction": base_t.get("prediction", {}),
                "baseline_category_correct": base_t.get("category_correct"),
                "baseline_priority_correct": base_t.get("priority_correct"),
            })
        scenarios_output.append({
            "scenario_id": scenario["scenario_id"],
            "scenario_type": scenario["scenario_type"],
            "failure_mode": scenario["failure_mode"],
            "turns": merged_turns,
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "scenarios": scenarios_output}, f, indent=2)

    print_summary(summary)
