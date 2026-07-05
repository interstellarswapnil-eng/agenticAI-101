import json
import re
import sys
import time

from agent import triage
from agent_v3 import triage_with_tools
from tools import TOOL_SCHEMAS, TOOL_SCHEMAS_VAGUE

# The chapter's one-variable experiment, same spirit as Chapter C's MAX_HISTORY_TURNS
# change: set to "vague" and rerun. Everything else — agent, dataset, checks — is
# identical; only the tool description prose changes.
SCHEMA_VARIANT = "precise"  # "precise" or "vague"

SCHEMA_VARIANTS = {
    "precise": (TOOL_SCHEMAS, "eval_tools_results.json"),
    "vague": (TOOL_SCHEMAS_VAGUE, "eval_tools_results_vague.json"),
}

DATASET_PATH = "dataset_tools.json"
CASE_TYPES = ["tool_required", "no_tool_needed", "missing_info", "escalation_judgment"]


def _is_id_like(value):
    """Values that could only be fabricated or copied: emails and INV-style IDs."""
    v = str(value).strip()
    return "@" in v or re.match(r"(?i)^inv-", v) is not None


def check_tool_selection(trajectory, expected_tools):
    """
    Did the agent call every required tool and avoid every forbidden one?

    Returns {"name": "tool_selection", "pass": bool, "reason": str}.
    Extra tools that are neither required nor forbidden are allowed — the cost of
    over-calling is reported separately via avg_tool_calls, not punished here.
    """
    called = {step["tool"] for step in trajectory}
    missing = [t for t in expected_tools["required"] if t not in called]
    violations = [t for t in expected_tools["forbidden"] if t in called]

    if missing and violations:
        reason = f"missing required {missing}; called forbidden {violations}"
    elif missing:
        reason = f"missing required tool call(s): {missing}"
    elif violations:
        reason = f"called forbidden tool(s): {violations}"
    else:
        reason = "required tools called, no forbidden tools used"

    return {"name": "tool_selection", "pass": not missing and not violations, "reason": reason}


def check_argument_fidelity(trajectory, message, expected_args):
    """
    Were the tool arguments real or invented?

    Two rules, both deterministic:
    1. Fabrication: any ID-like argument (email, INV-####) must appear in the
       customer's message OR in the result of an earlier tool call in the same
       trajectory. Chaining is legitimate — the smoke test showed the agent
       correctly passing an email it learned from an invoice lookup — but an ID
       that appears nowhere upstream was invented.
    2. expected_args: when the dataset pins an argument for a tool and that tool
       was called, at least one call to it must use the pinned value.

    Returns {"name": "argument_fidelity", "pass": bool or None, "reason": str}.
    pass=None means not applicable (the agent made no tool calls).
    """
    if not trajectory:
        return {"name": "argument_fidelity", "pass": None, "reason": "not applicable — no tool calls made"}

    failures = []

    seen_text = message.lower()
    for step in trajectory:
        for param, value in step["arguments"].items():
            if _is_id_like(value) and str(value).strip().lower() not in seen_text:
                failures.append(
                    f"{step['tool']}({param}='{value}') — value appears in neither "
                    f"the message nor any earlier tool result (fabricated)"
                )
        seen_text += " " + json.dumps(step["result"]).lower()

    for tool, pinned in expected_args.items():
        calls = [s for s in trajectory if s["tool"] == tool]
        if not calls:
            continue  # an uncalled required tool is tool_selection's failure, not this check's
        matched = any(
            all(str(c["arguments"].get(p, "")).strip().lower() == str(v).strip().lower() for p, v in pinned.items())
            for c in calls
        )
        if not matched:
            failures.append(f"{tool} called, but no call used expected args {pinned}")

    if failures:
        return {"name": "argument_fidelity", "pass": False, "reason": "; ".join(failures)}
    return {"name": "argument_fidelity", "pass": True, "reason": "all arguments traceable to the message or prior tool results"}


def run_tool_eval(dataset, tool_schemas):
    results = []
    for case in dataset:
        try:
            prediction, trajectory = triage_with_tools(case["message"], tool_schemas=tool_schemas)
            cat_correct = prediction.get("category") == case["label"]["category"]
            pri_correct = prediction.get("priority") == case["label"]["priority"]
            selection = check_tool_selection(trajectory, case["expected_tools"])
            fidelity = check_argument_fidelity(trajectory, case["message"], case["expected_args"])
        except Exception as e:
            prediction = {"error": str(e)}
            trajectory = []
            cat_correct = False
            pri_correct = False
            selection = {"name": "tool_selection", "pass": False, "reason": f"agent error: {e}"}
            fidelity = {"name": "argument_fidelity", "pass": False, "reason": f"agent error: {e}"}

        fid_label = "n/a" if fidelity["pass"] is None else ("PASS" if fidelity["pass"] else "FAIL")
        print(
            f"  {case['id']} ({case['case_type']}): "
            f"cat={'✓' if cat_correct else '✗'} pri={'✓' if pri_correct else '✗'}  "
            f"tools={'PASS' if selection['pass'] else 'FAIL'} args={fid_label}  "
            f"calls={len(trajectory)} [{', '.join(s['tool'] for s in trajectory) or 'none'}]",
            flush=True,
        )

        results.append({
            "id": case["id"],
            "case_type": case["case_type"],
            "message": case["message"],
            "label": case["label"],
            "expected_tools": case["expected_tools"],
            "expected_args": case["expected_args"],
            "prediction": prediction,
            "trajectory": trajectory,
            "category_correct": cat_correct,
            "priority_correct": pri_correct,
            "tool_selection": selection,
            "argument_fidelity": fidelity,
            "num_tool_calls": len(trajectory),
        })
        time.sleep(0.3)

    return results


def run_baseline_eval(dataset):
    results = []
    for case in dataset:
        try:
            prediction = triage(case["message"])
            cat_correct = prediction.get("category") == case["label"]["category"]
            pri_correct = prediction.get("priority") == case["label"]["priority"]
        except Exception as e:
            prediction = {"error": str(e)}
            cat_correct = False
            pri_correct = False

        results.append({
            "id": case["id"],
            "prediction": prediction,
            "category_correct": cat_correct,
            "priority_correct": pri_correct,
        })
        time.sleep(0.3)

    return results


def compute_stats(tool_results, baseline_results):
    total = len(tool_results)

    tool_cat = sum(c["category_correct"] for c in tool_results) / total
    tool_pri = sum(c["priority_correct"] for c in tool_results) / total
    base_cat = sum(c["category_correct"] for c in baseline_results) / total
    base_pri = sum(c["priority_correct"] for c in baseline_results) / total

    selection_rate = sum(1 for c in tool_results if c["tool_selection"]["pass"]) / total
    fid_applicable = [c for c in tool_results if c["argument_fidelity"]["pass"] is not None]
    fidelity_rate = (
        sum(1 for c in fid_applicable if c["argument_fidelity"]["pass"]) / len(fid_applicable)
        if fid_applicable else 0
    )
    avg_calls = sum(c["num_tool_calls"] for c in tool_results) / total

    baseline_by_id = {c["id"]: c for c in baseline_results}

    per_type = {}
    for ctype in CASE_TYPES:
        cases = [c for c in tool_results if c["case_type"] == ctype]
        if not cases:
            continue
        base_cases = [baseline_by_id[c["id"]] for c in cases]
        n = len(cases)
        per_type[ctype] = {
            "tool_cat_accuracy": round(sum(c["category_correct"] for c in cases) / n, 3),
            "baseline_cat_accuracy": round(sum(c["category_correct"] for c in base_cases) / n, 3),
            "tool_pri_accuracy": round(sum(c["priority_correct"] for c in cases) / n, 3),
            "baseline_pri_accuracy": round(sum(c["priority_correct"] for c in base_cases) / n, 3),
            "tool_selection_pass_rate": round(sum(1 for c in cases if c["tool_selection"]["pass"]) / n, 3),
            "avg_tool_calls": round(sum(c["num_tool_calls"] for c in cases) / n, 2),
        }

    return {
        "schema_variant": SCHEMA_VARIANT,
        "total_cases": total,
        "tool_agent": {
            "category_accuracy": round(tool_cat, 3),
            "priority_accuracy": round(tool_pri, 3),
            "tool_selection_pass_rate": round(selection_rate, 3),
            "argument_fidelity_pass_rate": round(fidelity_rate, 3),
            "argument_fidelity_applicable_cases": len(fid_applicable),
            "avg_tool_calls": round(avg_calls, 2),
        },
        "baseline_agent": {
            "category_accuracy": round(base_cat, 3),
            "priority_accuracy": round(base_pri, 3),
        },
        "regression": {
            "category_accuracy_delta": round(tool_cat - base_cat, 3),
            "priority_accuracy_delta": round(tool_pri - base_pri, 3),
            "interpretation": "positive = tool agent better; negative = tool agent regression",
        },
        "per_case_type": per_type,
    }


def print_summary(summary, output_path):
    t = summary["tool_agent"]
    b = summary["baseline_agent"]
    r = summary["regression"]

    print()
    print("=" * 58)
    print(f"TOOL EVAL RESULTS (schema variant: {summary['schema_variant']})")
    print("=" * 58)
    print(
        f"Tool agent      category: {t['category_accuracy']:.0%}  "
        f"priority: {t['priority_accuracy']:.0%}  "
        f"tool_selection: {t['tool_selection_pass_rate']:.0%}  "
        f"arg_fidelity: {t['argument_fidelity_pass_rate']:.0%}"
    )
    print(
        f"Baseline agent  category: {b['category_accuracy']:.0%}  "
        f"priority: {b['priority_accuracy']:.0%}"
    )
    print(
        f"Regression      category: {r['category_accuracy_delta']:+.0%}  "
        f"priority: {r['priority_accuracy_delta']:+.0%}  "
        f"(positive = tools help)"
    )
    print(f"Avg tool calls per case: {t['avg_tool_calls']}")
    print()
    print(f"{'Case type':<21} {'Tool pri':>8} {'Base pri':>9} {'Tools':>6} {'Calls':>6}")
    print("-" * 55)
    for ctype in CASE_TYPES:
        s = summary["per_case_type"].get(ctype)
        if s is None:
            continue
        print(
            f"{ctype:<21} {s['tool_pri_accuracy']:>7.0%} "
            f"{s['baseline_pri_accuracy']:>9.0%} "
            f"{s['tool_selection_pass_rate']:>6.0%} "
            f"{s['avg_tool_calls']:>6}"
        )
    print()
    print(f"Full results written to {output_path}")


if __name__ == "__main__":
    # Windows consoles that default to cp1252 can't print ✓/✗ — degrade to '?'
    # instead of crashing mid-eval.
    sys.stdout.reconfigure(errors="replace")

    tool_schemas, output_path = SCHEMA_VARIANTS[SCHEMA_VARIANT]

    print("Ledgr tool-use eval harness")
    print("---------------------------")

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\nRunning tool eval ({len(dataset)} cases, schema variant: {SCHEMA_VARIANT}) …")
    tool_results = run_tool_eval(dataset, tool_schemas)

    print(f"\nRunning baseline eval (no-tools agent, {len(dataset)} cases) …")
    baseline_results = run_baseline_eval(dataset)

    summary = compute_stats(tool_results, baseline_results)

    baseline_by_id = {c["id"]: c for c in baseline_results}
    cases_output = []
    for case in tool_results:
        base = baseline_by_id.get(case["id"], {})
        cases_output.append({
            **case,
            "baseline_prediction": base.get("prediction", {}),
            "baseline_category_correct": base.get("category_correct"),
            "baseline_priority_correct": base.get("priority_correct"),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": cases_output}, f, indent=2)

    print_summary(summary, output_path)
