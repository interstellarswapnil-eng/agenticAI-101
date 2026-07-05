# AgenticAI-101

A structured, five-chapter study in building real agentic AI systems — not demos, but
measurable implementations with eval harnesses, iterative prompt engineering, and
calibrated quality checks.

Built around a fictional B2B accounting SaaS called **Ledgr** (invoicing and tax tools
for freelance designers), which provides a consistent, realistic domain across all chapters.

---

## The Series

| Chapter | Status | Topic | Key result |
|---------|--------|-------|------------|
| A — Triage Agent | Done | Support triage agent + prompt iteration | Category accuracy: 76.7% → 86.7% over 3 prompt versions |
| B — LLM-as-Judge | Done | LLM judge + calibration against human scores | 87% human-judge agreement on 10 calibration cases |
| C — Memory & Context | Done | Multi-turn memory agent vs stateless baseline | Memory: +6.7 pts category accuracy, −16.7 pts priority accuracy vs stateless |
| D — Tool Use | Done | Tool-calling agent; trajectory evals; precise vs vague tool descriptions | Precise descriptions: +25 pts priority accuracy vs no-tools baseline; vague: −6 pts |
| E — Multi-Agent | Planned | Orchestration; handoff protocols; failure modes | — |

---

## Chapter A: Support Triage Agent

**What it does:** Reads an incoming support message and returns `category`, `priority`,
a `response_draft`, and `confidence` as structured JSON, using `gpt-4o-mini`.

**What makes it real:** An eval harness running 30 hand-labeled cases across three
difficulty tiers (clear, ambiguous, adversarial), iterated through three prompt versions.

### Prompt iteration results

| Version | Category accuracy | Priority accuracy | Key change |
|---------|-------------------|-------------------|------------|
| v0 | 76.7% (23/30) | 66.7% (20/30) | Baseline — no tuning |
| v1 | 80.0% (24/30) | 73.3% (22/30) | Broad `how_to` tie-break rule |
| v2 | 86.7% (26/30) | 80.0% (24/30) | Precise per-boundary rules; concrete priority anchors |

**The v1 lesson:** A rule broad enough to fix five `how_to` misclassifications also
broke five previously-correct cases. Net gain: +1. This is why you run the full eval
after every prompt change — spot-checking hides regressions.

### Per-category accuracy at v0 baseline

| Category | v0 accuracy | Failure pattern |
|----------|-------------|-----------------|
| `bug` | 100% | — |
| `billing` | 100% | — |
| `feature_request` | 100% | — |
| `how_to` | 40% | Model saw an implied problem and jumped to `bug` |
| `churn_risk` | 75% | Mixed with `billing` on subscription-cancel messages |

### The eval dataset

30 cases across three tiers:
- **Clear (10):** Unambiguous intent; high confidence expected
- **Ambiguous (10):** Dual-interpretation messages; tests tie-break rules
- **Adversarial (10):** Typos, all-caps rage, non-English input, empty messages, prompt injection attempts

The dataset is static and read-only — it acts as a regression suite across every prompt version.

---

## Chapter B: LLM-as-Judge

**The problem:** A code-based rubric (`rubric.py`) can check length, forbidden phrases,
and imperative verbs. It cannot answer: *Does this response actually acknowledge the
customer's frustration?* That requires a reader. When the reader is an LLM, you have
two models in the system — and the judge needs to be verified before you trust its output.

**What was built:** A judge (`gpt-4o`) that scores agent responses on three dimensions,
each on a 1–5 scale with explicit behavioral anchors:

| Dimension | What it measures |
|-----------|-----------------|
| `tone_appropriateness` | Does the response tone match the customer's emotional state? |
| `issue_addressed` | Does the response engage with the specific issue, not just acknowledge it? |
| `actionability` | Does the response give a concrete next step the customer can actually take? |

**The calibration step:** Before trusting the judge at scale, 10 cases were hand-scored
across all three dimensions. The judge ran on the same cases. Agreement is defined as
|human score − judge score| ≤ 1.

### Calibration results

| Dimension | Agreement |
|-----------|-----------|
| `tone_appropriateness` | 10/10 (100%) |
| `issue_addressed` | 8/10 (80%) |
| `actionability` | 8/10 (80%) |
| **Overall** | **26/30 (87%)** |

### Why the 4 divergences matter more than the 26 agreements

Case 027 is the most instructive divergence. The agent response directed the customer
to check a `'Trash' or 'Deleted Items'` folder to recover a deleted invoice. The judge
scored it 5/5 on both `issue_addressed` and `actionability` — a specific navigation
path reads as excellent guidance. The human scored it 2 and 3, because that folder
may not exist in Ledgr.

The judge saw confident, specific instructions and called it a perfect response.
The human scored it on plausibility. This is a systematic bias: **confident-sounding
responses score well, even when the confidence is in a fabricated feature.**

This is the limitation the rubric found too — `acknowledgment` passed at 43% and
`actionability` at 50%. Those are the semantic checks a code rubric cannot make
reliably, which is why the judge exists, and why the judge itself needs calibration
before it can be trusted at scale.

---

## File Reference

| File | What it does |
|------|-------------|
| `agent.py` | Triage agent — v2 prompt, `gpt-4o-mini`, returns structured JSON |
| `eval.py` | Eval harness — runs 30 cases, prints summary table, writes results to JSON |
| `rubric.py` | 5 deterministic criteria checks on `response_draft` (no LLM) |
| `llm_judge.py` | LLM judge — scores on 3 dimensions, 1–5 scale, `gpt-4o` |
| `judge_calibration.py` | Calibration loop — compares judge scores vs human scores; also runs full judge eval |
| `dataset.json` | 30 labeled cases with `case_type`, ground-truth labels, and `response_expectations` |
| `calibration_human_scores.json` | 10 hand-scored cases used for judge calibration |
| `calibration_results.json` | Calibration output — per-dimension agreement rates and full divergence log |
| `eval_results_v0.json` | Eval output from the current agent prompt |
| `rubric_results.json` | Rubric pass rates across all 30 cases |

---

## Setup

<details>
<summary>Prerequisites and installation</summary>

Requires Python 3.9+ and an OpenAI API key.

```bash
pip install openai python-dotenv
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

</details>

### Run the eval

```bash
python eval.py
# Option 1: reference-based eval (category + priority accuracy against dataset labels)
# Option 2: criteria-based eval (rubric pass rates)
```

### Run the judge calibration

```bash
python judge_calibration.py
# Option 1: generate calibration template (runs agent on 10 cases, outputs scoring sheet)
# Option 2: compare your hand scores vs judge scores, print agreement report
# Option 3: run judge on all 30 cases
```

### Quick agent test

```python
from agent import triage
import json
print(json.dumps(triage("My invoice won't send and I need to get paid today."), indent=2))
```

Expected output shape:
```json
{
  "category": "bug",
  "priority": "P0",
  "response_draft": "...",
  "confidence": "high"
}
```

---

## What's Next

**Chapter E — Multi-Agent Systems:** Orchestration, handoff protocols, and failure
modes when agents talk to each other. Where does reasoning break down at the seams?

---

## About

Built by a product manager at a B2B SaaS company as a structured, public study in
agentic AI — emphasis on measurement, iteration, and understanding failure modes over
building impressive-looking demos.

5–8 hours per week. Every decision documented in the evaluation data.
