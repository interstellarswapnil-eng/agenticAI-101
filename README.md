# Ledgr Support Triage Agent

Classifies incoming support messages into `category` + `priority`, drafts a reply, and scores confidence. Runs on **gpt-4o-mini**.

## Setup

```bash
pip install openai python-dotenv
cp .env.example .env
# edit .env and paste your real OPENAI_API_KEY
```

## Quick test

```python
python - <<'EOF'
from agent import triage
import json
result = triage("My invoice won't send and I need to get paid today.")
print(json.dumps(result, indent=2))
EOF
```

Expected shape:
```json
{
  "category": "bug",
  "priority": "P0",
  "response_draft": "...",
  "confidence": "high"
}
```

## Run the eval

```bash
python eval.py
```

Reads `dataset.json` (30 labelled cases), calls the agent for each, prints a summary table, and writes full results to `eval_results_v0.json`.

## Categories & priorities

| Category | Meaning |
|---|---|
| `bug` | Something broken or not working |
| `billing` | Payment / subscription / refund |
| `feature_request` | Asking for new functionality |
| `how_to` | How to use an existing feature |
| `churn_risk` | Intent to cancel or strong dissatisfaction |

| Priority | SLA |
|---|---|
| P0 | Urgent — blocking work or imminent churn |
| P1 | Significant — same-day response |
| P2 | Moderate — within 24 hours |
| P3 | Low — general inquiry |

## Iteration plan

After v0 runs end-to-end, iterate the prompt in `agent.py` to improve accuracy on the ambiguous and adversarial cases in `dataset.json`. The `case_type` field in each result (`clear` / `ambiguous` / `adversarial`) helps identify where to focus.
