import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JUDGE_MODEL = "gpt-4o"  # intentionally different model family from agent (gpt-4o-mini)
JUDGE_TEMPERATURE = 0.0

DIMENSIONS = ["tone_appropriateness", "issue_addressed", "actionability"]

JUDGE_SYSTEM_PROMPT = """You are an impartial quality evaluator for Ledgr, invoicing and tax software for freelance designers. You score customer support responses on three dimensions using the anchors below.

TONE APPROPRIATENESS — does the response tone match the customer's emotional state?
1 = Completely wrong tone: cheerful or promotional for a frustrated/urgent customer, or cold/formal for a casual question
2 = Noticeably off: overly apologetic for a simple how_to, or matter-of-fact for a churn-risk escalation
3 = Neutral and acceptable but generic — does not reflect the customer's specific emotional state
4 = Largely matches the customer's emotional state with only minor misalignment
5 = Precisely mirrors the customer's emotional state: empathetic for distress, efficient for simple questions, appropriately urgent for time-sensitive issues

ISSUE ADDRESSED — does the response engage with the specific issue raised, not a generic acknowledgment?
1 = Ignores the specific issue entirely — a generic "we'll look into it" or an off-topic reply
2 = Acknowledges the customer but addresses a related but different issue
3 = Addresses the issue at a surface level — correct topic, but misses key specifics the customer mentioned
4 = Addresses the main issue clearly with minor gaps (e.g., correct feature named but one detail overlooked)
5 = Directly addresses the exact issue — names the specific feature, error type, or billing scenario the customer described. Merely citing the identifier (invoice number, feature name) without demonstrating real understanding of the problem does NOT score 5.

ACTIONABILITY — does the response give a concrete next step the customer can take?
1 = No next step — the response ends without any direction for the customer
2 = Completely passive — no commitment that the issue will be addressed ("thank you for your message" with nothing more)
3 = Commits to investigate or follow up but gives no action or ask for the customer ("we'll look into this and get back to you")
4 = Asks the customer for specific information, or offers a step they can try, but requires some inference
5 = A concrete, immediately actionable next step: specific menu path, exact field to check, or precise information the customer should send. Only scores 5 when the step clearly exists in the product — if the response invents navigation paths or assumes features the customer's message did not confirm, cap at 3.

Return ONLY valid JSON. No text outside the JSON block."""

_USER_PROMPT = """Customer message:
{message}

Agent response:
{response}

Score the agent response on all three dimensions. Return JSON only:
{{
  "tone_appropriateness": {{"score": <1-5>, "reason": "<one sentence>"}},
  "issue_addressed":      {{"score": <1-5>, "reason": "<one sentence>"}},
  "actionability":        {{"score": <1-5>, "reason": "<one sentence>"}}
}}"""


def run_judge(message: str, response: str) -> list[dict]:
    """Score a response on 3 dimensions. Returns list of score dicts.

    Each dict: {"name": str, "score": int (1-5), "reason": str}
    Mirrors run_rubric() output shape but uses score instead of pass.
    """
    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT.format(message=message, response=response)},
        ],
    )

    raw = completion.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps its output
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)

    return [
        {"name": dim, "score": parsed[dim]["score"], "reason": parsed[dim]["reason"]}
        for dim in DIMENSIONS
    ]


if __name__ == "__main__":
    samples = [
        {
            "label": "good — urgent P0, specific workaround offered",
            "message": (
                "My invoice #4471 won't send to my client. I click send and nothing happens. "
                "Been trying for 2 hours and I need to get paid today."
            ),
            "response": (
                "I understand this is urgent — needing to get paid today adds real pressure. "
                "To help diagnose the issue, could you let me know which browser you're using? "
                "In the meantime, try downloading the invoice as a PDF from your invoice list "
                "and emailing it directly to your client. That should get payment moving while "
                "we investigate the send failure on our end."
            ),
        },
        {
            "label": "bad — generic, no next step",
            "message": (
                "My invoice #4471 won't send to my client. I click send and nothing happens. "
                "Been trying for 2 hours and I need to get paid today."
            ),
            "response": "Thank you for reaching out. We are looking into this and will get back to you soon.",
        },
    ]

    for s in samples:
        scores = run_judge(s["message"], s["response"])
        total = sum(r["score"] for r in scores)
        print(f"\n--- {s['label'].upper()} (total {total}/15) ---")
        for r in scores:
            print(f"  [{r['score']}/5] {r['name']}: {r['reason']}")
