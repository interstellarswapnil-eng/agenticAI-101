import re

FORBIDDEN_PHRASES = [
    "guaranteed",
    "100%",
    "immediately",
    "right away",
    "I will personally",
    "we will refund",
    "we will fix this today",
    "definitely",
]

ACKNOWLEDGMENT_KEYWORDS = [
    "understand",
    "sorry",
    "thanks for",
    "I see",
    "I hear you",
    "apologies",
    "got it",
]

IMPERATIVE_VERBS = [
    "try", "click", "send", "check", "reply",
    "visit", "open", "run", "share", "confirm",
]


def check_length(response: str) -> dict:
    n = len(response)
    if n < 50:
        return {"name": "length", "pass": False, "reason": f"length {n} below minimum 50"}
    if n > 400:
        return {"name": "length", "pass": False, "reason": f"length {n} above maximum 400"}
    return {"name": "length", "pass": True, "reason": f"length {n} within 50-400"}


def check_json_shape(response: str) -> dict:
    if isinstance(response, str) and len(response.strip()) > 0:
        return {"name": "json_shape", "pass": True, "reason": "non-empty string"}
    return {"name": "json_shape", "pass": False, "reason": "not a non-empty string"}


def check_no_forbidden_promises(response: str) -> dict:
    lower = response.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in lower:
            return {"name": "no_forbidden_promises", "pass": False, "reason": f"matched forbidden phrase: '{phrase}'"}
    return {"name": "no_forbidden_promises", "pass": True, "reason": "no forbidden phrases found"}


def check_acknowledgment(response: str) -> dict:
    lower = response.lower()
    for keyword in ACKNOWLEDGMENT_KEYWORDS:
        if keyword.lower() in lower:
            return {"name": "acknowledgment", "pass": True, "reason": f"found acknowledgment keyword: '{keyword}'"}
    return {"name": "acknowledgment", "pass": False, "reason": "no acknowledgment keyword found"}


def check_actionability(response: str) -> dict:
    if "?" in response:
        return {"name": "actionability", "pass": True, "reason": "contains question mark"}
    for verb in IMPERATIVE_VERBS:
        if re.search(rf"\b{verb}\b", response, re.IGNORECASE):
            return {"name": "actionability", "pass": True, "reason": f"imperative verb: '{verb}'"}
    return {"name": "actionability", "pass": False, "reason": "no question mark or imperative verb found"}


def run_rubric(response: str) -> list[dict]:
    """Run all five criteria. Return list of result dicts in the order above."""
    return [
        check_length(response),
        check_json_shape(response),
        check_no_forbidden_promises(response),
        check_acknowledgment(response),
        check_actionability(response),
    ]


if __name__ == "__main__":
    samples = [
        (
            "good",
            "I'm sorry to hear you're having trouble with invoice #4471 — I understand this is "
            "time-sensitive since you need to get paid today. Could you let me know which browser "
            "you're using? In the meantime, try downloading the PDF from your invoice list and "
            "emailing it to your client directly. That should get the invoice over while we "
            "investigate the sending issue.",
        ),
        (
            "bad",
            "We will fix this today, guaranteed.",
        ),
        (
            "borderline",
            "Your account shows the Pro upgrade went through at 14:32 UTC. We will refund the "
            "difference if the plan limits aren't reflecting properly. Check your subscription "
            "settings under Account and reply here with what you see.",
        ),
    ]

    for label, response in samples:
        results = run_rubric(response)
        passed = sum(1 for r in results if r["pass"])
        print(f"\n--- {label.upper()} ({passed}/{len(results)} pass) ---")
        print(f"Input: {response!r}")
        print()
        for r in results:
            status = "PASS" if r["pass"] else "FAIL"
            print(f"  [{status}] {r['name']}: {r['reason']}")
