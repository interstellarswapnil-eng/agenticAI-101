"""
Mock tools for the Chapter D tool-use agent.

Everything here is deterministic and in-memory — no real APIs. That's deliberate:
fixtures that never change mean every eval failure is the agent's fault, not the
tool's. The fixtures below are the "world" the agent can observe; the dataset's
ground-truth labels depend on what these tools return.

Two schema variants are exported:
- TOOL_SCHEMAS       — carefully written descriptions (when to use, when NOT to,
                       argument format rules)
- TOOL_SCHEMAS_VAGUE — identical names and parameters, lazy one-line descriptions

The descriptions are the ONLY difference between the two. Comparing eval runs
across them is the chapter's experiment: tool descriptions are the new system prompt.
"""

import copy
import json

# ---------------------------------------------------------------------------
# Fixtures — the world state the tools expose
# ---------------------------------------------------------------------------

MOCK_DB = {
    "customers": {
        "maya@studiofern.com": {
            "name": "Maya Chen",
            "plan": "pro",
            "account_status": "active",
            "member_since": "2024-03-12",
        },
        "diego@pixelpine.co": {
            "name": "Diego Alarcón",
            "plan": "starter",
            "account_status": "past_due",
            "member_since": "2025-11-02",
        },
        "lena@brightform.design": {
            "name": "Lena Okafor",
            "plan": "studio",
            "account_status": "active",
            "member_since": "2022-06-30",
        },
        "sam@inkwell.art": {
            "name": "Sam Whitaker",
            "plan": "starter",
            "account_status": "active",
            "member_since": "2026-05-18",
        },
        "priya@northloop.studio": {
            "name": "Priya Raman",
            "plan": "pro",
            "account_status": "canceled",
            "member_since": "2023-09-04",
        },
        "tomas@veldtcreative.com": {
            "name": "Tomas Berg",
            "plan": "pro",
            "account_status": "active",
            "member_since": "2024-10-21",
        },
    },
    "invoices": {
        "INV-2044": {
            "status": "failed",
            "amount": 1250.00,
            "currency": "USD",
            "issued": "2026-06-28",
            "customer_email": "diego@pixelpine.co",
            "failure_reason": "card_declined",
        },
        "INV-3117": {
            "status": "paid",
            "amount": 480.00,
            "currency": "USD",
            "issued": "2026-06-15",
            "customer_email": "maya@studiofern.com",
        },
        "INV-1893": {
            "status": "pending",
            "amount": 2900.00,
            "currency": "USD",
            "issued": "2026-07-01",
            "customer_email": "lena@brightform.design",
        },
        "INV-2551": {
            "status": "failed",
            "amount": 760.00,
            "currency": "USD",
            "issued": "2026-06-30",
            "customer_email": "tomas@veldtcreative.com",
            "failure_reason": "insufficient_funds",
        },
        "INV-4020": {
            "status": "paid",
            "amount": 95.00,
            "currency": "USD",
            "issued": "2026-06-20",
            "customer_email": "sam@inkwell.art",
        },
        "INV-3305": {
            "status": "pending",
            "amount": 1540.00,
            "currency": "USD",
            "issued": "2026-07-03",
            "customer_email": "tomas@veldtcreative.com",
        },
    },
    # One major incident and one minor — everything else is healthy. A bug report
    # in "invoicing" is a P0 confirmed outage; the same report in "templates" is
    # an ordinary P2 bug. The agent can only know the difference by checking.
    "known_issues": {
        "invoicing": {
            "active_incident": True,
            "severity": "major",
            "description": "Invoice delivery emails delayed up to 4 hours for all customers",
            "started": "2026-07-05T06:10:00Z",
        },
        "payments": {"active_incident": False},
        "tax_reports": {"active_incident": False},
        "templates": {"active_incident": False},
        "client_portal": {
            "active_incident": True,
            "severity": "minor",
            "description": "Client portal pages loading slowly for some EU customers",
            "started": "2026-07-04T18:40:00Z",
        },
        "login": {"active_incident": False},
    },
}

FEATURE_AREAS = sorted(MOCK_DB["known_issues"].keys())


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
# Every tool returns a dict. Not-found returns an explicit {"error": ...} dict
# instead of raising — the model should see its own bad lookups, and the eval
# should be able to observe them in the trajectory.


def lookup_customer(email):
    """Return account details for a customer email, or an error dict."""
    customer = MOCK_DB["customers"].get(email.strip().lower())
    if customer is None:
        return {"error": f"no customer account found for {email}"}
    return dict(customer, email=email.strip().lower())


def get_invoice_status(invoice_id):
    """Return status details for an invoice ID, or an error dict."""
    invoice = MOCK_DB["invoices"].get(invoice_id.strip().upper())
    if invoice is None:
        return {"error": f"no invoice found with id {invoice_id}"}
    return dict(invoice, invoice_id=invoice_id.strip().upper())


def check_known_issues(feature_area):
    """Return the active-incident record for a product area, or an error dict."""
    issues = MOCK_DB["known_issues"].get(feature_area.strip().lower())
    if issues is None:
        return {
            "error": f"unknown feature area: {feature_area}",
            "valid_areas": FEATURE_AREAS,
        }
    return dict(issues, feature_area=feature_area.strip().lower())


def escalate_to_human(reason, priority):
    """Create a (mock) escalation ticket. Calling this at all is the action under test."""
    if priority not in ("P0", "P1", "P2", "P3"):
        return {"error": f"invalid priority: {priority} (must be P0-P3)"}
    # Deterministic ticket — the eval cares that the escalation happened and why,
    # not what the ticket number is.
    return {
        "ticket_id": "ESC-1042",
        "status": "queued_for_human",
        "reason_recorded": reason,
        "priority": priority,
    }


TOOL_REGISTRY = {
    "lookup_customer": lookup_customer,
    "get_invoice_status": get_invoice_status,
    "check_known_issues": check_known_issues,
    "escalate_to_human": escalate_to_human,
}


def execute_tool(name, arguments):
    """
    Dispatch one tool call from the model.

    Never raises: unknown tools and malformed arguments come back as error dicts,
    so the model sees the failure in-conversation and the eval sees it in the
    trajectory.
    """
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**arguments)
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}


# ---------------------------------------------------------------------------
# Schemas — precise variant
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": (
                "Look up a Ledgr customer's account by email address. Returns plan tier "
                "(starter/pro/studio), account status (active/past_due/canceled), and signup date. "
                "Use when the customer's plan, account standing, or tenure would change the triage — "
                "for example a cancellation threat or a billing-standing question. "
                "Do NOT use for invoice questions (use get_invoice_status) or for something-is-broken "
                "reports (use check_known_issues). Only call with an email address that appears "
                "verbatim in the customer's message — never guess or invent one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The customer's email address, copied exactly from their message.",
                    }
                },
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_invoice_status",
            "description": (
                "Look up a single invoice by its ID. Returns status (paid/failed/pending), amount, "
                "issue date, and a failure reason when the payment failed. "
                "Use when the customer asks about a payment or invoice AND their message contains an "
                "invoice ID (format: INV-#### ). The ID must be copied verbatim from the message — "
                "never construct, guess, or autocomplete one. If the customer mentions an invoice but "
                "gives no ID, do not call this tool; ask for the ID in your response draft instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "Invoice ID in the form INV-####, copied exactly from the customer's message.",
                    }
                },
                "required": ["invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_known_issues",
            "description": (
                "Check whether Ledgr currently has an active known incident in one product area. "
                "Returns the incident description and severity, or active_incident=false if the area "
                "is healthy. Use whenever a customer reports that something is broken or not working — "
                "an active major incident makes the report P0 and the response draft should mention it. "
                "Not for billing, account, or how-to questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_area": {
                        "type": "string",
                        "enum": FEATURE_AREAS,
                        "description": "The product area the customer's report is about.",
                    }
                },
                "required": ["feature_area"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "Create a ticket that routes this conversation to a human support engineer. "
                "Escalation interrupts a person, so it must be justified by the message or by a tool "
                "result. Use ONLY when: (a) the customer states intent to cancel or describes "
                "business-critical harm happening now, or (b) a tool result confirms a P0/P1 situation "
                "you cannot resolve in a reply (a failed payment on a past_due account, an active major "
                "incident blocking their work). Do NOT escalate how-to questions, feature requests, "
                "casual mentions of competitors or cancellation, or issues with a workaround."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "One sentence: what was confirmed and why a human is needed.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["P0", "P1", "P2", "P3"],
                        "description": "Priority of the escalation ticket.",
                    },
                },
                "required": ["reason", "priority"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Schemas — vague variant (the experiment)
# ---------------------------------------------------------------------------
# Same tools, same names, same parameters and types. The only change is the
# prose: descriptions a rushed developer would write. Built by copying the
# precise schemas and overwriting description strings, so the two variants
# cannot drift apart structurally.

_VAGUE_DESCRIPTIONS = {
    "lookup_customer": ("Gets customer info.", {"email": "The email."}),
    "get_invoice_status": ("Gets invoice details.", {"invoice_id": "The ID."}),
    "check_known_issues": ("Checks for issues.", {"feature_area": "The area."}),
    "escalate_to_human": (
        "Escalates to support.",
        {"reason": "The reason.", "priority": "The priority."},
    ),
}

TOOL_SCHEMAS_VAGUE = copy.deepcopy(TOOL_SCHEMAS)
for _schema in TOOL_SCHEMAS_VAGUE:
    _fn = _schema["function"]
    _desc, _param_descs = _VAGUE_DESCRIPTIONS[_fn["name"]]
    _fn["description"] = _desc
    for _param, _param_desc in _param_descs.items():
        _fn["parameters"]["properties"][_param]["description"] = _param_desc


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    smoke_calls = [
        ("lookup_customer", {"email": "diego@pixelpine.co"}),
        ("lookup_customer", {"email": "nobody@nowhere.com"}),
        ("get_invoice_status", {"invoice_id": "INV-2044"}),
        ("get_invoice_status", {"invoice_id": "INV-9999"}),
        ("check_known_issues", {"feature_area": "invoicing"}),
        ("check_known_issues", {"feature_area": "payments"}),
        ("check_known_issues", {"feature_area": "spaceships"}),
        ("escalate_to_human", {"reason": "confirmed failed payment on past_due account", "priority": "P1"}),
        ("escalate_to_human", {"reason": "bad priority test", "priority": "urgent"}),
        ("get_invoice_status", {"wrong_param": "INV-2044"}),
        ("no_such_tool", {}),
    ]
    for name, args in smoke_calls:
        print(f"\n{name}({json.dumps(args)})")
        print(f"  -> {json.dumps(execute_tool(name, args))}")
