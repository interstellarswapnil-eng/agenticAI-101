"""
Chapter D: the tool-use triage agent.

Same model, same base system prompt, same output JSON as agent.py — the only new
variable is a set of tools the agent may call before answering. What changes is
not what the agent says but what it DOES first, so triage_with_tools returns two
things: the familiar result dict, and a trajectory of every tool call it made.
The trajectory is what Chapter D's eval judges; the words alone no longer tell
the whole story.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from agent import SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, execute_tool

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MAX_TOOL_CALLS = 5  # loop guard — deliberately simple, same spirit as MAX_HISTORY_TURNS

SYSTEM_PROMPT_V3 = SYSTEM_PROMPT + """

TOOL USE:
You have tools that look up real account, invoice, and incident data.
- Before triaging a billing question or a something-is-broken report, check the relevant tool — the result may change the priority (a confirmed failed payment or an active incident is more urgent than the message alone suggests).
- Never invent an email address or invoice ID. Only pass values that appear verbatim in the customer's message. If the customer mentions an invoice or account but gives no ID or email, do not guess — ask for it in your response_draft.
- Escalate to a human only when the escalation tool's criteria are met. An escalation interrupts a person; it is an action with a cost, not a formality.
- When a tool result is relevant to the customer, reference it in your response_draft.
- Once you are done with tools (or need none), your final message must be the JSON object only — same format as above, no other text."""


def triage_with_tools(message, tool_schemas=TOOL_SCHEMAS):
    """
    Triage a support message, letting the model call tools before it answers.

    Args:
        message: The customer message (str).
        tool_schemas: Which schema variant to expose — TOOL_SCHEMAS (precise) or
            TOOL_SCHEMAS_VAGUE. The descriptions are the experiment variable;
            everything else in this function is identical between runs.

    Returns:
        (result_dict, trajectory) — result has the same shape as agent.triage().
        trajectory is an ordered list of {"tool", "arguments", "result"} dicts,
        one per executed tool call. An empty list means the model answered
        without touching a tool.

    The loop runs until the model stops requesting tools. After MAX_TOOL_CALLS
    executed calls, tool_choice="none" forces a final answer — a stuck agent
    should produce a measurable bad answer, not an infinite loop.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V3},
        {"role": "user", "content": message},
    ]
    trajectory = []

    while True:
        force_answer = len(trajectory) >= MAX_TOOL_CALLS
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=512,
            messages=messages,
            tools=tool_schemas,
            tool_choice="none" if force_answer else "auto",
        )
        reply = response.choices[0].message

        if not reply.tool_calls:
            break

        messages.append(reply)
        for tool_call in reply.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            result = execute_tool(tool_call.function.name, arguments)
            trajectory.append(
                {
                    "tool": tool_call.function.name,
                    "arguments": arguments,
                    "result": result,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    text = reply.content.strip()
    return json.loads(text), trajectory


if __name__ == "__main__":
    # Three smoke messages, one per behavior worth seeing before the real eval:
    # a lookup the agent SHOULD make, a question needing no tools, and a
    # missing-ID trap where any lookup argument would be fabricated.
    demo_messages = [
        (
            "tool_required -- expect get_invoice_status(INV-2044) in trajectory",
            "My payment for INV-2044 was declined and I have a client project due tomorrow. What is going on?",
        ),
        (
            "no_tool_needed -- expect empty trajectory",
            "How do I add my studio logo to my invoice templates?",
        ),
        (
            "missing_info -- no ID given; watch whether it fabricates one",
            "One of my invoices failed to send to a client last week and I never got notified. Which one was it?",
        ),
    ]

    for note, message in demo_messages:
        print(f"\n=== {note}")
        print(f"customer: {message}")
        result, trajectory = triage_with_tools(message)
        if trajectory:
            for step in trajectory:
                print(f"  tool call: {step['tool']}({json.dumps(step['arguments'])})")
                print(f"     result: {json.dumps(step['result'])}")
        else:
            print("  tool calls: none")
        print(f"  triage: {json.dumps(result, indent=2)}")
