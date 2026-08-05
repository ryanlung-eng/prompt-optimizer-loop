"""
State-transition self-simulation: an appended builder instruction, run as its
own benchmark arm.

The motivating observation is that the "consequence" blockers found in trace
review are not several different bugs — they are ONE shape asked three ways.
Emails never marked read, a webhook that hangs when the approver denies, a
sheet-log skipped on the deny path: none is about a node fact. Each is about
*paths and the state left behind*, i.e. "what happens on the route I wasn't
picturing while I built the happy one."

That means the fix does not have to be an enumerated list of warnings (an
infinite tail, one entry per bug anyone happens to hit). It can be a small
FIXED set of questions applied to every path, which generates the whole tail —
including cases nobody has written down yet.

Why an appended block and not a rewrite of the base prompt: this is a
hypothesis, and it is benchmarked as its own arm against the identical prompt
without it, so any movement is attributable to this text and nothing else.
Same isolation discipline as query_rewriter.py and execution_checker.py.

Known limitation, stated plainly: the platform requires the builder to emit
raw workflow JSON and nothing else, so the simulation happens internally and
we cannot verify from the output that it actually ran. A wrong or skipped
trace is invisible; only its downstream effect on blocker counts is
measurable. If this arm wins, making the trace inspectable is the natural
follow-up.

Evidence that this SHAPE of instruction works, versus a generic exhortation:
the earlier "trace your workflow's execution" line moved nothing across three
runs, while the requirement-coverage sweep — enumerate the asks, match each to
a node, report the unmatched — lifted completeness immediately. The difference
is not the topic, it is that one produces an enumerable artifact and the other
is a mood. These four questions are written to be enumerable the same way.
"""

STATE_SIMULATION_INSTRUCTION = """
CRITICAL — before finalizing, simulate your workflow's state transitions. Do
not just re-read it; walk it. Enumerate every DISTINCT path an execution can
take, not only the happy one — that includes each branch of every IF/Switch,
the deny side of every approval gate, and the error output of every node that
has one. For EACH path, answer these four questions explicitly:

1. RE-ENTRY — after this path finishes, is the state my own trigger selects
   on still in the same condition that selected it? If the trigger fires on
   unread mail, a status of "new", an unlabeled row, a missing field, or an
   absent reply, then something on this path must CHANGE that state (mark as
   read, set the status, add the label, write the field). If nothing does,
   the same item is picked up again on the next poll, forever. Equally: if
   this path writes something my own trigger watches, does that write
   re-trigger me? If so, an identity or state guard must prevent it.

2. OBLIGATION — did this path discharge what the entry point promised? A
   webhook with responseMode "responseNode" owes the caller exactly one
   response on EVERY path, including the denied and errored ones — a path
   that ends without one leaves the caller hanging indefinitely. A chat
   trigger owes a reply. A sub-workflow owes its declared return.

3. UNCONDITIONAL WORK — for each thing the user said should ALWAYS happen
   (log it, record it, notify), is that action actually on every path, or did
   I place it downstream of a condition the user never attached to it?
   Logging that sits after an approval gate does not happen when approval is
   denied; logging chained after a send does not happen when the send is
   skipped. If the user said "log every one", the log cannot be conditional
   on something else succeeding.

4. REFERENCES ON THIS PATH — does every $('Node Name') reference resolve on
   THIS specific path? A node that did not execute on the route being walked
   has no data to read, and optional chaining does not save you: referencing
   a node that never ran is an error, not undefined.

If any answer exposes a gap, fix the workflow before returning it. This is
about the paths you did not have in mind while building — the happy path is
the one you already got right.
"""
