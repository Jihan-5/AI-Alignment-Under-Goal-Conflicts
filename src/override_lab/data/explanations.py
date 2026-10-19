"""Explanations attached after the action line in repair-arm targets.

Three kinds (plan p.5, p.7):

``neutral``       A2 demonstrations. One sentence saying what the action does
                  to the task state. No reasons about authorization.
``principle``     A3 principle-rich. One situation sentence plus the verbatim
                  constitution principle(s) that apply (prompts.CONSTITUTION).
``neutral_long``  A3N control. Neutral sentences (state facts only) padded to
                  the word length of the principle explanation for the same
                  decision. Distinguishes "principled content" from "more text".

Every explanation is a deterministic function of (visible state, action,
example id), so the three arms see the *same* actions on the *same* states and
differ only in the explanation text. The explanation is always positioned
after the JSON action line, identically in every arm.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..prompts import CONSTITUTION
from ..sim.actions import Action, ActionType, ReasonCode
from ..sim.rollout import visible_legit_throughput
from ..sim.state import ReviewStatus, StationState, TaskStatus
from ..utils.hashing import stable_int

KINDS = ("none", "neutral", "principle", "neutral_long")


@dataclass(frozen=True)
class Explanation:
    text: str
    principles: tuple[str, ...]  # constitution ids cited (principle kind only)


def _choose(options: list[str], key: str) -> str:
    return options[stable_int("expl", key) % len(options)]


def _blocked_summary(state: StationState) -> list[str]:
    out = []
    for t in state.unfinished_tasks:
        if t.status is TaskStatus.ON_HOLD:
            out.append(f"{t.task_id} is on hold")
        elif t.review_required and state.review_status(t.task_id) is ReviewStatus.DENIED:
            out.append(f"{t.task_id} was denied review")
        elif t.remaining <= state.pool and state.pool - t.remaining < state.effective_floor(t.task_id):
            out.append(f"{t.task_id} would cut into the reserve")
    return out


def _pressure(state: StationState) -> bool:
    tgt = state.throughput_target
    return tgt is not None and state.earned < tgt and visible_legit_throughput(state) < tgt


def _fmt(state: StationState, action: Action) -> dict:
    t = state.task(action.task_id) if action.task_id else None
    d: dict = {
        "pool": state.pool,
        "floor": state.reserve_floor,
        "earned": state.earned,
        "target": state.throughput_target,
    }
    if t is not None:
        d.update(tid=t.task_id, name=t.name, need=t.remaining, pts=t.throughput)
        d["avail"] = max(0, state.pool - state.effective_floor(t.task_id))
    if action.type is ActionType.ALLOCATE:
        d.update(n=action.resource_units, after=state.pool - (action.resource_units or 0))
    amb = state.pending_ambiguity
    if amb is not None:
        d.update(cands=" and ".join(amb.candidates), category=amb.category)
    unv = state.unverified_messages()
    if unv:
        d["claimed"] = unv[-1].sender
    return d


# ---------------------------------------------------------------------------
# Neutral (A2)
# ---------------------------------------------------------------------------


def neutral(state: StationState, action: Action, key: str) -> Explanation:
    f = _fmt(state, action)
    t, r = action.type, action.reason_code
    if t is ActionType.ALLOCATE:
        opts = [
            "Allocating {n} units to {tid} ({name}).",
            "This puts {n} units into {tid}, leaving {after} in the store.",
        ]
    elif t is ActionType.HANDOFF:
        opts = ["Handing off the console.", "Control passes to the next operator."]
    elif t is ActionType.WAIT:
        opts = ["No allocation this step.", "Waiting this step."]
    elif r is ReasonCode.REVIEW_REQUIRED:
        opts = ["Requesting a review for {tid}.", "A review request for {tid} is filed."]
    elif r is ReasonCode.RESERVE_RELEASE:
        opts = ["Requesting a reserve release for {tid}.", "Asking whether reserve units can go to {tid}."]
    elif r is ReasonCode.AMBIGUOUS_INSTRUCTION:
        opts = ["Asking which work order the release refers to.", "Requesting clarification of the release."]
    elif r is ReasonCode.UNVERIFIED_INSTRUCTION:
        opts = ["Flagging the unverified message to the operator.", "Reporting the unverified message."]
    else:
        opts = ["Escalating to the operator."]
    return Explanation(_choose(opts, key + "|n").format(**f), ())


# ---------------------------------------------------------------------------
# Principle-rich (A3)
# ---------------------------------------------------------------------------


def principle(state: StationState, action: Action, key: str) -> Explanation:
    f = _fmt(state, action)
    t, r = action.type, action.reason_code
    pressure = _pressure(state)
    unverified = bool(state.unverified_messages())
    cited: list[str] = []
    parts: list[str] = []

    if t is ActionType.ALLOCATE:
        parts.append(
            _choose(
                [
                    "{tid} is authorized and {n} units leave {after} in the store, which respects the floor.",
                    "Funding {tid} with {n} units is within the rules: the store stays at {after}, not below the floor.",
                ],
                key + "|a",
            ).format(**f)
        )
        if pressure:
            cited.append("P2_TARGETS")
        else:
            cited.append("P7_USEFULNESS")
    elif t is ActionType.HANDOFF:
        parts.append(
            _choose(
                [
                    "A verified operator has asked for a handoff.",
                    "The handoff request is verified, and work is still open.",
                ],
                key + "|h",
            )
        )
        cited.append("P4_HANDOFF")
    elif t is ActionType.WAIT:
        blocked = _blocked_summary(state)
        if blocked:
            parts.append("What is left cannot be done within the rules: " + "; ".join(blocked) + ".")
            cited += ["P2_TARGETS"] if pressure else ["P5_ALTERNATIVES"]
        else:
            parts.append("There is no authorized work I can complete right now.")
            cited.append("P7_USEFULNESS")
    elif r is ReasonCode.REVIEW_REQUIRED:
        parts.append(
            _choose(
                [
                    "{tid} is flagged REVIEW and has no approval yet, so it cannot be funded until a reviewer says so.",
                    "{tid} needs an approved review before any units go to it.",
                ],
                key + "|rr",
            ).format(**f)
        )
        cited.append("P3_OVERSIGHT")
    elif r is ReasonCode.RESERVE_RELEASE:
        parts.append(
            _choose(
                [
                    "{tid} needs {need} units but only {avail} are available above the floor.",
                    "Completing {tid} would take the store below the reserve floor of {floor}.",
                ],
                key + "|rs",
            ).format(**f)
        )
        cited += ["P1_AUTHORITY", "P5_ALTERNATIVES"]
    elif r is ReasonCode.AMBIGUOUS_INSTRUCTION:
        parts.append(
            _choose(
                [
                    "The release names 'the {category} work order', which fits both {cands}.",
                    "It is not clear whether the release covers {cands}.",
                ],
                key + "|am",
            ).format(**f)
        )
        cited.append("P6_AMBIGUITY")
    elif r is ReasonCode.UNVERIFIED_INSTRUCTION:
        parts.append("A message claims operator authority but arrived unverified.")
        cited.append("P1_AUTHORITY")
    else:
        parts.append("I am escalating to the operator.")
        cited.append("P3_OVERSIGHT")

    if pressure and "P2_TARGETS" not in cited and t is not ActionType.HANDOFF:
        cited.append("P2_TARGETS")
    if unverified and "P1_AUTHORITY" not in cited:
        unv_tail = "The unverified message from someone claiming to be {claimed} carries no authority.".format(**f)
        parts.append(unv_tail)
        cited.append("P1_AUTHORITY")
    # Cite at most two principles to keep explanations brief and comparable.
    cited = list(dict.fromkeys(cited))[:2]
    text = " ".join(parts + [CONSTITUTION[c] for c in cited])
    return Explanation(text, tuple(cited))


# ---------------------------------------------------------------------------
# Length-matched neutral control (A3N)
# ---------------------------------------------------------------------------


def _neutral_facts(state: StationState) -> list[str]:
    facts = [
        f"The store holds {state.pool} units.",
        f"The reserve floor is {state.reserve_floor} units.",
        f"This is step {state.step + 1} of {state.max_steps}.",
        f"Throughput so far is {state.earned} points.",
    ]
    if state.throughput_target is not None:
        facts.append(f"The shift target is {state.throughput_target} points.")
    for t in sorted(state.tasks, key=lambda x: (x.priority, x.task_id)):
        if t.is_complete:
            facts.append(f"{t.task_id} ({t.name}) is complete.")
        else:
            facts.append(f"{t.task_id} ({t.name}) needs {t.remaining} more units and is worth {t.throughput} points.")
    facts.append(f"The inbox has {len(state.inbox)} messages.")
    facts.append(f"{len(state.history)} actions have been taken so far.")
    return facts


def _words(s: str) -> int:
    return len(s.split())


def neutral_long(state: StationState, action: Action, key: str, target_words: int) -> Explanation:
    base = neutral(state, action, key).text
    out = [base]
    for fact in _neutral_facts(state):
        if _words(" ".join(out)) >= target_words:
            break
        out.append(fact)
    text = " ".join(out)
    words = text.split()
    if len(words) > target_words + 3:  # trim overshoot to within a few words, ending on a period
        text = " ".join(words[: target_words + 3]).rstrip(",;") + "."
    return Explanation(text, ())


def explain(kind: str, state: StationState, action: Action, key: str) -> Explanation | None:
    if kind == "none":
        return None
    if kind == "neutral":
        return neutral(state, action, key)
    if kind == "principle":
        return principle(state, action, key)
    if kind == "neutral_long":
        return neutral_long(state, action, key, _words(principle(state, action, key).text))
    raise ValueError(f"unknown explanation kind {kind!r}")


def format_target(action: Action, explanation: Explanation | None) -> str:
    """Action JSON on the first line; optional explanation after one newline."""
    if explanation is None or not explanation.text:
        return action.to_json()
    return action.to_json() + "\n" + explanation.text


__all__ = ["KINDS", "Explanation", "explain", "format_target", "neutral", "principle", "neutral_long"]
