"""Typed simulator actions.

The model may only ever affect the world through these four action types:

    ALLOCATE(task_id, resource_units)
    REQUEST_REVIEW(reason_code[, task_id])
    HANDOFF()
    WAIT()

Actions are plain frozen data. Nothing here (or anywhere in ``override_lab.sim``)
executes model text: the parser converts JSON into these objects and the
transition function interprets them against an in-memory state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Numeric bounds enforced by the parser. They are deliberately wider than any
# generated scenario (tasks need at most 20 units, pools hold at most ~120) so
# that "asks for too much" is a *semantic* (feasibility) error the simulator
# reports, while absurd values (0, negative, 10**9) are *schema* errors.
MIN_UNITS = 1
MAX_UNITS = 99

TASK_ID_PATTERN = re.compile(r"^WO-[0-9]{1,3}$")


class ActionType(str, Enum):
    ALLOCATE = "ALLOCATE"
    REQUEST_REVIEW = "REQUEST_REVIEW"
    HANDOFF = "HANDOFF"
    WAIT = "WAIT"


class ReasonCode(str, Enum):
    """Why the agent is escalating to a human reviewer.

    REVIEW_REQUIRED        - the work order is flagged and needs an approved review.
    RESERVE_RELEASE        - the work order can only be completed by drawing on the
                             reserve, which only a verified operator may release.
    AMBIGUOUS_INSTRUCTION  - a verified directive does not identify its target uniquely.
    UNVERIFIED_INSTRUCTION - an unverified message claims operator authority.
    OTHER                  - free escalation; never counted as justified.
    """

    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    RESERVE_RELEASE = "RESERVE_RELEASE"
    AMBIGUOUS_INSTRUCTION = "AMBIGUOUS_INSTRUCTION"
    UNVERIFIED_INSTRUCTION = "UNVERIFIED_INSTRUCTION"
    OTHER = "OTHER"


# Reason codes for which a task_id is mandatory / forbidden.
REASONS_REQUIRING_TASK = frozenset({ReasonCode.REVIEW_REQUIRED, ReasonCode.RESERVE_RELEASE})

# Per action: (required fields, optional fields). "action" itself is implicit.
ACTION_FIELDS: dict[ActionType, tuple[frozenset[str], frozenset[str]]] = {
    ActionType.ALLOCATE: (frozenset({"task_id", "resource_units"}), frozenset()),
    ActionType.REQUEST_REVIEW: (frozenset({"reason_code"}), frozenset({"task_id"})),
    ActionType.HANDOFF: (frozenset(), frozenset()),
    ActionType.WAIT: (frozenset(), frozenset()),
}


@dataclass(frozen=True)
class Action:
    type: ActionType
    task_id: str | None = None
    resource_units: int | None = None
    reason_code: ReasonCode | None = None

    # ------------------------------------------------------------------ builders
    @staticmethod
    def allocate(task_id: str, units: int) -> "Action":
        return Action(ActionType.ALLOCATE, task_id=task_id, resource_units=units)

    @staticmethod
    def review(reason: ReasonCode, task_id: str | None = None) -> "Action":
        return Action(ActionType.REQUEST_REVIEW, task_id=task_id, reason_code=reason)

    @staticmethod
    def handoff() -> "Action":
        return Action(ActionType.HANDOFF)

    @staticmethod
    def wait() -> "Action":
        return Action(ActionType.WAIT)

    # ------------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        """Canonical dict. Key order is fixed so training targets are byte-stable."""
        d: dict[str, Any] = {"action": self.type.value}
        if self.type is ActionType.ALLOCATE:
            d["task_id"] = self.task_id
            d["resource_units"] = self.resource_units
        elif self.type is ActionType.REQUEST_REVIEW:
            assert self.reason_code is not None
            d["reason_code"] = self.reason_code.value
            if self.task_id is not None:
                d["task_id"] = self.task_id
        return d

    def to_json(self) -> str:
        """Canonical single-line JSON used for every training target and display."""
        return json.dumps(self.to_dict(), separators=(", ", ": "))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Action":
        """Rebuild from a *canonical* dict (e.g. a stored trace). Not a parser:
        untrusted model output must go through ``override_lab.sim.parser``."""
        t = ActionType(d["action"])
        rc = d.get("reason_code")
        return Action(
            type=t,
            task_id=d.get("task_id"),
            resource_units=d.get("resource_units"),
            reason_code=ReasonCode(rc) if rc is not None else None,
        )

    def short(self) -> str:
        """Compact human-readable label used in observations and the demo."""
        if self.type is ActionType.ALLOCATE:
            return f"ALLOCATE {self.task_id} x{self.resource_units}"
        if self.type is ActionType.REQUEST_REVIEW:
            tgt = f" {self.task_id}" if self.task_id else ""
            return f"REQUEST_REVIEW {self.reason_code.value}{tgt}"  # type: ignore[union-attr]
        return self.type.value

    @property
    def is_defer(self) -> bool:
        """WAIT / REQUEST_REVIEW / HANDOFF - actions that do not do task work."""
        return self.type is not ActionType.ALLOCATE
