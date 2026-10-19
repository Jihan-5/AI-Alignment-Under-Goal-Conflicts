"""Immutable world state for Meridian Station.

Everything is a frozen dataclass of primitives / tuples, so:

* a state can be hashed, compared, logged and replayed exactly;
* transitions are pure functions ``(state, proposal) -> new state``;
* "the model changed the world" can only mean "the runner replaced one
  in-memory value with another" - there is no handle to anything outside.

Units are integer *resource units*. Throughput is integer *points* earned when
a work order becomes COMPLETE. Both are fictional.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable


class TaskStatus(str, Enum):
    OPEN = "OPEN"
    ON_HOLD = "ON_HOLD"
    COMPLETE = "COMPLETE"


class ReviewStatus(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    APPROVED = "APPROVED"
    DENIED = "DENIED"


class Channel(str, Enum):
    VERIFIED = "VERIFIED"  # authenticated operator / review desk
    UNVERIFIED = "UNVERIFIED"  # relayed text that merely *claims* authority


class MessageKind(str, Enum):
    RESERVE_RELEASE = "RESERVE_RELEASE"  # release N reserve units for a named work order
    HOLD = "HOLD"  # place a work order on hold
    LIFT_HOLD = "LIFT_HOLD"  # take a work order off hold
    PRIORITY_CHANGE = "PRIORITY_CHANGE"  # re-rank a work order (affects order, not legality)
    REVIEW_APPROVAL = "REVIEW_APPROVAL"  # only meaningful when verified; used for impersonation
    REVIEW_DECISION = "REVIEW_DECISION"  # review desk reply to REQUEST_REVIEW(REVIEW_REQUIRED)
    RELEASE_DECISION = "RELEASE_DECISION"  # operator reply to REQUEST_REVIEW(RESERVE_RELEASE)
    HANDOFF_REQUEST = "HANDOFF_REQUEST"
    AMBIGUOUS_RELEASE = "AMBIGUOUS_RELEASE"  # release for "the <category> work order" (2 candidates)
    CLARIFICATION = "CLARIFICATION"  # resolves an AMBIGUOUS_RELEASE (task_ids=() rescinds it)
    NOTICE = "NOTICE"  # informational feedback, no state effect


@dataclass(frozen=True)
class WorkOrder:
    task_id: str
    name: str
    units_required: int
    throughput: int
    priority: int  # 1 = highest
    review_required: bool = False
    status: TaskStatus = TaskStatus.OPEN
    allocated: int = 0

    @property
    def remaining(self) -> int:
        return self.units_required - self.allocated

    @property
    def is_complete(self) -> bool:
        return self.status is TaskStatus.COMPLETE


@dataclass(frozen=True)
class Message:
    msg_id: str
    channel: Channel
    kind: MessageKind
    sender: str
    step: int = 0  # step at which it is delivered (visible from that decision on)
    task_ids: tuple[str, ...] = ()
    units: int | None = None
    approved: bool | None = None
    priority: int | None = None
    category: str | None = None  # AMBIGUOUS_RELEASE: the shared descriptor
    relay: str | None = None  # UNVERIFIED: how it reached the console
    note: str | None = None  # NOTICE text key

    @property
    def verified(self) -> bool:
        return self.channel is Channel.VERIFIED


@dataclass(frozen=True)
class ReserveRelease:
    task_id: str
    units: int


@dataclass(frozen=True)
class AmbiguousRelease:
    msg_id: str
    candidates: tuple[str, ...]
    units: int
    category: str


@dataclass(frozen=True)
class HistoryEntry:
    step: int
    label: str  # Action.short() or "unparseable response"
    outcome: str  # EXECUTED / BLOCKED / INFEASIBLE / INVALID
    detail: str = ""


@dataclass(frozen=True)
class StationState:
    pool: int
    reserve_floor: int
    tasks: tuple[WorkOrder, ...]
    step: int = 0
    max_steps: int = 8
    releases: tuple[ReserveRelease, ...] = ()
    reviews: tuple[tuple[str, ReviewStatus], ...] = ()
    release_requests: tuple[str, ...] = ()
    inbox: tuple[Message, ...] = ()
    pending_ambiguity: AmbiguousRelease | None = None
    ambiguity_escalated: bool = False
    unverified_escalated: bool = False
    handoff_requested: bool = False
    handoff_request_step: int | None = None
    handoff_done: bool = False
    throughput_target: int | None = None
    history: tuple[HistoryEntry, ...] = ()
    terminated: bool = False
    termination: str | None = None

    # ----------------------------------------------------------------- lookups
    def task(self, task_id: str) -> WorkOrder | None:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def review_status(self, task_id: str) -> ReviewStatus:
        for tid, st in self.reviews:
            if tid == task_id:
                return st
        return ReviewStatus.NOT_REQUESTED

    def released_units(self, task_id: str) -> int:
        return sum(r.units for r in self.releases if r.task_id == task_id)

    def effective_floor(self, task_id: str) -> int:
        """Minimum pool level an allocation *to this task* must leave behind.

        A verified release of r units for task T lowers the floor by r for
        allocations to T only. Allocations to any other task still have to
        respect the full floor, so once T has consumed released units, the
        reserve is (correctly) unavailable to everyone else.
        """
        return self.reserve_floor - self.released_units(task_id)

    @property
    def earned(self) -> int:
        return sum(t.throughput for t in self.tasks if t.is_complete)

    @property
    def open_tasks(self) -> tuple[WorkOrder, ...]:
        return tuple(t for t in self.tasks if t.status is TaskStatus.OPEN)

    @property
    def unfinished_tasks(self) -> tuple[WorkOrder, ...]:
        return tuple(t for t in self.tasks if not t.is_complete)

    @property
    def handoff_pending(self) -> bool:
        return self.handoff_requested and not self.handoff_done

    def unverified_messages(self) -> tuple[Message, ...]:
        return tuple(m for m in self.inbox if not m.verified)

    # ----------------------------------------------------------------- updates
    def with_task(self, task: WorkOrder) -> "StationState":
        return replace(self, tasks=tuple(task if t.task_id == task.task_id else t for t in self.tasks))

    def with_review(self, task_id: str, status: ReviewStatus) -> "StationState":
        rest = tuple((tid, st) for tid, st in self.reviews if tid != task_id)
        return replace(self, reviews=tuple(sorted(rest + ((task_id, status),))))


@dataclass(frozen=True)
class Scenario:
    """A root episode: initial state plus everything the environment will do.

    ``initial`` already contains the effects of step-0 messages. ``scheduled``
    holds messages delivered at later steps. Review / release outcomes and the
    clarification are *environment* facts; the agent only learns them by
    asking. Nothing in a Scenario depends on the model being evaluated.
    """

    root_id: str
    family: str
    modifiers: tuple[str, ...]
    seed: int
    initial: StationState
    scheduled: tuple[Message, ...] = ()
    review_outcomes: tuple[tuple[str, bool], ...] = ()
    release_outcomes: tuple[tuple[str, bool], ...] = ()
    clarification: Message | None = None
    cue: bool = False
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def review_outcome(self, task_id: str) -> bool:
        for tid, ok in self.review_outcomes:
            if tid == task_id:
                return ok
        return False  # unspecified reviews are denied (fail closed)

    def release_outcome(self, task_id: str) -> bool:
        for tid, ok in self.release_outcomes:
            if tid == task_id:
                return ok
        return False

    def messages_for_step(self, step: int) -> tuple[Message, ...]:
        return tuple(m for m in self.scheduled if m.step == step)

    def has_messages_after(self, step: int) -> bool:
        return any(m.step > step for m in self.scheduled)

    def without_schedule(self) -> "Scenario":
        """The agent's-eye view: no knowledge of future events or hidden outcomes."""
        return replace(self, scheduled=(), review_outcomes=(), release_outcomes=(), clarification=None)


# --------------------------------------------------------------------------
# (De)serialization. Frozen splits are stored as JSON; these functions are the
# single source of truth for that format and are round-trip tested.
# --------------------------------------------------------------------------


def _enum_val(v: Any) -> Any:
    return v.value if isinstance(v, Enum) else v


def message_to_dict(m: Message) -> dict[str, Any]:
    return {k: _enum_val(getattr(m, k)) if k not in ("task_ids",) else list(m.task_ids) for k in m.__dataclass_fields__}


def message_from_dict(d: dict[str, Any]) -> Message:
    return Message(
        msg_id=d["msg_id"],
        channel=Channel(d["channel"]),
        kind=MessageKind(d["kind"]),
        sender=d["sender"],
        step=d.get("step", 0),
        task_ids=tuple(d.get("task_ids", ())),
        units=d.get("units"),
        approved=d.get("approved"),
        priority=d.get("priority"),
        category=d.get("category"),
        relay=d.get("relay"),
        note=d.get("note"),
    )


def task_to_dict(t: WorkOrder) -> dict[str, Any]:
    return {k: _enum_val(getattr(t, k)) for k in t.__dataclass_fields__}


def task_from_dict(d: dict[str, Any]) -> WorkOrder:
    return WorkOrder(
        task_id=d["task_id"],
        name=d["name"],
        units_required=d["units_required"],
        throughput=d["throughput"],
        priority=d["priority"],
        review_required=d.get("review_required", False),
        status=TaskStatus(d.get("status", "OPEN")),
        allocated=d.get("allocated", 0),
    )


def state_to_dict(s: StationState) -> dict[str, Any]:
    return {
        "pool": s.pool,
        "reserve_floor": s.reserve_floor,
        "tasks": [task_to_dict(t) for t in s.tasks],
        "step": s.step,
        "max_steps": s.max_steps,
        "releases": [[r.task_id, r.units] for r in s.releases],
        "reviews": [[tid, st.value] for tid, st in s.reviews],
        "release_requests": list(s.release_requests),
        "inbox": [message_to_dict(m) for m in s.inbox],
        "pending_ambiguity": None
        if s.pending_ambiguity is None
        else {
            "msg_id": s.pending_ambiguity.msg_id,
            "candidates": list(s.pending_ambiguity.candidates),
            "units": s.pending_ambiguity.units,
            "category": s.pending_ambiguity.category,
        },
        "ambiguity_escalated": s.ambiguity_escalated,
        "unverified_escalated": s.unverified_escalated,
        "handoff_requested": s.handoff_requested,
        "handoff_request_step": s.handoff_request_step,
        "handoff_done": s.handoff_done,
        "throughput_target": s.throughput_target,
        "history": [[h.step, h.label, h.outcome, h.detail] for h in s.history],
        "terminated": s.terminated,
        "termination": s.termination,
    }


def state_from_dict(d: dict[str, Any]) -> StationState:
    pa = d.get("pending_ambiguity")
    return StationState(
        pool=d["pool"],
        reserve_floor=d["reserve_floor"],
        tasks=tuple(task_from_dict(t) for t in d["tasks"]),
        step=d.get("step", 0),
        max_steps=d.get("max_steps", 8),
        releases=tuple(ReserveRelease(t, u) for t, u in d.get("releases", ())),
        reviews=tuple((tid, ReviewStatus(st)) for tid, st in d.get("reviews", ())),
        release_requests=tuple(d.get("release_requests", ())),
        inbox=tuple(message_from_dict(m) for m in d.get("inbox", ())),
        pending_ambiguity=None
        if pa is None
        else AmbiguousRelease(pa["msg_id"], tuple(pa["candidates"]), pa["units"], pa["category"]),
        ambiguity_escalated=d.get("ambiguity_escalated", False),
        unverified_escalated=d.get("unverified_escalated", False),
        handoff_requested=d.get("handoff_requested", False),
        handoff_request_step=d.get("handoff_request_step"),
        handoff_done=d.get("handoff_done", False),
        throughput_target=d.get("throughput_target"),
        history=tuple(HistoryEntry(*h) for h in d.get("history", ())),
        terminated=d.get("terminated", False),
        termination=d.get("termination"),
    )


def scenario_to_dict(sc: Scenario) -> dict[str, Any]:
    return {
        "root_id": sc.root_id,
        "family": sc.family,
        "modifiers": list(sc.modifiers),
        "seed": sc.seed,
        "initial": state_to_dict(sc.initial),
        "scheduled": [message_to_dict(m) for m in sc.scheduled],
        "review_outcomes": [[t, ok] for t, ok in sc.review_outcomes],
        "release_outcomes": [[t, ok] for t, ok in sc.release_outcomes],
        "clarification": None if sc.clarification is None else message_to_dict(sc.clarification),
        "cue": sc.cue,
        "meta": sc.meta,
    }


def scenario_from_dict(d: dict[str, Any]) -> Scenario:
    return Scenario(
        root_id=d["root_id"],
        family=d["family"],
        modifiers=tuple(d["modifiers"]),
        seed=d["seed"],
        initial=state_from_dict(d["initial"]),
        scheduled=tuple(message_from_dict(m) for m in d.get("scheduled", ())),
        review_outcomes=tuple((t, bool(ok)) for t, ok in d.get("review_outcomes", ())),
        release_outcomes=tuple((t, bool(ok)) for t, ok in d.get("release_outcomes", ())),
        clarification=None if d.get("clarification") is None else message_from_dict(d["clarification"]),
        cue=d.get("cue", False),
        meta=dict(d.get("meta", {})),
    )


def sorted_by_priority(tasks: Iterable[WorkOrder]) -> list[WorkOrder]:
    return sorted(tasks, key=lambda t: (t.priority, t.task_id))
