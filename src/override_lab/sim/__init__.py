"""Meridian Station: the typed, sandboxed simulator.

Host-sandbox invariant (tested by tests/test_safety.py): modules in this
package import only the standard library and each other. They never import
os/subprocess/socket/importlib/requests, never call eval/exec/compile/open,
and never touch the filesystem. The model's text reaches this package only
through ``parser.parse_action`` and leaves it only as frozen dataclasses.
"""

from .actions import Action, ActionType, ReasonCode
from .guard import GuardConfig, GuardKind
from .parser import ParseMode, ParseResult, parse_action
from .planner import reference_action
from .policy import ALL_VIOLATIONS, DeferKind, Verdict, ViolationType, check_feasibility, judge
from .rollout import make_biased_policy, planner_policy, rollout, unconstrained_policy
from .state import (
    Channel,
    Message,
    MessageKind,
    ReviewStatus,
    Scenario,
    StationState,
    TaskStatus,
    WorkOrder,
    scenario_from_dict,
    scenario_to_dict,
)
from .transition import Outcome, StepResult, Termination, deliver, step

__all__ = [
    "Action",
    "ActionType",
    "ReasonCode",
    "GuardConfig",
    "GuardKind",
    "ParseMode",
    "ParseResult",
    "parse_action",
    "reference_action",
    "ALL_VIOLATIONS",
    "DeferKind",
    "Verdict",
    "ViolationType",
    "check_feasibility",
    "judge",
    "make_biased_policy",
    "planner_policy",
    "rollout",
    "unconstrained_policy",
    "Channel",
    "Message",
    "MessageKind",
    "ReviewStatus",
    "Scenario",
    "StationState",
    "TaskStatus",
    "WorkOrder",
    "scenario_from_dict",
    "scenario_to_dict",
    "Outcome",
    "StepResult",
    "Termination",
    "deliver",
    "step",
]
