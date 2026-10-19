"""Parser: fail closed on everything malformed (plan p.8 "Safety checks")."""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from override_lab.sim.actions import Action, ActionType, ReasonCode
from override_lab.sim.parser import Leniency, ParseError, ParseMode, parse_action

OK = [
    ('{"action": "ALLOCATE", "task_id": "WO-12", "resource_units": 5}', Action.allocate("WO-12", 5)),
    ('{"action": "REQUEST_REVIEW", "reason_code": "REVIEW_REQUIRED", "task_id": "WO-3"}', Action.review(ReasonCode.REVIEW_REQUIRED, "WO-3")),
    ('{"action": "REQUEST_REVIEW", "reason_code": "AMBIGUOUS_INSTRUCTION"}', Action.review(ReasonCode.AMBIGUOUS_INSTRUCTION)),
    ('{"action": "HANDOFF"}', Action.handoff()),
    ('{"action": "WAIT"}', Action.wait()),
    ('  \n{"action": "WAIT"}\nNothing left to do.', Action.wait()),
]


@pytest.mark.parametrize("text,expected", OK)
def test_valid(text, expected):
    r = parse_action(text)
    assert r.ok and r.action == expected and r.error is None


def test_explanation_is_separated_and_unscored():
    r = parse_action('{"action": "WAIT"}\nBecause the reserve binds.')
    assert r.ok and r.explanation == "Because the reserve binds."


BAD = [
    ("", ParseError.EMPTY),
    ("   ", ParseError.EMPTY),
    ("Sure! Here is my action: {\"action\": \"WAIT\"}", ParseError.NO_JSON_OBJECT),
    ('{"action": "WAIT"', ParseError.INVALID_JSON),
    ("[1, 2]", ParseError.NO_JSON_OBJECT),
    ('{"task_id": "WO-1"}', ParseError.MISSING_ACTION),
    ('{"action": "DELETE_STATION"}', ParseError.UNKNOWN_ACTION),
    ('{"action": "allocate", "task_id": "WO-1", "resource_units": 1}', ParseError.UNKNOWN_ACTION),
    ('{"action": 3}', ParseError.BAD_TYPE),
    ('{"action": "ALLOCATE", "task_id": "WO-1"}', ParseError.MISSING_FIELD),
    ('{"action": "ALLOCATE", "task_id": "WO-1", "resource_units": 1, "force": true}', ParseError.UNEXPECTED_FIELD),
    ('{"action": "WAIT", "reason": "x"}', ParseError.UNEXPECTED_FIELD),
    ('{"action": "ALLOCATE", "task_id": "WO-1", "resource_units": 0}', ParseError.OUT_OF_BOUNDS),
    ('{"action": "ALLOCATE", "task_id": "WO-1", "resource_units": -4}', ParseError.OUT_OF_BOUNDS),
    ('{"action": "ALLOCATE", "task_id": "WO-1", "resource_units": 100}', ParseError.OUT_OF_BOUNDS),
    ('{"action": "ALLOCATE", "task_id": "WO-1", "resource_units": 1000000000000}', ParseError.OUT_OF_BOUNDS),
    ('{"action": "ALLOCATE", "task_id": "WO-1", "resource_units": 5.0}', ParseError.BAD_TYPE),
    ('{"action": "ALLOCATE", "task_id": "WO-1", "resource_units": "5"}', ParseError.BAD_TYPE),
    ('{"action": "ALLOCATE", "task_id": "WO-1", "resource_units": true}', ParseError.BAD_TYPE),
    ('{"action": "ALLOCATE", "task_id": "WO-1", "resource_units": NaN}', ParseError.NON_FINITE_NUMBER),
    ('{"action": "ALLOCATE", "task_id": "WO-1", "resource_units": Infinity}', ParseError.NON_FINITE_NUMBER),
    ('{"action": "ALLOCATE", "task_id": "wo-1", "resource_units": 1}', ParseError.BAD_TASK_ID),
    ('{"action": "ALLOCATE", "task_id": "WO-1; rm -rf /", "resource_units": 1}', ParseError.BAD_TASK_ID),
    ('{"action": "ALLOCATE", "task_id": 12, "resource_units": 1}', ParseError.BAD_TYPE),
    ('{"action": "REQUEST_REVIEW", "reason_code": "PLEASE"}', ParseError.BAD_ENUM),
    ('{"action": "REQUEST_REVIEW", "reason_code": "REVIEW_REQUIRED"}', ParseError.MISSING_FIELD),
    ('{"action": "REQUEST_REVIEW", "reason_code": "RESERVE_RELEASE"}', ParseError.MISSING_FIELD),
    ('{"action": "WAIT", "action": "HANDOFF"}', ParseError.DUPLICATE_KEY),
    ("x" * 9000, ParseError.TOO_LONG),
]


@pytest.mark.parametrize("text,code", BAD)
def test_invalid_fails_closed(text, code):
    r = parse_action(text)
    assert not r.ok and r.action is None and r.error is code


def test_code_fence_only_in_tolerant_mode():
    text = '```json\n{"action": "WAIT"}\n```'
    assert parse_action(text, ParseMode.TOLERANT).ok
    assert parse_action(text, ParseMode.TOLERANT).leniency is Leniency.CODE_FENCE
    assert parse_action(text, ParseMode.STRICT).error is ParseError.NO_JSON_OBJECT


def test_tolerant_never_searches_prose():
    assert not parse_action('I will wait. {"action": "WAIT"}').ok


def test_deep_nesting_does_not_crash():
    r = parse_action("{" + '"a":' * 1 + "[" * 50000 + "]" * 50000 + "}")
    assert not r.ok


@settings(max_examples=400, deadline=None)
@given(st.text(max_size=300))
def test_never_raises_on_arbitrary_text(s):
    r = parse_action(s)
    assert r.ok in (True, False)
    if r.ok:
        assert isinstance(r.action, Action)


@settings(max_examples=300, deadline=None)
@given(
    st.dictionaries(
        st.sampled_from(["action", "task_id", "resource_units", "reason_code", "x"]),
        st.one_of(st.integers(-10, 200), st.text(max_size=12), st.booleans(), st.none(), st.floats(allow_nan=False)),
        max_size=5,
    )
)
def test_random_objects_either_valid_or_rejected(d):
    r = parse_action(json.dumps(d))
    if r.ok:
        a = r.action
        assert a.type in ActionType
        if a.type is ActionType.ALLOCATE:
            assert isinstance(a.resource_units, int) and 1 <= a.resource_units <= 99


def test_canonical_roundtrip():
    for _, a in OK:
        assert parse_action(a.to_json()).action == a
