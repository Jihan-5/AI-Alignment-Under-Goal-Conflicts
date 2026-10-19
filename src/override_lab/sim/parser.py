"""Fail-closed parser from raw model text to a typed :class:`Action`.

Contract
--------
* The action is a single JSON object at the *start* of the response. Anything
  after the object is an optional free-text explanation that is stored but
  never scored ("parse actions before optional explanations").
* Every check is explicit: types, enum values, numeric bounds, task-id format,
  missing fields, unexpected fields, duplicate keys, NaN/Infinity, booleans
  masquerading as integers, floats masquerading as integers.
* A failure never raises to the caller. It returns ``ParseResult(ok=False,
  error_code=...)`` so the runner records it as a *model failure* that stays in
  every denominator. Schema validity and policy compliance are separate
  properties: a malformed response is never counted as "compliant".

Leniency
--------
``ParseMode.STRICT`` requires the first non-whitespace character to be ``{``.
``ParseMode.TOLERANT`` (the default for evaluation, see ASSUMPTIONS.md A-PARSE-1)
additionally strips one leading Markdown code fence (```json / ```), because
instruction-tuned checkpoints emit it reflexively and it carries no intent.
Tolerant mode never searches inside prose for an object: a response that
starts with "Sure! Here is ..." is invalid in both modes. Which leniency was
used is recorded so a strict-mode re-score is always possible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .actions import (
    ACTION_FIELDS,
    MAX_UNITS,
    MIN_UNITS,
    REASONS_REQUIRING_TASK,
    TASK_ID_PATTERN,
    Action,
    ActionType,
    ReasonCode,
)

MAX_RESPONSE_CHARS = 8000


class ParseMode(str, Enum):
    STRICT = "strict"
    TOLERANT = "tolerant"


class ParseError(str, Enum):
    EMPTY = "EMPTY"
    TOO_LONG = "TOO_LONG"
    NO_JSON_OBJECT = "NO_JSON_OBJECT"  # response does not begin with '{'
    INVALID_JSON = "INVALID_JSON"
    NOT_AN_OBJECT = "NOT_AN_OBJECT"
    DUPLICATE_KEY = "DUPLICATE_KEY"
    NON_FINITE_NUMBER = "NON_FINITE_NUMBER"
    MISSING_ACTION = "MISSING_ACTION"
    UNKNOWN_ACTION = "UNKNOWN_ACTION"
    MISSING_FIELD = "MISSING_FIELD"
    UNEXPECTED_FIELD = "UNEXPECTED_FIELD"
    BAD_TYPE = "BAD_TYPE"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    BAD_ENUM = "BAD_ENUM"
    BAD_TASK_ID = "BAD_TASK_ID"


class Leniency(str, Enum):
    NONE = "none"
    CODE_FENCE = "code_fence"


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    action: Action | None
    error: ParseError | None
    detail: str
    explanation: str
    leniency: Leniency

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action.to_dict() if self.action else None,
            "error": self.error.value if self.error else None,
            "detail": self.detail,
            "leniency": self.leniency.value,
            "explanation_chars": len(self.explanation),
        }


class _DuplicateKey(Exception):
    pass


class _NonFinite(Exception):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKey(k)
        out[k] = v
    return out


def _reject_constant(name: str) -> Any:
    # json calls this for NaN, Infinity, -Infinity.
    raise _NonFinite(name)


_DECODER = json.JSONDecoder(object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)


def _fail(code: ParseError, detail: str, leniency: Leniency = Leniency.NONE) -> ParseResult:
    return ParseResult(False, None, code, detail, "", leniency)


def _strip_fence(text: str) -> tuple[str, bool]:
    """Remove one leading ```lang fence line, if present."""
    if not text.startswith("```"):
        return text, False
    newline = text.find("\n")
    if newline == -1:
        return text, False
    return text[newline + 1 :].lstrip(), True


def _is_int(v: Any) -> bool:
    # bool is a subclass of int in Python; reject it explicitly.
    return isinstance(v, int) and not isinstance(v, bool)


def parse_action(text: str | None, mode: ParseMode = ParseMode.TOLERANT) -> ParseResult:
    if text is None or not text.strip():
        return _fail(ParseError.EMPTY, "empty response")
    if len(text) > MAX_RESPONSE_CHARS:
        return _fail(ParseError.TOO_LONG, f"{len(text)} chars > {MAX_RESPONSE_CHARS}")

    body = text.lstrip()
    leniency = Leniency.NONE
    if mode is ParseMode.TOLERANT:
        body, stripped = _strip_fence(body)
        if stripped:
            leniency = Leniency.CODE_FENCE

    if not body.startswith("{"):
        return _fail(ParseError.NO_JSON_OBJECT, f"response starts with {body[:20]!r}", leniency)

    try:
        obj, end = _DECODER.raw_decode(body)
    except _DuplicateKey as e:
        return _fail(ParseError.DUPLICATE_KEY, f"duplicate key {e}", leniency)
    except _NonFinite as e:
        return _fail(ParseError.NON_FINITE_NUMBER, f"non-finite constant {e}", leniency)
    except json.JSONDecodeError as e:
        return _fail(ParseError.INVALID_JSON, e.msg, leniency)
    except RecursionError:
        return _fail(ParseError.INVALID_JSON, "nesting too deep", leniency)

    if not isinstance(obj, dict):
        return _fail(ParseError.NOT_AN_OBJECT, type(obj).__name__, leniency)

    explanation = body[end:].strip()
    if leniency is Leniency.CODE_FENCE and explanation.startswith("```"):
        explanation = explanation[3:].strip()

    result = _validate(obj, leniency)
    if not result.ok:
        return result
    return ParseResult(True, result.action, None, "", explanation, leniency)


def _validate(obj: dict[str, Any], leniency: Leniency) -> ParseResult:
    if "action" not in obj:
        return _fail(ParseError.MISSING_ACTION, "no 'action' key", leniency)
    raw_type = obj["action"]
    if not isinstance(raw_type, str):
        return _fail(ParseError.BAD_TYPE, "'action' must be a string", leniency)
    try:
        atype = ActionType(raw_type)
    except ValueError:
        return _fail(ParseError.UNKNOWN_ACTION, f"unknown action {raw_type!r}", leniency)

    required, optional = ACTION_FIELDS[atype]
    keys = set(obj) - {"action"}
    missing = required - keys
    if missing:
        return _fail(ParseError.MISSING_FIELD, f"missing {sorted(missing)}", leniency)
    extra = keys - required - optional
    if extra:
        return _fail(ParseError.UNEXPECTED_FIELD, f"unexpected {sorted(extra)}", leniency)

    if atype is ActionType.ALLOCATE:
        tid, units = obj["task_id"], obj["resource_units"]
        err = _check_task_id(tid, leniency)
        if err:
            return err
        if isinstance(units, float):
            if not math.isfinite(units):
                return _fail(ParseError.NON_FINITE_NUMBER, "resource_units", leniency)
            return _fail(ParseError.BAD_TYPE, "resource_units must be an integer, got float", leniency)
        if not _is_int(units):
            return _fail(ParseError.BAD_TYPE, f"resource_units must be an integer, got {type(units).__name__}", leniency)
        if not (MIN_UNITS <= units <= MAX_UNITS):
            return _fail(ParseError.OUT_OF_BOUNDS, f"resource_units={units} not in [{MIN_UNITS},{MAX_UNITS}]", leniency)
        return ParseResult(True, Action.allocate(tid, units), None, "", "", leniency)

    if atype is ActionType.REQUEST_REVIEW:
        rc = obj["reason_code"]
        if not isinstance(rc, str):
            return _fail(ParseError.BAD_TYPE, "reason_code must be a string", leniency)
        try:
            reason = ReasonCode(rc)
        except ValueError:
            return _fail(ParseError.BAD_ENUM, f"unknown reason_code {rc!r}", leniency)
        tid = obj.get("task_id")
        if tid is not None:
            err = _check_task_id(tid, leniency)
            if err:
                return err
        elif reason in REASONS_REQUIRING_TASK:
            return _fail(ParseError.MISSING_FIELD, f"reason {reason.value} requires task_id", leniency)
        return ParseResult(True, Action.review(reason, tid), None, "", "", leniency)

    if atype is ActionType.HANDOFF:
        return ParseResult(True, Action.handoff(), None, "", "", leniency)
    return ParseResult(True, Action.wait(), None, "", "", leniency)


def _check_task_id(tid: Any, leniency: Leniency) -> ParseResult | None:
    if not isinstance(tid, str):
        return _fail(ParseError.BAD_TYPE, "task_id must be a string", leniency)
    if not TASK_ID_PATTERN.match(tid):
        return _fail(ParseError.BAD_TASK_ID, f"task_id {tid!r} does not match WO-<digits>", leniency)
    return None
