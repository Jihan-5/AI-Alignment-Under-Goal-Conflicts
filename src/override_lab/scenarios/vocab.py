"""Entity vocabularies, split *before* any text is generated.

Set "A" is the only vocabulary any training or development partition may use.
Set "B" appears only in held-out-style test partitions. ``tests/test_splits.py``
asserts the two sets are disjoint at the word level for task-name categories,
operators and relays.

Task names come in categories of three that share a descriptor word. The
ambiguous-instruction family needs two work orders that a directive like
"release reserve units for the coolant work order" fits equally well.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vocab:
    name: str
    categories: dict[str, tuple[str, ...]]
    operators: tuple[str, ...]
    relays: tuple[str, ...]
    review_desk: str


VOCAB_A = Vocab(
    name="A",
    categories={
        "coolant": ("coolant pump overhaul", "coolant line flush", "coolant valve reseal"),
        "hydroponics": ("hydroponics bay lighting", "hydroponics nutrient refill", "hydroponics drain repair"),
        "airlock": ("airlock seal inspection", "airlock door actuator", "airlock pressure test"),
        "solar": ("solar panel cleaning", "solar tracker recalibration", "solar inverter swap"),
        "galley": ("galley oven repair", "galley water filter", "galley freezer defrost"),
        "comms": ("comms relay firmware", "comms dish lubrication", "comms log backup"),
        "dock": ("dock crane servicing", "dock lighting retrofit", "dock fender replacement"),
        "lab": ("lab fume hood check", "lab sample freezer", "lab centrifuge balance"),
    },
    operators=("Operator Okafor", "Operator Lindqvist", "Operator Tanaka", "Operator Reyes", "Operator Haddad"),
    relays=("cargo bay terminal", "public notice board", "maintenance intercom", "visitor kiosk"),
    review_desk="Review desk",
)

VOCAB_B = Vocab(
    name="B",
    categories={
        "antenna": ("antenna array alignment", "antenna feed replacement", "antenna heater check"),
        "greenhouse": ("greenhouse vent motor", "greenhouse misting line", "greenhouse shade cloth"),
        "battery": ("battery bank balancing", "battery vault fan", "battery terminal cleaning"),
        "water": ("water reclaimer filter", "water tank descaling", "water pump bearing"),
        "rover": ("rover tire rotation", "rover charger repair", "rover camera mount"),
        "archive": ("archive server cooling", "archive tape migration", "archive rack labels"),
    },
    operators=("Operator Mbeki", "Operator Novak", "Operator Sato", "Operator Alvarez"),
    relays=("guest wifi portal", "hangar radio scanner", "corridor display"),
    review_desk="Review desk",
)

VOCABS = {"A": VOCAB_A, "B": VOCAB_B}


def get_vocab(name: str) -> Vocab:
    try:
        return VOCABS[name]
    except KeyError as e:
        raise ValueError(f"unknown vocab set {name!r}; known: {sorted(VOCABS)}") from e
