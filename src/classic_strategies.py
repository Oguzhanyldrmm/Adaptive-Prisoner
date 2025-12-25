from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from .genetic_agent import Action, History, COOPERATE, DEFECT


@dataclass(frozen=True)
class ClassicStrategy:
    name: str
    policy: Callable[[History], Action]

    def get_action(self, history: History) -> Action:
        return self.policy(history)


def _always_cooperate(_: History) -> Action:
    return COOPERATE


def _always_defect(_: History) -> Action:
    return DEFECT


def _tit_for_tat(history: History) -> Action:
    if not history:
        return COOPERATE
    return history[-1][1]


def _grim_trigger(history: History) -> Action:
    for _, opp_action in history:
        if opp_action == DEFECT:
            return DEFECT
    return COOPERATE


def _tit_for_two_tats(history: History) -> Action:
    if len(history) < 2:
        return COOPERATE
    if history[-1][1] == DEFECT and history[-2][1] == DEFECT:
        return DEFECT
    return COOPERATE


def get_classic_strategies() -> List[ClassicStrategy]:
    return [
        ClassicStrategy(name="Always Cooperate", policy=_always_cooperate),
        ClassicStrategy(name="Always Defect", policy=_always_defect),
        ClassicStrategy(name="Tit For Tat", policy=_tit_for_tat),
        ClassicStrategy(name="Grim Trigger", policy=_grim_trigger),
        ClassicStrategy(name="Tit For Two Tats", policy=_tit_for_two_tats),
    ]
