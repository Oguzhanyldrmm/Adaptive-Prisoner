from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, List, Sequence, Tuple

from config import GENOTYPE_LENGTH

Action = int
History = Sequence[Tuple[Action, Action]]

COOPERATE: Action = 1
DEFECT: Action = 0


def random_genotype(rng: random.Random, length: int = GENOTYPE_LENGTH) -> List[int]:
    return [rng.randint(0, 1) for _ in range(length)]


def _outcome_index(self_action: Action, opp_action: Action) -> int:
    if self_action == COOPERATE and opp_action == COOPERATE:
        return 0
    if self_action == COOPERATE and opp_action == DEFECT:
        return 1
    if self_action == DEFECT and opp_action == COOPERATE:
        return 2
    return 3


@dataclass
class GeneticAgent:
    id: int
    genotype: List[int] = field(default_factory=list)
    fitness: float = 0.0

    def __post_init__(self) -> None:
        if len(self.genotype) != GENOTYPE_LENGTH:
            raise ValueError(
                f"Genotype length must be {GENOTYPE_LENGTH}, got {len(self.genotype)}"
            )

    def reset_fitness(self) -> None:
        self.fitness = 0.0

    def get_action(self, history: History) -> Action:
        if not history:
            return self.genotype[0]
        if len(history) == 1:
            return self.genotype[1]

        prev2 = history[-2]
        prev1 = history[-1]
        idx = _outcome_index(prev2[0], prev2[1]) * 4 + _outcome_index(
            prev1[0], prev1[1]
        )
        return self.genotype[2 + idx]

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "fitness": self.fitness,
            "genotype": list(self.genotype),
        }
