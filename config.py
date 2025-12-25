from __future__ import annotations

from dataclasses import dataclass
from typing import Final

DEFAULT_POP_SIZE: Final[int] = 50
DEFAULT_GENERATIONS: Final[int] = 100
DEFAULT_MUTATION_RATE: Final[float] = 0.02
DEFAULT_ROUNDS_PER_MATCH: Final[int] = 150
DEFAULT_BENCHMARK_INTERVAL: Final[int] = 10
DEFAULT_ELITE_FRACTION: Final[float] = 0.1

MEMORY_DEPTH: Final[int] = 2
GENOTYPE_LENGTH: Final[int] = 18


@dataclass(frozen=True)
class PayoffMatrix:
    temptation: int
    reward: int
    punishment: int
    sucker: int


DEFAULT_PAYOFFS: Final[PayoffMatrix] = PayoffMatrix(
    temptation=5,
    reward=3,
    punishment=1,
    sucker=0,
)
