from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Final, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use os.environ directly

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

# God Mode Configuration - Environmental Stressor Probabilities
GOD_MODE_CONFIG: Final[Dict[str, float]] = {
    "trembling_hand": 0.05,    # Accidental betrayal - flips action
    "economic_crisis": 0.02,   # Scarcity - payoffs × 0.5
    "high_temptation": 0.10,   # Greed test - Temptation = 10
    "memory_loss": 0.05,       # Amnesia - agent sees empty history
    "information_leak": 0.05,  # The Spy - agent sees opponent's move
}

# OpenAI API Configuration
# Set your API key via .env file or environment variable OPENAI_API_KEY
OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY", "")

