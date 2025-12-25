from __future__ import annotations

from typing import Dict, List, Sequence

from config import DEFAULT_PAYOFFS, PayoffMatrix
from .classic_strategies import ClassicStrategy, get_classic_strategies
from .genetic_agent import GeneticAgent
from .simulation import play_match


def run_leaderboard(
    best_agent: GeneticAgent,
    rounds: int,
    strategies: Sequence[ClassicStrategy] | None = None,
    payoffs: PayoffMatrix = DEFAULT_PAYOFFS,
) -> List[Dict[str, float]]:
    if strategies is None:
        strategies = get_classic_strategies()

    results: List[Dict[str, float]] = []
    for strategy in strategies:
        score_agent, score_opponent = play_match(best_agent, strategy, rounds, payoffs)
        results.append(
            {
                "opponent": strategy.name,
                "agent_score": float(score_agent),
                "opponent_score": float(score_opponent),
                "agent_avg": float(score_agent) / rounds,
                "opponent_avg": float(score_opponent) / rounds,
            }
        )

    return results
