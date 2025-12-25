from __future__ import annotations

from typing import Dict, List, Protocol, Sequence, Tuple

from config import DEFAULT_PAYOFFS, PayoffMatrix
from .genetic_agent import Action, History, COOPERATE, DEFECT, GeneticAgent


class ActionAgent(Protocol):
    def get_action(self, history: History) -> Action:
        ...


def play_match(
    agent_a: ActionAgent,
    agent_b: ActionAgent,
    rounds: int,
    payoffs: PayoffMatrix = DEFAULT_PAYOFFS,
) -> Tuple[int, int]:
    history_a: List[Tuple[Action, Action]] = []
    history_b: List[Tuple[Action, Action]] = []
    score_a = 0
    score_b = 0

    temptation = payoffs.temptation
    reward = payoffs.reward
    punishment = payoffs.punishment
    sucker = payoffs.sucker

    for _ in range(rounds):
        action_a = agent_a.get_action(history_a)
        action_b = agent_b.get_action(history_b)

        history_a.append((action_a, action_b))
        history_b.append((action_b, action_a))

        if action_a == COOPERATE and action_b == COOPERATE:
            score_a += reward
            score_b += reward
        elif action_a == COOPERATE and action_b == DEFECT:
            score_a += sucker
            score_b += temptation
        elif action_a == DEFECT and action_b == COOPERATE:
            score_a += temptation
            score_b += sucker
        else:
            score_a += punishment
            score_b += punishment

    return score_a, score_b


def run_internal_tournament(
    population: Sequence[GeneticAgent],
    rounds: int,
    payoffs: PayoffMatrix = DEFAULT_PAYOFFS,
) -> Dict[str, float]:
    if not population:
        return {"avg_fitness": 0.0, "max_fitness": 0.0, "min_fitness": 0.0}

    for agent in population:
        agent.reset_fitness()

    pop_list = list(population)
    pop_size = len(pop_list)

    for i in range(pop_size):
        agent_i = pop_list[i]
        for j in range(i + 1, pop_size):
            agent_j = pop_list[j]
            score_i, score_j = play_match(agent_i, agent_j, rounds, payoffs)
            agent_i.fitness += score_i
            agent_j.fitness += score_j

    total_fitness = sum(agent.fitness for agent in pop_list)
    max_fitness = max(agent.fitness for agent in pop_list)
    min_fitness = min(agent.fitness for agent in pop_list)
    avg_fitness = total_fitness / pop_size

    return {
        "avg_fitness": avg_fitness,
        "max_fitness": max_fitness,
        "min_fitness": min_fitness,
    }
