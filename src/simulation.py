from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol, Sequence, Tuple

from config import DEFAULT_PAYOFFS, PayoffMatrix
from .genetic_agent import Action, History, COOPERATE, DEFECT, GeneticAgent


class ActionAgent(Protocol):
    def get_action(self, history: History) -> Action:
        ...

@dataclass(frozen=True)
class RoundLog:
    round_index: int
    action_a: Action
    action_b: Action
    payoff_a: int
    payoff_b: int
    cumulative_a: int
    cumulative_b: int


RoundCallback = Callable[[RoundLog, int], None]
MatchCallback = Callable[[int, int, int, int], None]


def _round_payoff(
    action_a: Action, action_b: Action, payoffs: PayoffMatrix
) -> Tuple[int, int]:
    if action_a == COOPERATE and action_b == COOPERATE:
        return payoffs.reward, payoffs.reward
    if action_a == COOPERATE and action_b == DEFECT:
        return payoffs.sucker, payoffs.temptation
    if action_a == DEFECT and action_b == COOPERATE:
        return payoffs.temptation, payoffs.sucker
    return payoffs.punishment, payoffs.punishment


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

    for _ in range(rounds):
        action_a = agent_a.get_action(history_a)
        action_b = agent_b.get_action(history_b)

        history_a.append((action_a, action_b))
        history_b.append((action_b, action_a))

        payoff_a, payoff_b = _round_payoff(action_a, action_b, payoffs)
        score_a += payoff_a
        score_b += payoff_b

    return score_a, score_b


def play_match_detailed(
    agent_a: ActionAgent,
    agent_b: ActionAgent,
    rounds: int,
    payoffs: PayoffMatrix = DEFAULT_PAYOFFS,
    round_callback: RoundCallback | None = None,
) -> Tuple[int, int, List[RoundLog]]:
    history_a: List[Tuple[Action, Action]] = []
    history_b: List[Tuple[Action, Action]] = []
    score_a = 0
    score_b = 0
    logs: List[RoundLog] = []

    for round_index in range(1, rounds + 1):
        action_a = agent_a.get_action(history_a)
        action_b = agent_b.get_action(history_b)

        history_a.append((action_a, action_b))
        history_b.append((action_b, action_a))

        payoff_a, payoff_b = _round_payoff(action_a, action_b, payoffs)
        score_a += payoff_a
        score_b += payoff_b

        log = RoundLog(
            round_index=round_index,
            action_a=action_a,
            action_b=action_b,
            payoff_a=payoff_a,
            payoff_b=payoff_b,
            cumulative_a=score_a,
            cumulative_b=score_b,
        )
        logs.append(log)
        if round_callback is not None:
            round_callback(log, rounds)

    return score_a, score_b, logs


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


def run_internal_tournament_with_progress(
    population: Sequence[GeneticAgent],
    rounds: int,
    payoffs: PayoffMatrix = DEFAULT_PAYOFFS,
    spotlight_pair: Tuple[int, int] | None = None,
    match_callback: MatchCallback | None = None,
    round_callback: RoundCallback | None = None,
) -> Tuple[Dict[str, float], List[RoundLog] | None]:
    if not population:
        return {"avg_fitness": 0.0, "max_fitness": 0.0, "min_fitness": 0.0}, None

    for agent in population:
        agent.reset_fitness()

    pop_list = list(population)
    pop_size = len(pop_list)
    total_matches = pop_size * (pop_size - 1) // 2
    match_index = 0
    spotlight_logs: List[RoundLog] | None = None

    for i in range(pop_size):
        agent_i = pop_list[i]
        for j in range(i + 1, pop_size):
            agent_j = pop_list[j]
            match_index += 1
            if match_callback is not None:
                match_callback(match_index, total_matches, agent_i.id, agent_j.id)

            is_spotlight = False
            if spotlight_pair is not None:
                a_id, b_id = spotlight_pair
                is_spotlight = (agent_i.id, agent_j.id) == (a_id, b_id) or (
                    agent_i.id,
                    agent_j.id,
                ) == (b_id, a_id)

            if is_spotlight:
                score_i, score_j, spotlight_logs = play_match_detailed(
                    agent_i,
                    agent_j,
                    rounds,
                    payoffs,
                    round_callback=round_callback,
                )
            else:
                score_i, score_j = play_match(agent_i, agent_j, rounds, payoffs)

            agent_i.fitness += score_i
            agent_j.fitness += score_j

    total_fitness = sum(agent.fitness for agent in pop_list)
    max_fitness = max(agent.fitness for agent in pop_list)
    min_fitness = min(agent.fitness for agent in pop_list)
    avg_fitness = total_fitness / pop_size

    return (
        {
            "avg_fitness": avg_fitness,
            "max_fitness": max_fitness,
            "min_fitness": min_fitness,
        },
        spotlight_logs,
    )
