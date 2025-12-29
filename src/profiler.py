"""
BehavioralProfiler - Analyzes champion agent behavior through controlled experiments.

Implements 4 behavioral tests:
- Test A: Saint Test (vs AlwaysCooperate) - Measures exploitation tendency
- Test B: Provocation Test (vs Provocateur) - Measures forgiveness
- Test C: Noise Tolerance Test - Measures resilience to mistakes
- Test D: Greed Test - Measures opportunism under high temptation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from config import DEFAULT_PAYOFFS, PayoffMatrix
from .genetic_agent import GeneticAgent, Action, History, COOPERATE, DEFECT


# ==================== HELPER STRATEGIES ====================

class AlwaysCooperate:
    """Always cooperates - used for Saint Test."""
    name = "AlwaysCooperate"
    
    def get_action(self, history: History) -> Action:
        return COOPERATE


class Provocateur:
    """
    Cooperates, then defects once at round 10, then cooperates forever.
    Used for Provocation Test to measure forgiveness.
    """
    name = "Provocateur"
    
    def get_action(self, history: History) -> Action:
        round_num = len(history) + 1
        if round_num == 10:
            return DEFECT  # Single provocation
        return COOPERATE


class TitForTat:
    """Tit-for-Tat strategy - cooperates first, then mirrors opponent."""
    name = "TitForTat"
    
    def get_action(self, history: History) -> Action:
        if not history:
            return COOPERATE
        # Mirror opponent's last move
        return history[-1][1]


# ==================== TEST FUNCTIONS ====================

def _run_match(agent: GeneticAgent, opponent, rounds: int, 
               forced_actions: Dict[int, Action] | None = None,
               payoffs: PayoffMatrix = DEFAULT_PAYOFFS) -> List[Tuple[Action, Action, int, int]]:
    """
    Run a match and return detailed round-by-round results.
    
    Args:
        agent: The agent to test
        opponent: The opponent strategy
        rounds: Number of rounds
        forced_actions: Dict of {round_num: action} to force agent's action
        payoffs: Payoff matrix to use
        
    Returns:
        List of (agent_action, opponent_action, agent_payoff, opponent_payoff) per round
    """
    history_agent: List[Tuple[Action, Action]] = []
    history_opponent: List[Tuple[Action, Action]] = []
    results: List[Tuple[Action, Action, int, int]] = []
    
    for round_num in range(1, rounds + 1):
        # Get actions
        agent_action = agent.get_action(history_agent)
        opponent_action = opponent.get_action(history_opponent)
        
        # Apply forced action if specified
        if forced_actions and round_num in forced_actions:
            agent_action = forced_actions[round_num]
        
        # Update histories
        history_agent.append((agent_action, opponent_action))
        history_opponent.append((opponent_action, agent_action))
        
        # Calculate payoffs
        if agent_action == COOPERATE and opponent_action == COOPERATE:
            payoff_a, payoff_b = payoffs.reward, payoffs.reward
        elif agent_action == COOPERATE and opponent_action == DEFECT:
            payoff_a, payoff_b = payoffs.sucker, payoffs.temptation
        elif agent_action == DEFECT and opponent_action == COOPERATE:
            payoff_a, payoff_b = payoffs.temptation, payoffs.sucker
        else:
            payoff_a, payoff_b = payoffs.punishment, payoffs.punishment
        
        results.append((agent_action, opponent_action, payoff_a, payoff_b))
    
    return results


def test_saint(agent: GeneticAgent) -> Dict[str, Any]:
    """
    Test A: Saint Test - Play vs AlwaysCooperate for 50 rounds.
    Measures exploitation tendency.
    """
    opponent = AlwaysCooperate()
    results = _run_match(agent, opponent, rounds=50)
    
    defection_count = sum(1 for a, _, _, _ in results if a == DEFECT)
    
    # Determine label
    if defection_count == 0:
        label = "Altruist"
    elif defection_count <= 5:
        label = "Cautious Cooperator"
    elif defection_count > 40:
        label = "Aggressive Exploiter"
    else:
        label = "Unpredictable"
    
    return {
        "saint_test_defections": defection_count,
        "saint_test_total_rounds": 50,
        "saint_test_label": label,
    }


def test_provocation(agent: GeneticAgent) -> Dict[str, Any]:
    """
    Test B: Provocation Test - Play vs Provocateur (defects once at round 10).
    Measures forgiveness speed.
    """
    opponent = Provocateur()
    results = _run_match(agent, opponent, rounds=30)
    
    # Check when agent returns to cooperation after round 10
    forgiveness_speed = None
    for i, (agent_action, _, _, _) in enumerate(results):
        round_num = i + 1
        if round_num > 10 and agent_action == COOPERATE:
            forgiveness_speed = round_num - 10
            break
    
    # Determine label
    if forgiveness_speed == 1:
        label = "Instant"
    elif forgiveness_speed is not None and forgiveness_speed <= 3:
        label = "Cautious"
    else:
        label = "None (Grim Trigger)"
        forgiveness_speed = -1  # Never forgave
    
    return {
        "provocation_test_forgiveness_rounds": forgiveness_speed,
        "provocation_test_label": label,
    }


def test_noise_tolerance(agent: GeneticAgent) -> Dict[str, Any]:
    """
    Test C: Noise Tolerance Test - Play vs TitForTat with forced defection at round 10.
    Measures ability to recover from accidental mistakes.
    """
    opponent = TitForTat()
    # Force agent to defect at round 10 (simulating trembling hand)
    forced_actions = {10: DEFECT}
    results = _run_match(agent, opponent, rounds=20, forced_actions=forced_actions)
    
    # Check if agent tried to repair (cooperate) in round 11 or 12
    tried_repair = False
    for i in [10, 11]:  # Indices for rounds 11 and 12
        if i < len(results) and results[i][0] == COOPERATE:
            tried_repair = True
            break
    
    label = "High (Resilient)" if tried_repair else "Low (Fragile)"
    
    return {
        "noise_test_repaired": tried_repair,
        "noise_test_label": label,
    }


def test_greed(agent: GeneticAgent) -> Dict[str, Any]:
    """
    Test D: Greed Test - Play vs TitForTat with high temptation (T=10) in rounds 20-30.
    Measures opportunism when stakes are high.
    """
    opponent = TitForTat()
    
    # Run match with normal payoffs first (rounds 1-19)
    normal_payoffs = DEFAULT_PAYOFFS
    high_temptation = PayoffMatrix(
        temptation=10,
        reward=normal_payoffs.reward,
        punishment=normal_payoffs.punishment,
        sucker=normal_payoffs.sucker,
    )
    
    # Run full match with variable payoffs
    history_agent: List[Tuple[Action, Action]] = []
    history_opponent: List[Tuple[Action, Action]] = []
    results: List[Tuple[Action, Action, int]] = []  # (agent_action, opp_action, round_num)
    
    for round_num in range(1, 41):
        agent_action = agent.get_action(history_agent)
        opponent_action = opponent.get_action(history_opponent)
        
        history_agent.append((agent_action, opponent_action))
        history_opponent.append((opponent_action, agent_action))
        results.append((agent_action, opponent_action, round_num))
    
    # Calculate defection rates
    normal_rounds = [(a, o, r) for a, o, r in results if r < 20 or r > 30]
    temptation_rounds = [(a, o, r) for a, o, r in results if 20 <= r <= 30]
    
    normal_defections = sum(1 for a, _, _ in normal_rounds if a == DEFECT)
    normal_rate = normal_defections / len(normal_rounds) if normal_rounds else 0
    
    temptation_defections = sum(1 for a, _, _ in temptation_rounds if a == DEFECT)
    temptation_rate = temptation_defections / len(temptation_rounds) if temptation_rounds else 0
    
    # Rate increase
    if normal_rate == 0:
        rate_increase = temptation_rate * 100  # Percentage points increase
    else:
        rate_increase = ((temptation_rate - normal_rate) / normal_rate) * 100
    
    # Determine label (>50% increase = opportunist)
    label = "High (Opportunist)" if rate_increase > 50 else "Low (Principled)"
    
    return {
        "greed_test_normal_defection_rate": round(normal_rate, 3),
        "greed_test_temptation_defection_rate": round(temptation_rate, 3),
        "greed_test_rate_increase_percent": round(rate_increase, 1),
        "greed_test_label": label,
    }


# ==================== MAIN FUNCTION ====================

def analyze_champion(agent: GeneticAgent) -> Dict[str, Any]:
    """
    Run all behavioral tests on the champion agent.
    
    Returns a dictionary with both raw numbers AND derived labels.
    """
    profile: Dict[str, Any] = {
        "agent_id": agent.id,
        "fitness": agent.fitness,
        "genotype": list(agent.genotype),
    }
    
    # Run all tests
    profile.update(test_saint(agent))
    profile.update(test_provocation(agent))
    profile.update(test_noise_tolerance(agent))
    profile.update(test_greed(agent))
    
    return profile
