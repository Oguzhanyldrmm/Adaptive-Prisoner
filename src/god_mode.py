"""
GodEngine - Environmental Stressor System

Implements 5 sociological rules that can affect agent behavior and payoffs:
1. Trembling Hand - Accidental betrayal (action flip)
2. Economic Crisis - Scarcity of resources (payoff reduction)
3. High Temptation - Greed test (increased temptation payoff)
4. Memory Loss - Amnesia (empty history)
5. Information Leak - The Spy (see opponent's move)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from config import GOD_MODE_CONFIG, PayoffMatrix


@dataclass
class GodModeResult:
    """Result of applying God Mode rules to a round."""
    action_a: int
    action_b: int
    history_a: List[Tuple[int, int]]
    history_b: List[Tuple[int, int]]
    payoffs: PayoffMatrix
    active_rules: List[str]
    info_leak_a: int | None  # Opponent's move if agent A has info leak
    info_leak_b: int | None  # Opponent's move if agent B has info leak


class GodEngine:
    """
    Environmental stressor engine that applies sociological rules
    to test agent resilience and adaptability.
    """
    
    def __init__(
        self,
        config: Dict[str, float] | None = None,
        rng: random.Random | None = None,
    ):
        """
        Initialize the GodEngine.
        
        Args:
            config: Probability configuration for each rule.
            rng: Random number generator for reproducibility.
        """
        self.config = config or GOD_MODE_CONFIG.copy()
        self.rng = rng or random.Random()
    
    def _should_trigger(self, rule_name: str) -> bool:
        """Check if a rule should trigger based on its probability."""
        prob = self.config.get(rule_name, 0.0)
        return self.rng.random() < prob
    
    def apply_environment_rules(
        self,
        action_a: int,
        action_b: int,
        history_a: List[Tuple[int, int]],
        history_b: List[Tuple[int, int]],
        payoffs: PayoffMatrix,
    ) -> GodModeResult:
        """
        Apply all environmental rules to a single round.
        
        Rules are applied independently (can stack).
        
        Args:
            action_a: Agent A's intended action (0=Defect, 1=Cooperate)
            action_b: Agent B's intended action
            history_a: Agent A's view of history
            history_b: Agent B's view of history
            payoffs: Current payoff matrix
            
        Returns:
            GodModeResult with modified values and list of active rules.
        """
        modified_action_a = action_a
        modified_action_b = action_b
        modified_history_a = history_a
        modified_history_b = history_b
        modified_payoffs = payoffs
        active_rules: List[str] = []
        info_leak_a: int | None = None
        info_leak_b: int | None = None
        
        # Rule 1: Trembling Hand - Accidental betrayal
        # Each agent independently might have their action flipped
        if self._should_trigger("trembling_hand"):
            modified_action_a = 1 - modified_action_a  # Flip: 0->1, 1->0
            active_rules.append("trembling_hand_a")
        
        if self._should_trigger("trembling_hand"):
            modified_action_b = 1 - modified_action_b
            active_rules.append("trembling_hand_b")
        
        # Rule 2: Economic Crisis - Scarcity of resources
        # All payoffs are halved for this round
        if self._should_trigger("economic_crisis"):
            modified_payoffs = PayoffMatrix(
                temptation=int(payoffs.temptation * 0.5),
                reward=int(payoffs.reward * 0.5),
                punishment=int(payoffs.punishment * 0.5),
                sucker=int(payoffs.sucker * 0.5),
            )
            active_rules.append("economic_crisis")
        
        # Rule 3: High Temptation - Greed test
        # Temptation payoff is doubled (5 -> 10)
        if self._should_trigger("high_temptation"):
            modified_payoffs = PayoffMatrix(
                temptation=modified_payoffs.temptation * 2,
                reward=modified_payoffs.reward,
                punishment=modified_payoffs.punishment,
                sucker=modified_payoffs.sucker,
            )
            active_rules.append("high_temptation")
        
        # Rule 4: Memory Loss - Amnesia
        # Agent sees empty history, forcing use of opening move gene
        if self._should_trigger("memory_loss"):
            modified_history_a = []
            active_rules.append("memory_loss_a")
        
        if self._should_trigger("memory_loss"):
            modified_history_b = []
            active_rules.append("memory_loss_b")
        
        # Rule 5: Information Leak - The Spy
        # Agent gets to see opponent's intended move
        if self._should_trigger("information_leak"):
            info_leak_a = action_b  # A sees B's move
            active_rules.append("info_leak_a")
        
        if self._should_trigger("information_leak"):
            info_leak_b = action_a  # B sees A's move
            active_rules.append("info_leak_b")
        
        return GodModeResult(
            action_a=modified_action_a,
            action_b=modified_action_b,
            history_a=modified_history_a,
            history_b=modified_history_b,
            payoffs=modified_payoffs,
            active_rules=active_rules,
            info_leak_a=info_leak_a,
            info_leak_b=info_leak_b,
        )
