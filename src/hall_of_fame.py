"""
Hall of Fame - Champion storage and management.

Stores champion genotypes with their LLM-generated character profiles
for the All-Star Tournament feature.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .genetic_agent import GeneticAgent

# Default path for hall of fame storage
HALL_OF_FAME_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hall_of_fame.json")


def _load_hall_of_fame(path: str = HALL_OF_FAME_PATH) -> List[Dict[str, Any]]:
    """Load existing hall of fame or return empty list."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_hall_of_fame(champions: List[Dict[str, Any]], path: str = HALL_OF_FAME_PATH) -> None:
    """Save hall of fame to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(champions, f, indent=2, ensure_ascii=False)


def add_champion(
    genotype: List[int],
    character_name: str,
    motto: str = "",
    rpg_alignment: str = "",
    description: str = "",
    original_fitness: int = 0,
    session_id: Optional[str] = None,
    path: str = HALL_OF_FAME_PATH,
) -> str:
    """
    Add a champion to the hall of fame.
    
    Returns the champion ID.
    """
    champions = _load_hall_of_fame(path)
    
    # Generate unique ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    champion_id = f"champion_{timestamp}_{len(champions)}"
    
    champion = {
        "id": champion_id,
        "genotype": genotype,
        "character_name": character_name,
        "motto": motto,
        "rpg_alignment": rpg_alignment,
        "description": description,
        "original_fitness": original_fitness,
        "session_id": session_id or timestamp,
        "created_at": datetime.now().isoformat(),
    }
    
    champions.append(champion)
    _save_hall_of_fame(champions, path)
    
    return champion_id


def get_all_champions(path: str = HALL_OF_FAME_PATH) -> List[Dict[str, Any]]:
    """Get all champions from hall of fame."""
    return _load_hall_of_fame(path)


def get_champion_by_id(champion_id: str, path: str = HALL_OF_FAME_PATH) -> Optional[Dict[str, Any]]:
    """Get a specific champion by ID."""
    champions = _load_hall_of_fame(path)
    for champion in champions:
        if champion["id"] == champion_id:
            return champion
    return None


def delete_champion(champion_id: str, path: str = HALL_OF_FAME_PATH) -> bool:
    """Delete a champion from hall of fame."""
    champions = _load_hall_of_fame(path)
    original_count = len(champions)
    champions = [c for c in champions if c["id"] != champion_id]
    
    if len(champions) < original_count:
        _save_hall_of_fame(champions, path)
        return True
    return False


def reconstruct_agent(champion: Dict[str, Any], agent_id: int = 0) -> GeneticAgent:
    """
    Reconstruct a GeneticAgent from a champion's genotype.
    
    Args:
        champion: Champion dict from hall of fame
        agent_id: ID to assign to the reconstructed agent
        
    Returns:
        GeneticAgent with the champion's genotype
    """
    genotype = tuple(champion["genotype"])
    return GeneticAgent(id=agent_id, genotype=genotype)
