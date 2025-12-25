from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Sequence

from .genetic_agent import GeneticAgent


def setup_experiment_folder(base_path: str = "data") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(base_path, f"tournament_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def log_generation(
    population: Sequence[GeneticAgent],
    generation_num: int,
    folder_path: str,
) -> str:
    os.makedirs(folder_path, exist_ok=True)
    payload = [agent.to_dict() for agent in population]
    file_path = os.path.join(folder_path, f"gen_{generation_num}.json")
    with open(file_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return file_path
