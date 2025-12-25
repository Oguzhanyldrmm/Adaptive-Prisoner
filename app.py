from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from config import (
    DEFAULT_BENCHMARK_INTERVAL,
    DEFAULT_ELITE_FRACTION,
    DEFAULT_GENERATIONS,
    DEFAULT_MUTATION_RATE,
    DEFAULT_POP_SIZE,
    DEFAULT_ROUNDS_PER_MATCH,
)
from src.evolution import evolve_population
from src.genetic_agent import GeneticAgent, random_genotype
from src.leaderboard import run_leaderboard
from src.logger import log_generation, setup_experiment_folder
from src.simulation import run_internal_tournament

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _init_population(pop_size: int, rng: random.Random) -> List[GeneticAgent]:
    return [
        GeneticAgent(id=idx, genotype=random_genotype(rng), fitness=0.0)
        for idx in range(pop_size)
    ]


def _initialize_session(pop_size: int, seed: Optional[int]) -> None:
    rng = random.Random(seed) if seed is not None else random.Random()
    st.session_state.population = _init_population(pop_size, rng)
    st.session_state.rng = rng
    st.session_state.generation = 0
    st.session_state.history = {
        "generation": [],
        "avg_fitness": [],
        "max_fitness": [],
        "min_fitness": [],
    }
    st.session_state.leaderboard = []
    st.session_state.log_folder = setup_experiment_folder(BASE_DATA_DIR)


st.set_page_config(page_title="Evolutionary IPD", layout="wide")

st.title("Evolutionary Iterated Prisoner's Dilemma")

st.sidebar.header("Simulation Controls")
pop_size = st.sidebar.number_input(
    "Population size",
    min_value=10,
    max_value=500,
    value=DEFAULT_POP_SIZE,
    step=10,
)
generations = st.sidebar.number_input(
    "Generations",
    min_value=1,
    max_value=500,
    value=DEFAULT_GENERATIONS,
    step=1,
)
rounds_per_match = st.sidebar.number_input(
    "Rounds per match",
    min_value=10,
    max_value=500,
    value=DEFAULT_ROUNDS_PER_MATCH,
    step=10,
)
mutation_rate = st.sidebar.slider(
    "Mutation rate",
    min_value=0.0,
    max_value=0.2,
    value=DEFAULT_MUTATION_RATE,
    step=0.005,
)
elite_fraction = st.sidebar.slider(
    "Elite fraction",
    min_value=0.0,
    max_value=0.5,
    value=DEFAULT_ELITE_FRACTION,
    step=0.05,
)
benchmark_interval = st.sidebar.number_input(
    "Benchmark interval",
    min_value=1,
    max_value=100,
    value=DEFAULT_BENCHMARK_INTERVAL,
    step=1,
)
animation_delay = st.sidebar.slider(
    "Animation delay (seconds)",
    min_value=0.0,
    max_value=0.5,
    value=0.05,
    step=0.01,
)
seed_input = st.sidebar.text_input("Random seed (optional)", value="")

seed_value: Optional[int] = None
if seed_input.strip():
    try:
        seed_value = int(seed_input)
    except ValueError:
        st.sidebar.error("Seed must be an integer.")

reset_clicked = st.sidebar.button("Initialize / Reset")

if (
    "population" not in st.session_state
    or "pop_size" not in st.session_state
    or reset_clicked
    or st.session_state.pop_size != pop_size
    or st.session_state.get("seed_value") != seed_value
):
    _initialize_session(pop_size, seed_value)
    st.session_state.pop_size = pop_size
    st.session_state.seed_value = seed_value

st.caption(f"Logging to: {st.session_state.log_folder}")

metrics_placeholder = st.empty()
chart_placeholder = st.empty()
leaderboard_placeholder = st.empty()

run_clicked = st.button("Run Simulation")
if run_clicked:
    population: List[GeneticAgent] = st.session_state.population
    rng: random.Random = st.session_state.rng
    history: Dict[str, List[Any]] = st.session_state.history

    for gen in range(st.session_state.generation, generations):
        stats = run_internal_tournament(population, rounds_per_match)
        log_generation(population, gen, st.session_state.log_folder)

        history["generation"].append(gen)
        history["avg_fitness"].append(stats["avg_fitness"])
        history["max_fitness"].append(stats["max_fitness"])
        history["min_fitness"].append(stats["min_fitness"])

        history_df = pd.DataFrame(history).set_index("generation")

        cols = metrics_placeholder.columns(3)
        cols[0].metric("Avg Fitness", f"{stats['avg_fitness']:.2f}")
        cols[1].metric("Max Fitness", f"{stats['max_fitness']:.2f}")
        cols[2].metric("Min Fitness", f"{stats['min_fitness']:.2f}")

        chart_placeholder.line_chart(history_df)

        if gen % benchmark_interval == 0:
            best_agent = max(population, key=lambda agent: agent.fitness)
            leaderboard = run_leaderboard(best_agent, rounds_per_match)
            st.session_state.leaderboard = leaderboard
            leaderboard_placeholder.dataframe(pd.DataFrame(leaderboard))

        population = evolve_population(
            population,
            mutation_rate=mutation_rate,
            rng=rng,
            elite_fraction=elite_fraction,
        )
        st.session_state.population = population
        st.session_state.generation = gen + 1

        if animation_delay > 0:
            time.sleep(animation_delay)

if st.session_state.leaderboard:
    leaderboard_placeholder.dataframe(pd.DataFrame(st.session_state.leaderboard))

if st.session_state.history["generation"]:
    history_df = pd.DataFrame(st.session_state.history).set_index("generation")
    chart_placeholder.line_chart(history_df)
