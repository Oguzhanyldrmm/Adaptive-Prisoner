from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

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
from src.classic_strategies import get_classic_strategies
from src.evolution import evolve_population
from src.genetic_agent import GeneticAgent, random_genotype
from src.leaderboard import run_leaderboard
from src.logger import log_generation, setup_experiment_folder
from src.simulation import (
    RoundLog,
    play_match_detailed,
    run_internal_tournament,
    run_internal_tournament_with_progress,
)

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
    st.session_state.spotlight_logs = []
    st.session_state.spotlight_pair = None
    st.session_state.spotlight_generation = None
    st.session_state.benchmark_logs = []
    st.session_state.benchmark_opponent = None
    st.session_state.benchmark_generation = None
    st.session_state.log_folder = setup_experiment_folder(BASE_DATA_DIR)


def _action_label(action: int) -> str:
    return "C" if action == 1 else "D"


def _round_logs_to_df(
    logs: List[RoundLog], label_a: str, label_b: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round": [log.round_index for log in logs],
            f"{label_a} action": [_action_label(log.action_a) for log in logs],
            f"{label_b} action": [_action_label(log.action_b) for log in logs],
            f"{label_a} payoff": [log.payoff_a for log in logs],
            f"{label_b} payoff": [log.payoff_b for log in logs],
            f"{label_a} total": [log.cumulative_a for log in logs],
            f"{label_b} total": [log.cumulative_b for log in logs],
        }
    )


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

st.sidebar.subheader("Visualization")
detailed_tournament = st.sidebar.checkbox(
    "Detailed tournament progress (slower)", value=True
)
show_round_table = st.sidebar.checkbox("Show round-by-round tables", value=True)
spotlight_mode = st.sidebar.selectbox(
    "Spotlight match",
    options=["None", "Best vs Random", "Random Pair", "Custom IDs"],
    index=1,
)
spotlight_a_id = 0
spotlight_b_id = 1
if spotlight_mode == "Custom IDs":
    spotlight_a_id = int(
        st.sidebar.number_input(
            "Spotlight agent A ID",
            min_value=0,
            max_value=max(0, pop_size - 1),
            value=0,
            step=1,
        )
    )
    spotlight_b_id = int(
        st.sidebar.number_input(
            "Spotlight agent B ID",
            min_value=0,
            max_value=max(0, pop_size - 1),
            value=min(1, max(0, pop_size - 1)),
            step=1,
        )
    )
    if spotlight_a_id == spotlight_b_id:
        st.sidebar.error("Spotlight agents must be different.")

show_benchmark_rounds = st.sidebar.checkbox(
    "Show benchmark round-by-round", value=True
)
classic_strategies = get_classic_strategies()
classic_names = [strategy.name for strategy in classic_strategies]
selected_classic_name = st.sidebar.selectbox(
    "Benchmark spotlight opponent", options=classic_names, index=0
)
selected_classic = next(
    strategy for strategy in classic_strategies if strategy.name == selected_classic_name
)

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

metrics_container = st.container()
metric_cols = metrics_container.columns(3)
avg_metric = metric_cols[0].empty()
max_metric = metric_cols[1].empty()
min_metric = metric_cols[2].empty()

chart_placeholder = st.empty()

progress_container = st.container()
progress_cols = progress_container.columns(3)
gen_progress = progress_cols[0].progress(0.0, text="Generation")
match_progress = progress_cols[1].progress(0.0, text="Match")
round_progress = progress_cols[2].progress(0.0, text="Spotlight Round")
status_placeholder = progress_container.empty()

vis_cols = st.columns(2)
tournament_panel = vis_cols[0]
benchmark_panel = vis_cols[1]

with tournament_panel:
    st.subheader("Internal Tournament")
    tournament_caption = st.empty()
    tournament_rounds_placeholder = st.empty()

with benchmark_panel:
    st.subheader("Benchmark vs Classics")
    benchmark_round_progress = st.progress(0.0, text="Benchmark Round")
    benchmark_status = st.empty()
    leaderboard_placeholder = st.empty()
    benchmark_caption = st.empty()
    benchmark_rounds_placeholder = st.empty()

run_clicked = st.button("Run Simulation")
if run_clicked:
    population: List[GeneticAgent] = st.session_state.population
    rng: random.Random = st.session_state.rng
    history: Dict[str, List[Any]] = st.session_state.history

    for gen in range(st.session_state.generation, generations):
        gen_progress.progress(
            (gen + 1) / generations, text=f"Generation {gen + 1}/{generations}"
        )
        match_progress.progress(0.0, text="Match")
        round_progress.progress(0.0, text="Spotlight Round")

        spotlight_pair: Tuple[int, int] | None = None
        if pop_size > 1 and spotlight_mode != "None":
            if spotlight_mode == "Best vs Random":
                spotlight_pair = (0, rng.randrange(1, pop_size))
            elif spotlight_mode == "Random Pair":
                a_id = rng.randrange(0, pop_size)
                b_id = rng.randrange(0, pop_size - 1)
                if b_id >= a_id:
                    b_id += 1
                spotlight_pair = (a_id, b_id)
            else:
                if spotlight_a_id != spotlight_b_id:
                    spotlight_pair = (spotlight_a_id, spotlight_b_id)

        spotlight_label = "None"
        if spotlight_pair is not None:
            spotlight_label = f"A{spotlight_pair[0]} vs A{spotlight_pair[1]}"

        spotlight_logs: List[RoundLog] | None = None
        if detailed_tournament:
            progress_state: Dict[str, int] = {"match_index": 0, "match_total": 0}

            def match_callback(
                match_index: int, match_total: int, agent_a_id: int, agent_b_id: int
            ) -> None:
                progress_state["match_index"] = match_index
                progress_state["match_total"] = match_total
                if match_total > 0:
                    match_progress.progress(
                        match_index / match_total,
                        text=f"Match {match_index}/{match_total} (A{agent_a_id} vs A{agent_b_id})",
                    )
                status_placeholder.text(
                    f"Gen {gen + 1}/{generations} | Match {match_index}/{match_total} | Spotlight {spotlight_label}"
                )

            def round_callback(round_log: RoundLog, rounds_total: int) -> None:
                round_progress.progress(
                    round_log.round_index / rounds_total,
                    text=f"Spotlight Round {round_log.round_index}/{rounds_total}",
                )
                status_placeholder.text(
                    f"Gen {gen + 1}/{generations} | Match {progress_state['match_index']}/{progress_state['match_total']} "
                    f"| {spotlight_label} | Round {round_log.round_index}/{rounds_total} | "
                    f"Score {round_log.cumulative_a}-{round_log.cumulative_b}"
                )

            stats, spotlight_logs = run_internal_tournament_with_progress(
                population,
                rounds_per_match,
                spotlight_pair=spotlight_pair,
                match_callback=match_callback,
                round_callback=round_callback,
            )
        else:
            stats = run_internal_tournament(population, rounds_per_match)

        log_generation(population, gen, st.session_state.log_folder)

        history["generation"].append(gen)
        history["avg_fitness"].append(stats["avg_fitness"])
        history["max_fitness"].append(stats["max_fitness"])
        history["min_fitness"].append(stats["min_fitness"])

        history_df = pd.DataFrame(history).set_index("generation")

        avg_metric.metric("Avg Fitness", f"{stats['avg_fitness']:.2f}")
        max_metric.metric("Max Fitness", f"{stats['max_fitness']:.2f}")
        min_metric.metric("Min Fitness", f"{stats['min_fitness']:.2f}")

        chart_placeholder.line_chart(history_df)

        if spotlight_logs and show_round_table:
            st.session_state.spotlight_logs = spotlight_logs
            st.session_state.spotlight_pair = spotlight_pair
            st.session_state.spotlight_generation = gen

        if gen % benchmark_interval == 0:
            best_agent = max(population, key=lambda agent: agent.fitness)
            leaderboard = run_leaderboard(best_agent, rounds_per_match)
            st.session_state.leaderboard = leaderboard
            leaderboard_placeholder.dataframe(pd.DataFrame(leaderboard))

            if show_benchmark_rounds:
                benchmark_round_progress.progress(0.0, text="Benchmark Round")

                def benchmark_round_callback(
                    round_log: RoundLog, rounds_total: int
                ) -> None:
                    benchmark_round_progress.progress(
                        round_log.round_index / rounds_total,
                        text=f"Round {round_log.round_index}/{rounds_total}",
                    )
                    benchmark_status.text(
                        f"Best vs {selected_classic.name} | Round {round_log.round_index}/{rounds_total} "
                        f"| Score {round_log.cumulative_a}-{round_log.cumulative_b}"
                    )

                _, _, benchmark_logs = play_match_detailed(
                    best_agent,
                    selected_classic,
                    rounds_per_match,
                    round_callback=benchmark_round_callback,
                )
                st.session_state.benchmark_logs = benchmark_logs
                st.session_state.benchmark_opponent = selected_classic.name
                st.session_state.benchmark_generation = gen

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

if show_round_table and st.session_state.spotlight_logs:
    pair = st.session_state.spotlight_pair
    generation = st.session_state.spotlight_generation
    if pair is not None:
        label_a = f"A{pair[0]}"
        label_b = f"A{pair[1]}"
    else:
        label_a = "A"
        label_b = "B"
    if generation is not None:
        tournament_caption.caption(
            f"Spotlight match from generation {generation}: {label_a} vs {label_b}"
        )
    tournament_rounds_placeholder.dataframe(
        _round_logs_to_df(st.session_state.spotlight_logs, label_a, label_b),
        use_container_width=True,
    )

if show_benchmark_rounds and st.session_state.benchmark_logs:
    opponent = st.session_state.benchmark_opponent or selected_classic.name
    generation = st.session_state.benchmark_generation
    if generation is not None:
        benchmark_caption.caption(
            f"Benchmark spotlight from generation {generation}: Best vs {opponent}"
        )
    benchmark_rounds_placeholder.dataframe(
        _round_logs_to_df(st.session_state.benchmark_logs, "Best", opponent),
        use_container_width=True,
    )
