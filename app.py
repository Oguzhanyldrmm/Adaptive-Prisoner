from __future__ import annotations

import copy
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
    DEFAULT_PAYOFFS,
    GENOTYPE_LENGTH,
)
from src.classic_strategies import get_classic_strategies
from src.evolution import evolve_population
from src.genetic_agent import GeneticAgent, random_genotype, COOPERATE, DEFECT
from src.god_mode import GodEngine
from src.leaderboard import run_leaderboard
from src.logger import log_generation, setup_experiment_folder
from src.simulation import (
    RoundLog,
    play_match_detailed,
    play_match,
    run_internal_tournament,
    run_internal_tournament_with_progress,
)
from src.character_generator import generate_character_profile, decode_genotype_to_text

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
        "cooperation_rate": [],
    }
    st.session_state.leaderboard = []
    st.session_state.spotlight_logs = []
    st.session_state.spotlight_pair = None
    st.session_state.spotlight_generation = None
    st.session_state.benchmark_logs = []
    st.session_state.benchmark_opponent = None
    st.session_state.benchmark_generation = None
    st.session_state.log_folder = setup_experiment_folder(BASE_DATA_DIR)
    st.session_state.final_population = []
    st.session_state.simulation_completed = False


def _action_label(action: int) -> str:
    return "C" if action == 1 else "D"


def _action_emoji(action: int) -> str:
    return "🤝" if action == 1 else "⚔️"


def _round_logs_to_df(
    logs: List[RoundLog], label_a: str, label_b: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Round": [log.round_index for log in logs],
            f"{label_a}": [_action_emoji(log.action_a) for log in logs],
            f"{label_b}": [_action_emoji(log.action_b) for log in logs],
            f"{label_a} Payoff": [log.payoff_a for log in logs],
            f"{label_b} Payoff": [log.payoff_b for log in logs],
            f"{label_a} Total": [log.cumulative_a for log in logs],
            f"{label_b} Total": [log.cumulative_b for log in logs],
        }
    )


def _genotype_to_str(genotype: List[int]) -> str:
    return "".join(str(g) for g in genotype)


def _calculate_cooperation_rate(population: List[GeneticAgent]) -> float:
    """Calculate the average cooperation tendency of the population based on genotype."""
    if not population:
        return 0.0
    total_coop = sum(sum(agent.genotype) for agent in population)
    return total_coop / (len(population) * GENOTYPE_LENGTH)


def _get_result_emoji(agent_score: float, opponent_score: float) -> str:
    if agent_score > opponent_score:
        return "✅ WIN"
    elif agent_score < opponent_score:
        return "❌ LOSS"
    return "➖ TIE"


def _render_payoff_matrix():
    """Render the Prisoner's Dilemma payoff matrix."""
    st.markdown("### 📊 Payoff Matrix")
    payoff_data = {
        "": ["You: **C**ooperate", "You: **D**efect"],
        "Opponent: **C**ooperate": [
            f"R={DEFAULT_PAYOFFS.reward}, R={DEFAULT_PAYOFFS.reward}",
            f"T={DEFAULT_PAYOFFS.temptation}, S={DEFAULT_PAYOFFS.sucker}",
        ],
        "Opponent: **D**efect": [
            f"S={DEFAULT_PAYOFFS.sucker}, T={DEFAULT_PAYOFFS.temptation}",
            f"P={DEFAULT_PAYOFFS.punishment}, P={DEFAULT_PAYOFFS.punishment}",
        ],
    }
    st.table(pd.DataFrame(payoff_data).set_index(""))
    st.caption("T=Temptation, R=Reward, P=Punishment, S=Sucker")


def _render_genotype_decoder(agent: GeneticAgent):
    """Render a visual decoder for the agent's genotype."""
    st.markdown("### 🧬 Genotype Decoder")
    st.markdown(f"**Agent A{agent.id}** | Fitness: `{agent.fitness:.0f}`")
    
    genotype = agent.genotype
    
    # First two moves
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Gene 0** (First move): {_action_emoji(genotype[0])} `{_action_label(genotype[0])}`")
    with col2:
        st.markdown(f"**Gene 1** (Second move): {_action_emoji(genotype[1])} `{_action_label(genotype[1])}`")
    
    # Response table (4x4 grid for 2-round history)
    st.markdown("**Response Table** (Genes 2-17)")
    st.caption("Rows: Previous round outcome | Columns: Two rounds ago outcome")
    
    outcomes = ["CC", "CD", "DC", "DD"]
    response_matrix = []
    for i, row_outcome in enumerate(outcomes):
        row = []
        for j, col_outcome in enumerate(outcomes):
            gene_idx = 2 + j * 4 + i
            action = genotype[gene_idx] if gene_idx < len(genotype) else 0
            row.append(_action_emoji(action))
        response_matrix.append(row)
    
    df = pd.DataFrame(response_matrix, index=outcomes, columns=outcomes)
    df.index.name = "t-1"
    st.dataframe(df, use_container_width=True)
    st.caption("🤝 = Cooperate, ⚔️ = Defect")


def _render_agents_scoreboard(population: List[GeneticAgent], elite_fraction: float):
    """Render the ranked scoreboard of all genetic agents."""
    st.markdown("### 🏆 Agent Rankings")
    
    sorted_pop = sorted(population, key=lambda a: a.fitness, reverse=True)
    elite_count = int(len(population) * elite_fraction)
    
    data = []
    for rank, agent in enumerate(sorted_pop, 1):
        is_elite = rank <= elite_count
        data.append({
            "Rank": rank,
            "Agent": f"A{agent.id}",
            "Genotype": _genotype_to_str(agent.genotype)[:8] + "...",
            "Fitness": f"{agent.fitness:.0f}",
            "Elite": "⭐" if is_elite else "",
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"⭐ = Elite (Top {int(elite_fraction * 100)}%, copied to next generation)")


def _render_classics_scoreboard(leaderboard: List[Dict[str, float]]):
    """Render the best agent vs classic strategies scoreboard."""
    st.markdown("### 🎯 Best Agent vs. Classic Strategies")
    
    if not leaderboard:
        st.info("Run simulation to see benchmark results.")
        return
    
    data = []
    wins, losses, ties = 0, 0, 0
    for entry in leaderboard:
        agent_score = entry["agent_score"]
        opponent_score = entry["opponent_score"]
        result = _get_result_emoji(agent_score, opponent_score)
        
        if "WIN" in result:
            wins += 1
        elif "LOSS" in result:
            losses += 1
        else:
            ties += 1
        
        data.append({
            "Opponent": entry["opponent"],
            "Agent Score": f"{agent_score:.0f}",
            "Opponent Score": f"{opponent_score:.0f}",
            "Result": result,
            "Avg/Round": f"{entry['agent_avg']:.2f}",
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown(f"**Summary**: {wins} Wins ✅ | {losses} Losses ❌ | {ties} Ties ➖")


def _render_match_emoji_sequence(logs: List[RoundLog], max_rounds: int = 50):
    """Render an emoji-based visualization of the match."""
    if not logs:
        return
    
    st.markdown("**Match Sequence** (🤝=C, ⚔️=D)")
    
    display_logs = logs[:max_rounds]
    sequence_a = " ".join(_action_emoji(log.action_a) for log in display_logs)
    sequence_b = " ".join(_action_emoji(log.action_b) for log in display_logs)
    
    st.text(f"Agent A: {sequence_a}")
    st.text(f"Agent B: {sequence_b}")
    
    if len(logs) > max_rounds:
        st.caption(f"(Showing first {max_rounds} of {len(logs)} rounds)")


# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Evolutionary IPD", layout="wide")

st.title("🎮 Evolutionary Iterated Prisoner's Dilemma")
st.markdown("*Genetic Algorithm Approach to Game Theory*")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Simulation Controls")
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

st.sidebar.subheader("📊 Visualization")
detailed_tournament = st.sidebar.checkbox(
    "Detailed tournament progress (slower)", value=True
)
show_payoff_matrix = st.sidebar.checkbox("Show payoff matrix", value=True)
show_genotype_decoder = st.sidebar.checkbox("Show genotype decoder", value=True)

st.sidebar.subheader("⚡ God Mode")
god_mode_enabled = st.sidebar.checkbox(
    "Enable God Mode (Environmental Stressors)", value=False
)
if god_mode_enabled:
    st.sidebar.caption("🎲 Rules: Trembling Hand (5%), Economic Crisis (2%), High Temptation (10%), Memory Loss (5%), Info Leak (5%)")

seed_value: Optional[int] = None
if seed_input.strip():
    try:
        seed_value = int(seed_input)
    except ValueError:
        st.sidebar.error("Seed must be an integer.")

reset_clicked = st.sidebar.button("🔄 Initialize / Reset")

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

st.caption(f"📁 Logging to: `{st.session_state.log_folder}`")

# ==================== MAIN LAYOUT ====================

# Top row: Metrics and Payoff Matrix
top_cols = st.columns([2, 1]) if show_payoff_matrix else [st.container()]

with top_cols[0]:
    metrics_container = st.container()
    metric_cols = metrics_container.columns(4)
    avg_metric = metric_cols[0].empty()
    max_metric = metric_cols[1].empty()
    min_metric = metric_cols[2].empty()
    coop_metric = metric_cols[3].empty()

if show_payoff_matrix and len(top_cols) > 1:
    with top_cols[1]:
        _render_payoff_matrix()

# Charts row
chart_cols = st.columns(2)
with chart_cols[0]:
    st.markdown("### 📈 Fitness Over Generations")
    chart_placeholder = st.empty()
with chart_cols[1]:
    st.markdown("### 🤝 Cooperation Rate Over Generations")
    coop_chart_placeholder = st.empty()

# Progress bars
progress_container = st.container()
progress_cols = progress_container.columns(2)
gen_progress = progress_cols[0].progress(0.0, text="Generation")
match_progress = progress_cols[1].progress(0.0, text="Match")
status_placeholder = progress_container.empty()

# God Mode Event Log (only shown when God Mode is enabled)
god_event_log_placeholder = st.empty()

# Run button
run_clicked = st.button("▶️ Run Simulation", type="primary")

st.divider()

# Scoreboards row
scoreboard_cols = st.columns(2)

with scoreboard_cols[0]:
    agents_scoreboard_placeholder = st.empty()

with scoreboard_cols[1]:
    classics_scoreboard_placeholder = st.empty()

st.divider()

# Genotype decoder section
genotype_decoder_placeholder = st.empty()

# ==================== SIMULATION LOGIC ====================
if run_clicked:
    population: List[GeneticAgent] = st.session_state.population
    rng: random.Random = st.session_state.rng
    history: Dict[str, List[Any]] = st.session_state.history
    
    # Initialize God Engine if enabled
    god_engine: GodEngine | None = None
    god_events: List[str] = []  # Store recent events
    god_event_counter = [0]  # Use list to allow mutation in closure
    
    # Rule emoji and description mapping
    RULE_INFO = {
        "trembling_hand_a": ("🤚", "Agent A's hand trembled!"),
        "trembling_hand_b": ("🤚", "Agent B's hand trembled!"),
        "economic_crisis": ("📉", "ECONOMIC CRISIS - Payoffs halved!"),
        "high_temptation": ("💰", "HIGH TEMPTATION - Greed test!"),
        "memory_loss_a": ("🧠", "Agent A forgot everything!"),
        "memory_loss_b": ("🧠", "Agent B forgot everything!"),
        "info_leak_a": ("🕵️", "Agent A is SPYING!"),
        "info_leak_b": ("🕵️", "Agent B is SPYING!"),
    }
    
    def god_event_callback(round_num: int, active_rules: List[str], agent_a_id: int, agent_b_id: int) -> None:
        """Called when God Mode rules are triggered during a match. Just collects events."""
        for rule in active_rules:
            emoji, desc = RULE_INFO.get(rule, ("⚡", rule))
            event_msg = f"{emoji} R{round_num} | A{agent_a_id} vs A{agent_b_id}: {desc}"
            god_events.append(event_msg)
            god_event_counter[0] += 1
            # Keep only last 50 events in memory (display shows last 10)
            if len(god_events) > 50:
                god_events.pop(0)
    
    if god_mode_enabled:
        god_engine = GodEngine(rng=rng)
        status_placeholder.info("⚡ God Mode ACTIVE - Environmental stressors enabled")

    for gen in range(st.session_state.generation, generations):
        gen_progress.progress(
            (gen + 1) / generations, text=f"Generation {gen + 1}/{generations}"
        )
        match_progress.progress(0.0, text="Match")
        
        # Clear events for new generation
        god_events.clear()
        god_event_counter[0] = 0

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
                god_mode_label = " ⚡" if god_mode_enabled else ""
                status_placeholder.text(
                    f"Gen {gen + 1}/{generations} | Match {match_index}/{match_total}{god_mode_label}"
                )
                
                # Update god event log during match updates (synchronized with progress bar)
                if god_mode_enabled and god_events:
                    with god_event_log_placeholder.container():
                        st.markdown("### ⚡ God Mode Event Log")
                        for event in reversed(god_events[-10:]):
                            st.text(event)

            stats, _ = run_internal_tournament_with_progress(
                population,
                rounds_per_match,
                spotlight_pair=None,
                match_callback=match_callback,
                round_callback=None,
                god_engine=god_engine,
                god_event_callback=god_event_callback if god_mode_enabled else None,
            )
        else:
            stats = run_internal_tournament(
                population, 
                rounds_per_match, 
                god_engine=god_engine,
                god_event_callback=god_event_callback if god_mode_enabled else None,
            )

        log_generation(population, gen, st.session_state.log_folder)

        # Calculate cooperation rate
        coop_rate = _calculate_cooperation_rate(population)

        history["generation"].append(gen)
        history["avg_fitness"].append(stats["avg_fitness"])
        history["max_fitness"].append(stats["max_fitness"])
        history["min_fitness"].append(stats["min_fitness"])
        history["cooperation_rate"].append(coop_rate)

        # Update fitness chart
        fitness_df = pd.DataFrame({
            "Avg": history["avg_fitness"],
            "Max": history["max_fitness"],
            "Min": history["min_fitness"],
        }, index=history["generation"])
        chart_placeholder.line_chart(fitness_df)

        # Update cooperation chart
        coop_df = pd.DataFrame({
            "Cooperation %": [r * 100 for r in history["cooperation_rate"]],
        }, index=history["generation"])
        coop_chart_placeholder.line_chart(coop_df)

        avg_metric.metric("Avg Fitness", f"{stats['avg_fitness']:.0f}")
        max_metric.metric("Max Fitness", f"{stats['max_fitness']:.0f}")
        min_metric.metric("Min Fitness", f"{stats['min_fitness']:.0f}")
        coop_metric.metric("Cooperation %", f"{coop_rate * 100:.1f}%")

        # Update agents scoreboard
        with agents_scoreboard_placeholder.container():
            _render_agents_scoreboard(population, elite_fraction)

        if gen % benchmark_interval == 0:
            best_agent = max(population, key=lambda agent: agent.fitness)
            leaderboard = run_leaderboard(best_agent, rounds_per_match)
            st.session_state.leaderboard = leaderboard
            
            # Update classics scoreboard
            with classics_scoreboard_placeholder.container():
                _render_classics_scoreboard(leaderboard)

        # Save final population before evolution (to preserve fitness scores)
        st.session_state.final_population = copy.deepcopy(population)

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

    # Mark simulation as completed
    st.session_state.simulation_completed = True

# ==================== STATIC DISPLAY (after simulation) ====================

# Show completion status
if st.session_state.get("simulation_completed") and st.session_state.get("final_population"):
    status_placeholder.success(f"✅ Simulation completed! Final results from generation {st.session_state.generation}")

# Display leaderboards if available
if st.session_state.leaderboard:
    with classics_scoreboard_placeholder.container():
        _render_classics_scoreboard(st.session_state.leaderboard)

# Use final_population (with fitness scores) if available, otherwise use current population
display_population = st.session_state.get("final_population") or st.session_state.get("population")
if display_population:
    with agents_scoreboard_placeholder.container():
        _render_agents_scoreboard(display_population, elite_fraction)

# Display charts if history available
if st.session_state.history["generation"]:
    fitness_df = pd.DataFrame({
        "Avg": st.session_state.history["avg_fitness"],
        "Max": st.session_state.history["max_fitness"],
        "Min": st.session_state.history["min_fitness"],
    }, index=st.session_state.history["generation"])
    chart_placeholder.line_chart(fitness_df)
    
    if st.session_state.history.get("cooperation_rate"):
        coop_df = pd.DataFrame({
            "Cooperation %": [r * 100 for r in st.session_state.history["cooperation_rate"]],
        }, index=st.session_state.history["generation"])
        coop_chart_placeholder.line_chart(coop_df)

# Display genotype decoder for best agent
if show_genotype_decoder:
    decoder_population = st.session_state.get("final_population") or st.session_state.get("population")
    if decoder_population:
        best_agent = max(decoder_population, key=lambda a: a.fitness)
        with genotype_decoder_placeholder.container():
            _render_genotype_decoder(best_agent)

# ==================== CHARACTER ANALYSIS ====================
st.divider()
st.markdown("### 🎭 Character Analysis")

if st.session_state.get("simulation_completed") and st.session_state.get("final_population"):
    analyze_col1, analyze_col2 = st.columns([1, 2])
    
    with analyze_col1:
        analyze_clicked = st.button("🔬 Analyze Champion", type="secondary")
    
    with analyze_col2:
        st.caption("Run behavioral tests and generate character profile with LLM")
    
    if analyze_clicked:
        best_agent = max(st.session_state.final_population, key=lambda a: a.fitness)
        
        with st.spinner("Running behavioral tests..."):
            result = generate_character_profile(
                agent=best_agent,
                base_dir=BASE_DATA_DIR,
            )
        
        st.success(f"✅ Analysis complete! Files saved to: `{result['save_dir']}`")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Behavioral Profile")
            raw_stats = result['raw_stats']
            st.markdown(f"**Saint Test:** {raw_stats.get('saint_test_label', 'N/A')} ({raw_stats.get('saint_test_defections', 0)}/50 defections)")
            st.markdown(f"**Provocation Test:** {raw_stats.get('provocation_test_label', 'N/A')}")
            st.markdown(f"**Noise Tolerance:** {raw_stats.get('noise_test_label', 'N/A')}")
            st.markdown(f"**Greed Test:** {raw_stats.get('greed_test_label', 'N/A')}")
        
        with col2:
            if result.get('llm_response'):
                st.markdown("#### 🎭 Character Profile")
                llm = result['llm_response']
                st.markdown(f"**Name:** {llm.get('name', 'Unknown')}")
                st.markdown(f"**Motto:** _{llm.get('motto', 'N/A')}_")
                st.markdown(f"**Alignment:** {llm.get('rpg_alignment', 'Unknown')}")
                st.markdown(f"**Analysis:** {llm.get('description', 'N/A')}")
            else:
                st.warning("LLM response not available. Set OPENAI_API_KEY in config.py")
else:
    st.info("Run a simulation first to analyze the champion agent.")
