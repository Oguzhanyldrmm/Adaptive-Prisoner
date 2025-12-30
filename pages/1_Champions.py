"""
Champions Page - All-Star Tournament Mode

Run tournaments between previously saved champions with their LLM-generated
character names and profiles.
"""
import streamlit as st
import pandas as pd
import random
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hall_of_fame import get_all_champions, reconstruct_agent
from src.simulation import run_internal_tournament
from src.genetic_agent import GeneticAgent

st.set_page_config(
    page_title="Champions Tournament",
    page_icon="🏆",
    layout="wide",
)

st.title("🏆 Champions Hall of Fame")
st.markdown("Select champions from previous sessions to battle in an All-Star Tournament!")

# Load all champions
champions = get_all_champions()

if not champions:
    st.warning("No champions found! Run a simulation and analyze the champion to add them to the Hall of Fame.")
    st.info("Go to the main simulation page, run a tournament, then click '🔬 Analyze Champion' to save champions.")
    st.stop()

# Display champions for selection
st.subheader(f"📜 Available Champions ({len(champions)})")

# Create selection UI
selected_ids = []
champion_cols = st.columns(min(3, len(champions)))

for idx, champion in enumerate(champions):
    col_idx = idx % 3
    with champion_cols[col_idx]:
        with st.container(border=True):
            st.markdown(f"### {champion.get('character_name', 'Unknown')}")
            st.caption(f"_{champion.get('motto', 'No motto')}_")
            st.markdown(f"**Alignment:** {champion.get('rpg_alignment', 'Unknown')}")
            st.markdown(f"**Original Fitness:** {champion.get('original_fitness', 0):,}")
            st.caption(f"Created: {champion.get('created_at', 'Unknown')[:10]}")
            
            if st.checkbox("Select", key=f"select_{champion['id']}"):
                selected_ids.append(champion['id'])

st.divider()

# Tournament section
st.subheader("⚔️ All-Star Tournament")

if len(selected_ids) < 2:
    st.info("Select at least 2 champions to start a tournament.")
else:
    st.success(f"✅ {len(selected_ids)} champions selected for tournament!")
    
    # Tournament settings
    settings_cols = st.columns(3)
    with settings_cols[0]:
        rounds_per_match = st.number_input("Rounds per match", 50, 500, 150, 10)
    with settings_cols[1]:
        num_tournaments = st.number_input("Number of tournaments", 1, 10, 3)
    with settings_cols[2]:
        random_seed = st.number_input("Random seed", 0, 99999, 42)
    
    if st.button("🏟️ Start All-Star Tournament", type="primary"):
        # Get selected champions
        selected_champions = [c for c in champions if c['id'] in selected_ids]
        
        # Reconstruct agents
        agents: List[GeneticAgent] = []
        champion_map: Dict[int, Dict[str, Any]] = {}  # agent_id -> champion
        
        for idx, champ in enumerate(selected_champions):
            agent = reconstruct_agent(champ, agent_id=idx)
            agents.append(agent)
            champion_map[idx] = champ
        
        # Run tournaments
        rng = random.Random(random_seed)
        
        progress_bar = st.progress(0.0, text="Running tournaments...")
        results_placeholder = st.empty()
        
        aggregate_fitness = {agent.id: 0 for agent in agents}
        
        for t in range(num_tournaments):
            progress_bar.progress((t + 1) / num_tournaments, text=f"Tournament {t+1}/{num_tournaments}")
            
            # Reset fitness
            for agent in agents:
                agent.reset_fitness()
            
            # Run round-robin tournament
            stats = run_internal_tournament(agents, rounds_per_match)
            
            # Aggregate fitness
            for agent in agents:
                aggregate_fitness[agent.id] += agent.fitness
        
        progress_bar.progress(1.0, text="Tournaments complete!")
        
        # Create results dataframe
        results_data = []
        for agent in agents:
            champ = champion_map[agent.id]
            results_data.append({
                "Rank": 0,
                "Champion": champ.get('character_name', f'Champion {agent.id}'),
                "Alignment": champ.get('rpg_alignment', 'Unknown'),
                "Total Fitness": aggregate_fitness[agent.id],
                "Avg Fitness": aggregate_fitness[agent.id] / num_tournaments,
            })
        
        # Sort by fitness and assign ranks
        results_data.sort(key=lambda x: x["Total Fitness"], reverse=True)
        for idx, row in enumerate(results_data):
            row["Rank"] = idx + 1
        
        results_df = pd.DataFrame(results_data)
        
        # Display results
        st.markdown("### 🏆 Tournament Results")
        
        # Winner announcement
        winner = results_data[0]
        st.success(f"🎉 **WINNER: {winner['Champion']}** ({winner['Alignment']}) with {winner['Total Fitness']:,} total fitness!")
        
        # Results table
        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True,
        )
        
        # Fitness comparison chart
        st.markdown("### 📊 Fitness Comparison")
        chart_df = pd.DataFrame({
            "Champion": [d["Champion"] for d in results_data],
            "Average Fitness": [d["Avg Fitness"] for d in results_data],
        })
        st.bar_chart(chart_df.set_index("Champion"))

# Footer
st.divider()
st.caption("💡 Tip: Analyze more champions in the main simulation to add them to the Hall of Fame!")
