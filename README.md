# 🧬 Evolutionary Iterated Prisoner's Dilemma

A Genetic Algorithm approach to evolving strategies for the Iterated Prisoner's Dilemma (IPD) game, featuring environmental stressors, behavioral profiling, and LLM-powered character analysis.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [The Prisoner's Dilemma](#the-prisoners-dilemma)
3. [System Architecture](#system-architecture)
4. [Genetic Algorithm Pipeline](#genetic-algorithm-pipeline)
5. [The 18-Gene Genome](#the-18-gene-genome)
6. [God Mode: Environmental Stressors](#god-mode-environmental-stressors)
7. [Behavioral Profiler](#behavioral-profiler)
8. [LLM Character Generator](#llm-character-generator)
9. [Installation](#installation)
10. [Usage Guide](#usage-guide)
11. [Example Scenario](#example-scenario)
12. [File Structure](#file-structure)
13. [Configuration](#configuration)

---

## Overview

This system evolves AI agents to play the **Iterated Prisoner's Dilemma** - a classic game theory scenario where two players repeatedly choose to either **Cooperate** or **Defect**. Through genetic algorithms, agents develop sophisticated strategies that can rival or surpass human-designed strategies like Tit-for-Tat.

### Key Features

- 🧬 **Genetic Evolution**: 18-gene genome encoding complete behavioral strategies
- ⚔️ **Tournament System**: Round-robin competition between all agents
- ⚡ **God Mode**: Environmental stressors testing agent resilience
- 🔬 **Behavioral Profiler**: Quantitative analysis of champion agents
- 🎭 **LLM Character Generator**: AI-powered personality profiles
- 📊 **Real-time Visualization**: Fitness charts, cooperation rates, rankings

---

## The Prisoner's Dilemma

### Payoff Matrix

|                    | Opponent: Cooperate | Opponent: Defect |
|--------------------|---------------------|------------------|
| **You: Cooperate** | R=3, R=3            | S=0, T=5         |
| **You: Defect**    | T=5, S=0            | P=1, P=1         |

- **T (Temptation)**: 5 points - Betray a cooperator
- **R (Reward)**: 3 points - Mutual cooperation
- **P (Punishment)**: 1 point - Mutual defection
- **S (Sucker)**: 0 points - Cooperate but get betrayed

### The Dilemma

Individually, defection always seems better (T > R, P > S). But collectively, mutual cooperation (R=3 each) beats mutual defection (P=1 each). This tension between individual and collective rationality makes the IPD a perfect testbed for evolving adaptive strategies.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │  Parameters  │ │   Charts     │ │    Scoreboards           │ │
│  │  - Pop Size  │ │ - Fitness    │ │  - Agent Rankings        │ │
│  │  - Mutation  │ │ - Coop Rate  │ │  - vs Classic Strategies │ │
│  │  - Rounds    │ │              │ │  - Genotype Decoder      │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SIMULATION ENGINE                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Tournament   │ │  Evolution   │ │      God Mode            │ │
│  │ (simulation) │ │  (evolution) │ │    (god_mode.py)         │ │
│  │              │ │              │ │  - Trembling Hand        │ │
│  │ Round-robin  │ │ - Selection  │ │  - Economic Crisis       │ │
│  │ matches      │ │ - Crossover  │ │  - High Temptation       │ │
│  │              │ │ - Mutation   │ │  - Memory Loss           │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS ENGINE                               │
│  ┌──────────────────────────┐ ┌────────────────────────────────┐│
│  │   Behavioral Profiler    │ │   LLM Character Generator      ││
│  │   (profiler.py)          │ │   (character_generator.py)     ││
│  │                          │ │                                ││
│  │ - Saint Test             │ │ - Genotype Decoder             ││
│  │ - Provocation Test       │ │ - Prompt Generator             ││
│  │ - Noise Tolerance Test   │ │ - OpenAI API Integration       ││
│  │ - Greed Test             │ │ - File Storage                 ││
│  └──────────────────────────┘ └────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Genetic Algorithm Pipeline

### 1. Population Initialization

```python
population = [GeneticAgent(id=i, genotype=random_18_bits()) for i in range(50)]
```

Each agent starts with a random 18-bit genotype encoding its complete strategy.

### 2. Tournament (Fitness Evaluation)

Every agent plays against every other agent in a **round-robin tournament**:

```
Matches = 50 × 49 / 2 = 1,225 matches per generation
Each match = 150 rounds
```

Fitness = total points accumulated across all matches.

### 3. Selection (Roulette Wheel)

We use **Roulette Wheel Selection** (also known as Fitness Proportionate Selection) to choose parents for the next generation. This is implemented in `src/evolution.py`.

#### How It Works

Imagine a roulette wheel where each agent gets a slice proportional to their fitness:

```
┌─────────────────────────────────────────┐
│                                         │
│    Agent A (fitness=1000) → 33% slice   │
│    Agent B (fitness=500)  → 17% slice   │
│    Agent C (fitness=800)  → 27% slice   │
│    Agent D (fitness=700)  → 23% slice   │
│                                         │
│         Total = 3000 → 100%             │
└─────────────────────────────────────────┘
```

**Visual representation:**

```
        ┌────────────────┐
       /   Agent A (33%) \
      /                   \
     │    Agent D (23%)    │
     │         🎯          │  ← Spin the wheel!
     │    Agent C (27%)    │
      \                   /
       \   Agent B (17%) /
        └────────────────┘
```

#### Selection Probability Formula

```python
probability(agent) = agent.fitness / total_fitness
```

Higher-fitness agents have greater chance of reproducing.

### 4. Crossover (Uniform)

For each gene position, randomly inherit from either parent:

```
Parent A: [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]
Parent B: [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
Child:    [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
          (random mix from both parents)
```

### 5. Mutation (Bit Flip)

Each gene has a 2% chance to flip:

```python
if random() < 0.02:
    gene = 1 - gene  # 0→1 or 1→0
```

### 6. Elitism

Top 10% of agents are copied directly to the next generation, preserving the best strategies.

---

## The 18-Gene Genome

Each agent's strategy is encoded in exactly **18 binary genes** (0 or 1).

### Gene Layout

| Gene Index | Purpose | Description |
|------------|---------|-------------|
| **0** | First Move | What to do on Round 1 (no history) |
| **1** | Second Move | What to do on Round 2 (1 round of history) |
| **2-17** | Response Table | 16 genes for all 2-round history combinations |

### Response Table (Genes 2-17)

After Round 2, the agent looks at **the last two rounds** to decide its action. Each round outcome is encoded as:

- **CC** = Both cooperated (index 0)
- **CD** = I cooperated, opponent defected (index 1)  
- **DC** = I defected, opponent cooperated (index 2)
- **DD** = Both defected (index 3)

The gene index for a given history is:

```
index = 2 + (outcome_two_rounds_ago × 4) + outcome_last_round
```

### Example: Decoding a Genome

```
Genome: [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]

Gene 0 = 1: First move → Cooperate
Gene 1 = 1: Second move → Cooperate

Response Table:
         t-2: CC  CD  DC  DD
t-1: CC  [1]  [0]  [0]  [1]  → If peace, cooperate; if I was betrayed, defect
     CD  [1]  [0]  [0]  [1]
     DC  [0]  [1]  [1]  [0]
     DD  [1]  [0]  [0]  [1]
```

This genome encodes a **Tit-for-Tat variant**: starts cooperating, retaliates when betrayed, forgives when opponent returns to cooperation.

---

## God Mode: Environmental Stressors

God Mode introduces **chaos** into the simulation to test agent resilience. Each rule triggers independently with its own probability.

### The 5 Rules

| Rule | Probability | Effect | Sociological Test |
|------|-------------|--------|-------------------|
| **Trembling Hand** | 5% per agent | Action is flipped (C→D or D→C) | Tests forgiveness of mistakes |
| **Economic Crisis** | 2% per round | All payoffs × 0.5 | Tests loyalty under scarcity |
| **High Temptation** | 10% per round | T increases from 5 to 10 | Tests resistance to greed |
| **Memory Loss** | 5% per agent | Agent sees empty history | Tests opening move tendency |
| **Information Leak** | 5% per agent | Agent sees opponent's move | Tests strategic adaptation |

### Rule Stacking

Rules can stack! In a single round, you might have:
- Economic Crisis (payoffs halved) + High Temptation (T=10 → T=5)
- Trembling Hand for Agent A + Memory Loss for Agent B

---

## Behavioral Profiler

The profiler runs **4 controlled experiments** on the champion agent to quantify its psychology.

### Test A: Saint Test

- **Opponent**: AlwaysCooperate (50 rounds)
- **Metric**: Number of defections
- **Labels**:
  - 0 defections → **"Altruist"**
  - 1-5 defections → **"Cautious Cooperator"**
  - >40 defections → **"Aggressive Exploiter"**
  - Else → **"Unpredictable"**

### Test B: Provocation Test

- **Opponent**: Provocateur (Cooperates, defects at Round 10, then cooperates forever)
- **Metric**: Rounds to return to cooperation after Round 10
- **Labels**:
  - Immediate (Round 11) → **"Instant Forgiveness"**
  - 2-3 rounds → **"Cautious Forgiveness"**
  - Never returns → **"None (Grim Trigger)"**

### Test C: Noise Tolerance Test

- **Opponent**: Tit-for-Tat
- **Setup**: Force agent to accidentally defect at Round 10
- **Metric**: Does agent play C in Round 11 or 12?
- **Labels**:
  - Yes → **"High (Resilient)"**
  - No → **"Low (Fragile)"**

### Test D: Greed Test

- **Opponent**: Tit-for-Tat
- **Setup**: Force High Temptation (T=10) for Rounds 20-30
- **Metric**: Defection rate increase during temptation period
- **Labels**:
  - Rate increase >50% → **"High (Opportunist)"**
  - Stable rate → **"Low (Principled)"**

---

## LLM Character Generator

After profiling, the system generates a **creative character profile** using OpenAI's GPT-5.1.

### Prompt Structure

```
**AGENT PROFILE:**

1. **Genetic Hardwiring (The Logic Map):**
   - START: Round 1 play Cooperate
   - SCENARIO [Peaceful]: If last 2 rounds were (CC, CC) → I will Cooperate
   - SCENARIO [Total War]: If last 2 rounds were (DD, DD) → I will Defect
   ...

2. **Social Disposition (Sterile Lab Test):**
   - Reaction to Kindness: Aggressive Exploiter (48/50 defections)
   - Reaction to Betrayal: None (Grim Trigger)

3. **Stress Response (Chaos Simulation):**
   - Tolerance to Mistakes: Low (Fragile)
   - Behavior during High Temptation: High (Opportunist)

**TASK:** Generate JSON with name, motto, description, rpg_alignment.
```

### Example LLM Output

```json
{
  "name": "The Ruthless Capitalist",
  "motto": "Trust is for the weak; profit is eternal.",
  "description": "This agent has evolved into a pure exploiter, defecting against kind opponents while showing no capacity for forgiveness. Its strategy maximizes short-term gains at the expense of long-term cooperation.",
  "rpg_alignment": "Neutral Evil"
}
```

### File Storage

Results are saved to:

```
data/tournament_{timestamp}/agent_{id}_champion/
├── input_telemetry.md    # The exact prompt sent to LLM
├── character_profile.json # LLM's JSON response
└── raw_stats.json        # Numerical profiler data
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/evolutionary-ipd.git
cd evolutionary-ipd

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key (optional, for LLM character generation)
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
python-dotenv>=1.0.0
openai>=1.0.0
```

---

## Usage Guide

### Starting the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### Sidebar Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Population size | 50 | Number of agents |
| Generations | 100 | Evolution cycles |
| Rounds per match | 150 | Rounds per head-to-head match |
| Mutation rate | 0.02 | Probability of gene flip |
| Elite fraction | 0.1 | Top % copied directly |
| Benchmark interval | 10 | Gens between classic strategy tests |

### Running a Simulation

1. Adjust parameters in the sidebar
2. Click **"▶️ Run Simulation"**
3. Watch real-time fitness and cooperation charts
4. View final rankings in **"🏆 Agent Rankings"**

### Analyzing the Champion

1. Complete a simulation
2. Scroll to **"🎭 Character Analysis"**
3. Click **"🔬 Analyze Champion"**
4. View behavioral test results and LLM-generated profile

---

## Example Scenario

### Step-by-Step Walkthrough

#### 1. Initial Setup

We configure a small experiment:
- **Population**: 20 agents
- **Generations**: 10
- **Rounds per match**: 100
- **God Mode**: Enabled

#### 2. Generation 0

The population is randomly initialized. Agent genotypes might include:
- Agent A0: `[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]` → Mostly cooperative
- Agent A5: `[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]` → Mostly defecting

**Tournament**: 20 × 19 / 2 = 190 matches × 100 rounds = 19,000 total rounds.

**Results**:
- Cooperative agents score well against each other (R=3 each round)
- Defectors exploit cooperators (T=5) but get low scores against each other (P=1)
- Fitness distribution is spread widely

#### 3. God Mode Event (Round 47, Match 83)

**Economic Crisis triggered!** All payoffs are halved:
- T: 5 → 2
- R: 3 → 1
- P: 1 → 0
- S: 0 → 0

Agents that relied on exploiting cooperators now get fewer points. Mutual cooperators are relatively less affected.

#### 4. Evolution to Generation 1

**Selection**: High-fitness agents (those who cooperated or used smart retaliation) have higher probability of being selected as parents.

**Crossover**: Agent A0's genes mix with Agent A3's genes:
```
A0:    [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
A3:    [1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1]
Child: [1,1,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,1]
```

**Mutation**: Gene 7 flips from 0 to 1:
```
Before: [...,0,0,1,...]
After:  [...,1,0,1,...]
```

**Elitism**: Top 2 agents (10% of 20) are copied unchanged.

#### 5. Generation 10 (Final)

After 10 cycles, the population has converged:
- **Champion (A4)**: Fitness 15,847
- Genotype: `[1,1,1,1,0,1,1,0,0,1,0,1,1,0,1,0,0,1]`

This genome encodes a **Tit-for-Tat variant** with:
- Cooperates first two rounds
- Retaliates when defected against
- Forgives after opponent cooperates

#### 6. Behavioral Profiling

We run the 4 tests on Agent A4:

| Test | Result | Label |
|------|--------|-------|
| Saint Test | 3 defections | Cautious Cooperator |
| Provocation | Returns in 1 round | Instant Forgiveness |
| Noise Tolerance | Plays C in R11 | High (Resilient) |
| Greed Test | 5% rate increase | Low (Principled) |

#### 7. LLM Character Generation

The profiler data is sent to GPT-5.1, which returns:

```json
{
  "name": "The Diplomatic Enforcer",
  "motto": "Peace is profitable, but betrayal is never forgotten.",
  "description": "This agent evolved to maximize cooperation while maintaining strict boundaries. It forgives quickly but retaliates precisely, creating an environment where cooperation is the optimal choice for all opponents.",
  "rpg_alignment": "Lawful Neutral"
}
```

#### 8. Final Output

Files are saved to `data/tournament_20251229_120000/agent_4_champion/`:
- `input_telemetry.md` - Complete prompt
- `character_profile.json` - LLM response
- `raw_stats.json` - All test numbers

---

## File Structure

```
evolutionary-ipd/
├── app.py                      # Streamlit UI
├── config.py                   # Configuration constants
├── requirements.txt            # Dependencies
├── .env                        # API keys (gitignored)
├── .env.example                # Template for .env
├── .gitignore                  # Git ignore patterns
├── README.md                   # This file
│
├── src/
│   ├── __init__.py
│   ├── genetic_agent.py        # GeneticAgent class
│   ├── evolution.py            # Selection, crossover, mutation
│   ├── simulation.py           # Match and tournament logic
│   ├── classic_strategies.py   # Tit-for-Tat, AlwaysCooperate, etc.
│   ├── leaderboard.py          # Benchmark vs classic strategies
│   ├── logger.py               # JSON logging per generation
│   ├── god_mode.py             # Environmental stressors
│   ├── profiler.py             # Behavioral tests
│   └── character_generator.py  # LLM integration
│
└── data/
    └── tournament_{timestamp}/
        ├── generation_0.json
        ├── generation_1.json
        ├── ...
        └── agent_{id}_champion/
            ├── input_telemetry.md
            ├── character_profile.json
            └── raw_stats.json
```

---

## Configuration

### config.py

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_POP_SIZE` | 50 | Population size |
| `DEFAULT_GENERATIONS` | 100 | Evolution cycles |
| `DEFAULT_MUTATION_RATE` | 0.02 | Mutation probability |
| `DEFAULT_ROUNDS_PER_MATCH` | 150 | Rounds per match |
| `DEFAULT_ELITE_FRACTION` | 0.1 | Elitism percentage |
| `DEFAULT_BENCHMARK_INTERVAL` | 10 | Benchmark frequency |
| `GENOTYPE_LENGTH` | 18 | Fixed genome size |

### God Mode Probabilities

| Rule | Default | 
|------|---------|
| `trembling_hand` | 0.05 |
| `economic_crisis` | 0.02 |
| `high_temptation` | 0.10 |
| `memory_loss` | 0.05 |
| `information_leak` | 0.05 |

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-your-api-key-here
```

---

## License

MIT License - feel free to use, modify, and distribute.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Acknowledgments

- Robert Axelrod's "The Evolution of Cooperation"
- OpenAI for LLM API
- Streamlit for the visualization framework
