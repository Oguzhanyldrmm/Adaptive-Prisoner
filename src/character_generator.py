"""
Character Generator - Generates LLM-based character profiles for champion agents.

Includes:
- Genotype decoder for readable rules
- LLM prompt generator
- OpenAI API integration
- File storage for analysis results
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import OPENAI_API_KEY


def decode_genotype_to_text(genotype: List[int]) -> str:
    """
    Convert 18-bit genome to human-readable rules.
    """
    rules = []
    
    # First 2 genes (opening moves)
    rules.append(f"START: Round 1 play {'Cooperate' if genotype[0] else 'Defect'}.")
    rules.append(f"START: Round 2 play {'Cooperate' if genotype[1] else 'Defect'}.")
    
    # Response table (genes 2-17)
    # ID Logic: (Old_State * 4) + New_State
    # States: CC=0, CD=1, DC=2, DD=3
    table = genotype[2:]
    
    # Key scenarios
    scenarios = [
        (0, "Peaceful", "CC, CC", "Peace continues"),
        (5, "Betrayed Once", "CC, CD", "I was betrayed after peace"),
        (10, "Revenge Success", "CD, DC", "I retaliated successfully"),
        (15, "Total War", "DD, DD", "Mutual defection continues"),
        (3, "Forgiveness Test", "CC, DD", "Both defected after peace"),
        (12, "Recovery", "DC, CC", "Peace restored after chaos"),
    ]
    
    for idx, name, history, description in scenarios:
        if idx < len(table):
            action = 'Cooperate' if table[idx] else 'Defect'
            rules.append(f"SCENARIO [{name}]: If last 2 rounds were ({history}) -> I will {action}. ({description})")
    
    return "\n".join(rules)


def generate_llm_prompt(profile: Dict[str, Any]) -> str:
    """
    Generate the structured LLM prompt from profiler data.
    """
    # Decode genome
    genotype = profile.get("genotype", [])
    decoded_rules = decode_genotype_to_text(genotype) if genotype else "No genotype available"
    
    prompt = f'''You are an expert Game Theory Psychologist. I have analyzed an AI agent's behavior in a series of controlled experiments (Prisoner's Dilemma).

**AGENT PROFILE:**

1. **Genetic Hardwiring (The Logic Map):**
   - Decode the genome bits into text rules here:
{decoded_rules}

2. **Social Disposition (Sterile Lab Test):**
   - Reaction to Kindness: {profile.get('saint_test_label', 'Unknown')} (Raw Defections: {profile.get('saint_test_defections', 'N/A')}/50)
   - Reaction to Betrayal: {profile.get('provocation_test_label', 'Unknown')} (Forgiveness Speed: {profile.get('provocation_test_forgiveness_rounds', 'N/A')} rounds)

3. **Stress Response (Chaos Simulation):**
   - Tolerance to Mistakes: {profile.get('noise_test_label', 'Unknown')}
   - Behavior during High Temptation: {profile.get('greed_test_label', 'Unknown')} (Rate increase: {profile.get('greed_test_rate_increase_percent', 'N/A')}%)

**TASK:**
Based strictly on the data above, generate a JSON response with:
- 'name': A creative, archetypal name (e.g., 'The Ruthless Capitalist', 'The Stoic Monk').
- 'motto': A one-sentence quote representing their philosophy.
- 'description': A 2-sentence psychological analysis of why they behave this way.
- 'rpg_alignment': (e.g., Lawful Evil, Chaotic Good).

Return ONLY valid JSON, no markdown formatting.'''
    
    return prompt


def call_llm_api(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Call OpenAI API to generate character profile.
    Returns the parsed JSON response or None on failure.
    """
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set in config.py")
        return None
    
    try:
        import openai
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a game theory psychologist. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON from response
        # Handle potential markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        return json.loads(content)
        
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response as JSON: {e}")
        return None
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None


def save_character_analysis(
    base_dir: str,
    agent_id: int,
    prompt: str,
    llm_response: Optional[Dict[str, Any]],
    raw_stats: Dict[str, Any],
    timestamp: Optional[str] = None,
) -> str:
    """
    Save all character analysis files.
    
    Returns the directory path where files were saved.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure
    dir_path = os.path.join(base_dir, f"tournament_{timestamp}", f"agent_{agent_id}_champion")
    os.makedirs(dir_path, exist_ok=True)
    
    # File 1: input_telemetry.md (the prompt)
    telemetry_path = os.path.join(dir_path, "input_telemetry.md")
    with open(telemetry_path, "w", encoding="utf-8") as f:
        f.write("# LLM Input Telemetry\n\n")
        f.write("## Prompt Sent to LLM\n\n")
        f.write("```\n")
        f.write(prompt)
        f.write("\n```\n")
    
    # File 2: character_profile.json (LLM response)
    profile_path = os.path.join(dir_path, "character_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        if llm_response:
            json.dump(llm_response, f, indent=2, ensure_ascii=False)
        else:
            json.dump({"error": "No LLM response available"}, f, indent=2)
    
    # File 3: raw_stats.json (profiler data)
    stats_path = os.path.join(dir_path, "raw_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(raw_stats, f, indent=2, ensure_ascii=False)
    
    return dir_path


def generate_character_profile(
    agent,
    base_dir: str,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: profile agent, generate prompt, call LLM, save files.
    
    Returns dict with profile data and file paths.
    """
    from .profiler import analyze_champion
    
    # Step 1: Run behavioral tests
    raw_stats = analyze_champion(agent)
    
    # Step 2: Generate prompt
    prompt = generate_llm_prompt(raw_stats)
    
    # Step 3: Call LLM
    llm_response = call_llm_api(prompt)
    
    # Step 4: Save files
    save_dir = save_character_analysis(
        base_dir=base_dir,
        agent_id=agent.id,
        prompt=prompt,
        llm_response=llm_response,
        raw_stats=raw_stats,
        timestamp=timestamp,
    )
    
    return {
        "raw_stats": raw_stats,
        "llm_response": llm_response,
        "prompt": prompt,
        "save_dir": save_dir,
    }
