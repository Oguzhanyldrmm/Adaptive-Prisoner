from __future__ import annotations

import random
from typing import List, Sequence

from config import DEFAULT_ELITE_FRACTION
from .genetic_agent import GeneticAgent


def _select_parent(population: Sequence[GeneticAgent], rng: random.Random) -> GeneticAgent:
    total_fitness = sum(agent.fitness for agent in population)
    if total_fitness <= 0:
        return rng.choice(list(population))

    pick = rng.random() * total_fitness
    current = 0.0
    for agent in population:
        current += agent.fitness
        if current >= pick:
            return agent
    return population[-1]


def _crossover(
    genotype_a: Sequence[int],
    genotype_b: Sequence[int],
    rng: random.Random,
) -> List[int]:
    return [
        gene_a if rng.random() < 0.5 else gene_b
        for gene_a, gene_b in zip(genotype_a, genotype_b)
    ]


def _mutate(
    genotype: Sequence[int],
    mutation_rate: float,
    rng: random.Random,
) -> List[int]:
    if mutation_rate <= 0:
        return list(genotype)
    return [
        1 - gene if rng.random() < mutation_rate else gene for gene in genotype
    ]


def evolve_population(
    population: Sequence[GeneticAgent],
    mutation_rate: float,
    rng: random.Random,
    elite_fraction: float = DEFAULT_ELITE_FRACTION,
) -> List[GeneticAgent]:
    if not population:
        return []

    pop_list = list(population)
    pop_size = len(pop_list)
    elite_count = int(pop_size * elite_fraction)
    elite_count = max(0, min(elite_count, pop_size))

    sorted_pop = sorted(pop_list, key=lambda agent: agent.fitness, reverse=True)
    new_population: List[GeneticAgent] = []

    for _ in range(elite_count):
        elite = sorted_pop[len(new_population)]
        new_population.append(
            GeneticAgent(
                id=len(new_population),
                genotype=list(elite.genotype),
                fitness=0.0,
            )
        )

    while len(new_population) < pop_size:
        parent_a = _select_parent(sorted_pop, rng)
        parent_b = _select_parent(sorted_pop, rng)
        child_genotype = _crossover(parent_a.genotype, parent_b.genotype, rng)
        child_genotype = _mutate(child_genotype, mutation_rate, rng)
        new_population.append(
            GeneticAgent(
                id=len(new_population),
                genotype=child_genotype,
                fitness=0.0,
            )
        )

    return new_population
