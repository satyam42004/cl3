# ============================================================
#   Clonal Selection Algorithm (CSA) - Simple Python Program
# ============================================================
# HOW TO RUN:
#   python clonal_selection.py
#
# WHAT IT DOES:
#   Uses the Clonal Selection Algorithm to find the minimum
#   value of a simple math function: f(x) = x^2
#   The correct answer is x = 0, where f(0) = 0.
# ============================================================

import random
import math

# ── STEP 0: Settings ─────────────────────────────────────────

POPULATION_SIZE = 10       # Total number of candidate solutions (antibodies)
NUM_GENERATIONS = 20       # How many rounds of evolution
NUM_CLONES      = 5        # How many clones each antibody makes
MUTATION_RATE   = 0.5      # How much clones change (mutation strength)
NUM_RANDOM_NEW  = 2        # New random antibodies added each generation

# Search space: x values between -10 and 10
LOWER_BOUND = -10
UPPER_BOUND =  10


# ── STEP 1: Objective Function ────────────────────────────────
# This is the problem we want to solve.
# We want to MINIMIZE f(x) = x^2 (answer: x = 0)

def objective_function(x):
    return x ** 2


# ── STEP 2: Affinity (fitness) ────────────────────────────────
# Higher affinity = better solution
# We use 1 / (1 + cost) so that lower cost → higher affinity

def affinity(x):
    cost = objective_function(x)
    return 1.0 / (1.0 + cost)


# ── STEP 3: Create a random antibody ─────────────────────────

def random_antibody():
    return random.uniform(LOWER_BOUND, UPPER_BOUND)


# ── STEP 4: Clone an antibody ────────────────────────────────
# Better antibodies get more clones (here we keep it simple: fixed clones)

def clone(antibody):
    return [antibody for _ in range(NUM_CLONES)]


# ── STEP 5: Hypermutation ────────────────────────────────────
# Add random noise to a clone to explore nearby solutions.
# High affinity → small changes (almost perfect, explore nearby)
# Low affinity  → big changes (explore widely)

def mutate(clone_val, aff):
    mutation_strength = MUTATION_RATE * math.exp(-aff)  # smaller if affinity is high
    new_val = clone_val + random.gauss(0, mutation_strength)
    # Keep within bounds
    new_val = max(LOWER_BOUND, min(UPPER_BOUND, new_val))
    return new_val


# ── MAIN ALGORITHM ────────────────────────────────────────────

def clonal_selection():
    print("=" * 50)
    print("   Clonal Selection Algorithm (CSA)")
    print("   Minimize f(x) = x^2   [Answer: x = 0]")
    print("=" * 50)

    # Step A: Create initial population randomly
    population = [random_antibody() for _ in range(POPULATION_SIZE)]

    best_x    = None
    best_cost = float('inf')

    for generation in range(1, NUM_GENERATIONS + 1):

        # Step B: Calculate affinity for each antibody
        scored = [(x, affinity(x)) for x in population]

        # Step C: Sort by affinity (best first)
        scored.sort(key=lambda item: item[1], reverse=True)

        new_population = []

        # Step D: Clone and mutate each antibody
        for x, aff in scored:
            clones = clone(x)                           # make copies
            mutated_clones = [mutate(c, aff) for c in clones]  # mutate each copy

            # Keep the best mutated clone
            best_clone = min(mutated_clones, key=objective_function)
            new_population.append(best_clone)

        # Step E: Add new random antibodies to maintain diversity
        for _ in range(NUM_RANDOM_NEW):
            new_population.append(random_antibody())

        # Step F: Keep only the top POPULATION_SIZE antibodies
        new_population.sort(key=objective_function)
        population = new_population[:POPULATION_SIZE]

        # Track global best
        gen_best_x    = min(population, key=objective_function)
        gen_best_cost = objective_function(gen_best_x)

        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_x    = gen_best_x

        print(f"  Generation {generation:>2} | Best x = {gen_best_x:>8.4f} | f(x) = {gen_best_cost:.6f}")

    print("=" * 50)
    print(f"  FINAL ANSWER:")
    print(f"  Best x    = {best_x:.6f}")
    print(f"  f(best_x) = {best_cost:.6f}   (closer to 0 = better)")
    print("=" * 50)


# ── Run the program ───────────────────────────────────────────
if __name__ == "__main__":
    clonal_selection()
