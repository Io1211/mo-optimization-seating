"""
Combines a population-based global search (a permutation genetic algorithm with
order crossover + swap mutation) with an individual-level local search
(first-improvement 2-swap). Each offspring is locally refined before it competes.

Objective: the same scalar fitness the other single-objective methods use
(reward_function = -energy + lux - distance), evaluated with the cached
FastEvaluator so the many local-search moves stay cheap.
"""
import os
import sys

if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import json
import time
import numpy as np

import common.Preprocessing as pp
from common.fast_reward import FastEvaluator


class MemeticSeating:
    def __init__(self, users, zones, evaluator,
                 population_size, generations,
                 crossover_probability, mutation_probability,
                 local_search_iters, objective="scalar",
                 tournament_size=3, elite=2):
        self.users = list(users)
        self.zones = list(zones)
        self.n = len(self.users)
        self.ev = evaluator
        self.objective = objective
        self.reward_fn = evaluator.reward_bench if objective == "bench" else evaluator.reward
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.local_search_iters = local_search_iters
        self.tournament_size = tournament_size
        self.elite = elite
        self.rng = None
        self._start_time = 0.0
        self._benchmark_hit = False

    def _decode(self, perm):
        return {self.users[perm[j]]: self.zones[j] for j in range(self.n)}

    def _fitness(self, perm):
        return self.reward_fn(self._decode(perm))

    def _local_search(self, perm, fit):
        """Try `local_search_iters` random 2-swaps; keep any that improve.

        A bounded (constant-cost) hill climber -- predictable runtime while
        still refining every offspring, which is what makes this memetic.
        """
        perm = perm.copy()
        for _ in range(self.local_search_iters):
            a, b = self.rng.choice(self.n, size=2, replace=False)
            perm[a], perm[b] = perm[b], perm[a]
            new_fit = self._fitness(perm)
            if new_fit > fit + 1e-9:
                fit = new_fit
            else:
                perm[a], perm[b] = perm[b], perm[a]  # revert
        return perm, fit

    def _crossover(self, p1, p2):
        n = self.n
        i, j = sorted(self.rng.choice(n, size=2, replace=False))
        child = -np.ones(n, dtype=int)
        child[i:j + 1] = p1[i:j + 1]
        taken = set(p1[i:j + 1].tolist())
        fill = [g for g in p2 if g not in taken]
        pos = 0
        for k in range(n):
            if child[k] == -1:
                child[k] = fill[pos]
                pos += 1
        return child

    def _mutate(self, perm):
        perm = perm.copy()
        for _ in range(self.n):
            if self.rng.random() < self.mutation_probability:
                a, b = self.rng.choice(self.n, size=2, replace=False)
                perm[a], perm[b] = perm[b], perm[a]
        return perm

    def _tournament(self, pop, fits):
        idx = self.rng.choice(len(pop), size=self.tournament_size, replace=False)
        best = idx[np.argmax(fits[idx])]
        return pop[best]

    def run(self, seed, benchmark_path=None, hp=None):
        self.rng = np.random.default_rng(seed)
        self._start_time = time.time()

        # initial population
        pop = [self.rng.permutation(self.n) for _ in range(self.population_size)]
        fits = np.array([self._fitness(p) for p in pop])
        initial_reward = float(fits.max())

        best_perm = pop[int(np.argmax(fits))].copy()
        best_fit = float(fits.max())
        self._maybe_log_benchmark(best_perm, best_fit, initial_reward, benchmark_path, hp)

        for gen in range(1, self.generations + 1):
            order = np.argsort(-fits)
            new_pop = [pop[order[k]].copy() for k in range(self.elite)]
            new_fits = [float(fits[order[k]]) for k in range(self.elite)]

            while len(new_pop) < self.population_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                child = self._crossover(p1, p2) if self.rng.random() < self.crossover_probability else p1.copy()
                child = self._mutate(child)
                cfit = self._fitness(child)
                # memetic refinement
                child, cfit = self._local_search(child, cfit)
                new_pop.append(child)
                new_fits.append(cfit)

            pop = new_pop
            fits = np.array(new_fits)
            gen_best = int(np.argmax(fits))
            if fits[gen_best] > best_fit:
                best_fit = float(fits[gen_best])
                best_perm = pop[gen_best].copy()
            self._maybe_log_benchmark(best_perm, best_fit, initial_reward, benchmark_path, hp)

        execution_time = time.time() - self._start_time
        best_ass = self._decode(best_perm)
        return best_ass, best_fit, initial_reward, execution_time

    def _maybe_log_benchmark(self, perm, best_fit, initial_reward, benchmark_path, hp):
        if not benchmark_path or self._benchmark_hit:
            return
        ass = self._decode(perm)
        en, lux, dist = self.ev.evaluate(ass)
        if self.ev.benchmark(ass):
            write_benchmark(hp, initial_reward, best_fit,
                            time.time() - self._start_time, ass, en, lux, dist, benchmark_path)
            self._benchmark_hit = True


def _ensure_dir(filename):
    d = os.path.dirname(filename)
    if d:
        os.makedirs(d, exist_ok=True)


def write_summary(hp, seed, initial_reward, final_reward, execution_time, assignment,
                  energy, lux, dist, filename):
    _ensure_dir(filename)
    exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if not exists:
            w.writerow(["population_size", "generations", "crossover_probability",
                        "mutation_probability", "local_search_iters", "objective", "seed",
                        "initial_reward", "final_reward", "execution_time",
                        "final_assignment", "final_energy", "final_lux",
                        "final_weighted_distance"])
        w.writerow([hp["population_size"], hp["generations"], hp["crossover_probability"],
                    hp["mutation_probability"], hp["local_search_iters"], hp["objective"], seed,
                    initial_reward, final_reward, execution_time,
                    json.dumps(assignment), energy, lux, dist])


def write_benchmark(hp, initial_reward, best_reward, execution_time, assignment,
                    energy, lux, dist, filename):
    _ensure_dir(filename)
    exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if not exists:
            w.writerow(["population_size", "generations", "crossover_probability",
                        "mutation_probability", "local_search_iters", "objective",
                        "initial_reward", "best_reward", "execution_time", "assignment",
                        "final_energy", "final_lux", "final_weighted_distance"])
        w.writerow([hp["population_size"], hp["generations"], hp["crossover_probability"],
                    hp["mutation_probability"], hp["local_search_iters"], hp["objective"],
                    initial_reward, best_reward, execution_time,
                    json.dumps(assignment), energy, lux, dist])


def main():
    parser = argparse.ArgumentParser(description="Hybrid (memetic) metaheuristic for seating assignment")
    parser.add_argument("--summary_path", type=str, default="result/summary_hybrid.csv")
    parser.add_argument("--benchmark_path", type=str, default=None)
    parser.add_argument("--population_size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--crossover_probability", type=float, default=0.9)
    parser.add_argument("--mutation_probability", type=float, default=0.2)
    parser.add_argument("--local_search_iters", type=int, default=60)
    parser.add_argument("--objective", choices=["scalar", "bench"], default="scalar",
                        help="scalar = -E+L-D (like GA/MC); bench = penalize benchmark-threshold violations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    zones = pp.get_unique_zones()
    users = pp.get_unique_users()
    ev = FastEvaluator()

    hp = {
        "population_size": args.population_size,
        "generations": args.generations,
        "crossover_probability": args.crossover_probability,
        "mutation_probability": args.mutation_probability,
        "local_search_iters": args.local_search_iters,
        "objective": args.objective,
    }

    algo = MemeticSeating(
        users, zones, ev,
        population_size=args.population_size,
        generations=args.generations,
        crossover_probability=args.crossover_probability,
        mutation_probability=args.mutation_probability,
        local_search_iters=args.local_search_iters,
        objective=args.objective,
    )
    best_ass, best_fit, initial_reward, execution_time = algo.run(
        args.seed, benchmark_path=args.benchmark_path, hp=hp)

    energy, lux, dist = ev.evaluate(best_ass)
    write_summary(hp, args.seed, initial_reward, best_fit, execution_time,
                  best_ass, energy, lux, dist, args.summary_path)
    print(f"[HYBRID] seed={args.seed} best_reward={best_fit:.4f} "
          f"E={energy:.3f} L={lux:.3f} D={dist:.4f} time={execution_time:.1f}s "
          f"benchmark={'yes' if algo._benchmark_hit else 'no'}")


if __name__ == "__main__":
    main()
