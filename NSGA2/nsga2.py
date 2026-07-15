import argparse
import json
import csv
import os
import sys
import time
import numpy as np

if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import common.Preprocessing as pp
from common.reward import total_reward, benchmark_reached


class SeatingProblem(ElementwiseProblem):
    def __init__(self, users, zones):
        super().__init__(n_var=len(users), n_obj=3, n_ieq_constr=0, xl=0.0, xu=1.0)
        self.users = users
        self.zones = zones
        self._evals = 0
        self.bench_hits = []
        self.bench_reached = False

    def _decode(self, x: np.ndarray):
        order = np.argsort(x)  # random-key decoding
        return {self.users[i]: self.zones[j] for j, i in enumerate(order)}

    def _evaluate(self, x, out, *args, **kwargs):
        self._evals += 1
        assignment = self._decode(x)
        en, lux, dist = total_reward(
            assignment
        )
        benchmark = benchmark_reached(en, lux, dist)
        if benchmark and not self.bench_reached:
            self.bench_hits.append({
                "x": np.array(x, dtype=float).copy(),
                "en": float(en),
                "lux": float(lux),
                "dist": float(dist),
                "eval_idx": int(self._evals),
                "time": time.time()
            })
            self.bench_reached = True

        out["F"] = np.array([en, dist, -lux], dtype=float)
def run_nsga2_and_log(problem, pop_size, n_gen, seed, summary_path, benchmark_path=None):
    algorithm = NSGA2(pop_size=pop_size)
    start_time = time.time()
    res = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, verbose=False)
    elapsed = time.time() - start_time
    # res.X is an array of decision vectors (random-keys)
    # res.F is an array of objective values [energy, distance, -lux]
    for idx, (x_vec, f_vec) in enumerate(zip(np.atleast_2d(res.X), np.atleast_2d(res.F))):
        energy, distance, neg_lux = f_vec.tolist()
        lux = -neg_lux
        assignment = problem._decode(x_vec)
        write_summary(pop_size, n_gen, seed, idx, elapsed, energy, lux, distance, assignment, summary_path)

    if benchmark_path and problem.bench_reached:
        for idx, hit in enumerate(problem.bench_hits):
            assignment = problem._decode(hit["x"])
            write_benchmark(pop_size, n_gen, seed, hit["eval_idx"], hit["time"] - start_time, hit["en"], hit["lux"], hit["dist"],
                            assignment, benchmark_path)


def write_summary(pop_size, generations, seed, solution_index, execution_time, best_energy, best_lux, best_dist, assignment,
                  filename=r"result\summary_nsga2.csv"):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        w = csv.writer(f, delimiter=';')
        if not file_exists:
            w.writerow(["population_size", "generations", "seed", "solution_index", "execution_time", "final_energy",
                        "final_lux", "final_weighted_distance", "assignment"])
        w.writerow(
            [pop_size, generations, seed, solution_index, execution_time, best_energy, best_lux, best_dist,
             json.dumps(assignment)])

def write_benchmark(pop_size, generation, seed, eval_index, execution_time, best_energy, best_lux, best_dist,
                    assignment, filename=r"result\benchmark_nsga2.csv"):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        w = csv.writer(f, delimiter=';')
        if not file_exists:
            w.writerow(["population_size", "generation", "seed", "eval_index", "execution_time", "final_energy",
                        "final_lux", "final_weighted_distance", "assignment"])
        w.writerow(
            [pop_size, generation, seed, eval_index, execution_time, best_energy, best_lux, best_dist,
             json.dumps(assignment)])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NSGA-II grid search")
    parser.add_argument("--summary_path", type=str, help="Path to the CSV file for logging summary.",
                        default=r"result\summary_nsga2_results.csv")
    parser.add_argument("--benchmark_path", type=str, help="Path to the benchmark CSV file.", default=None)
    parser.add_argument("--population_size", type=int, default=100,
                        help="Population size for the algorithm")
    parser.add_argument("--generations", type=int, default=1000,
                        help="Number of generations for the algorithm")
    parser.add_argument("--seed", type=int, default=1, help='Random seed')
    args = parser.parse_args()

    zones = pp.get_unique_zones()
    users = pp.get_unique_users()
    problem = SeatingProblem(users, zones)

    print(f"Running NSGA-II: pop_size={args.population_size}, generations={args.generations}")
    run_nsga2_and_log(problem, args.population_size, args.generations, args.seed, args.summary_path, args.benchmark_path)

