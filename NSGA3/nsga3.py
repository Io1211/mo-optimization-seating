import argparse
import csv
import json
import os
import sys
import time
import numpy as np

if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
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
def run_nsga3_and_log(problem, pop_size, n_gen, seed, summary_path, benchmark_path=None):
    # Reference directions for 3 objectives
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)

    start_time = time.time()
    res = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, verbose=False)
    elapsed = time.time() - start_time

    for idx, (x_vec, f_vec) in enumerate(zip(np.atleast_2d(res.X), np.atleast_2d(res.F))):
        energy, distance, neg_lux = f_vec.tolist()
        lux = -neg_lux
        assignment = problem._decode(x_vec)
        write_summary(pop_size, n_gen, seed, idx, elapsed, energy, lux, distance, assignment, summary_path)

    if benchmark_path and problem.bench_reached:
        for idx, hit in enumerate(problem.bench_hits):
            assignment = problem._decode(hit["x"])
            write_benchmark(
                pop_size, n_gen, seed, idx,
                hit["time"] - start_time, hit["en"], hit["lux"], hit["dist"],
                assignment, benchmark_path
            )

def write_summary(pop_size, generations, seed, solution_index, execution_time, best_energy, best_lux, best_dist, assignment,
                  filename=r"result\summary_nsga3.csv"):
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

def write_benchmark(pop_size, generations, seed, solution_index, execution_time, best_energy, best_lux, best_dist,
                    assignment, filename=r"result\benchmark_nsga3.csv"):
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
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NSGA-III grid search")
    parser.add_argument("--summary_path", type=str,
                        default=r"result\summary_nsga3_results.csv")
    parser.add_argument("--benchmark_path", type=str, default=None)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    zones = pp.get_unique_zones()
    users = pp.get_unique_users()
    problem = SeatingProblem(users, zones)

    print(f"Running NSGA-III: pop_size={args.population_size}, generations={args.generations}")
    run_nsga3_and_log(
        problem,
        args.population_size,
        args.generations,
        args.seed,
        args.summary_path,
        args.benchmark_path
    )
