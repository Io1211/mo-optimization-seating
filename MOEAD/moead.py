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

from pymoo.algorithms.moo.moead import MOEAD
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

def run_moead_and_log(problem, n_neighbors, prob_neighbor_mating, n_gen, seed, summary_path, benchmark_path=None):
    # Reference directions for 3 objectives

    ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)  # "das-dennis"

    # determine population size from reference directions
    pop_size_from_dirs = len(ref_dirs)

    # sanitize/derive n_neighbors: default to 10% of pop, clipped between 2 and pop_size-1
    if n_neighbors is None:
        n_neighbors = min(max(10, int(0.1 * pop_size_from_dirs)), max(2, pop_size_from_dirs - 1))
    else:
        # clip to valid range
        n_neighbors = int(n_neighbors)
        n_neighbors = min(max(2, n_neighbors), max(2, pop_size_from_dirs - 1))

    print(f"MOEAD: pop_size_from_dirs={pop_size_from_dirs}, using n_neighbors={n_neighbors}, prob_neighbor_mating={prob_neighbor_mating}")

    algorithm = MOEAD(
        ref_dirs,
        n_neighbors=n_neighbors,
        prob_neighbor_mating=prob_neighbor_mating,
    )

    start_time = time.time()
    res = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, verbose=False)
    elapsed = time.time() - start_time

    for idx, (x_vec, f_vec) in enumerate(zip(np.atleast_2d(res.X), np.atleast_2d(res.F))):
        energy, distance, neg_lux = f_vec.tolist()
        lux = -neg_lux
        assignment = problem._decode(x_vec)
        # write_summary expects: n_neighbors, prob_neighbor_mating, generations, seed, solution_index, execution_time, best_energy, best_lux, best_dist, assignment, filename
        write_summary(n_neighbors, prob_neighbor_mating, n_gen, seed, idx, elapsed, energy, lux, distance, assignment, summary_path)

    if benchmark_path and problem.bench_reached:
        for idx, hit in enumerate(problem.bench_hits):
            assignment = problem._decode(hit["x"])
            # write_benchmark expects: n_neighbors, prob_neighbor_mating, generations, seed, solution_index, execution_time, best_energy, best_lux, best_dist, assignment, filename
            write_benchmark(
                n_neighbors, prob_neighbor_mating, n_gen, seed, idx,
                hit["time"] - start_time, hit["en"], hit["lux"], hit["dist"],
                assignment, benchmark_path
            )

def write_summary(n_neighbors, prob_neighbor_mating, generations, seed, solution_index, execution_time, best_energy, best_lux, best_dist, assignment,
                  filename=r"result\summary_moead.csv"):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        w = csv.writer(f, delimiter=';')
        if not file_exists:
            w.writerow(["n_neighbors", "prob_neighbor_mating", "generations", "seed", "solution_index", "execution_time", "final_energy",
                        "final_lux", "final_weighted_distance", "assignment"])
        w.writerow(
            [n_neighbors, prob_neighbor_mating, generations, seed, solution_index, execution_time, best_energy, best_lux, best_dist,
             json.dumps(assignment)])

def write_benchmark(n_neighbors, prob_neighbor_mating, generations, seed, solution_index, execution_time, best_energy, best_lux, best_dist,
                    assignment, filename=r"result\benchmark_moead.csv"):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        w = csv.writer(f, delimiter=';')
        if not file_exists:
            w.writerow(["n_neighbors", "prob_neighbor_mating", "generations", "seed", "solution_index", "execution_time", "final_energy",
                        "final_lux", "final_weighted_distance", "assignment"])
        w.writerow(
            [n_neighbors, prob_neighbor_mating, generations, seed, solution_index, execution_time, best_energy, best_lux, best_dist,
             json.dumps(assignment)])
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MOEAD grid search")
    parser.add_argument("--summary_path", type=str,
                        default=r"result\summary_moead_results.csv")
    parser.add_argument("--benchmark_path", type=str, default=r"result\benchmark_moead_results.csv")
    parser.add_argument("--n_neighbors", type=int, default=5)
    parser.add_argument("--prob_neighbor_mating", type=float, default=0.7)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    zones = pp.get_unique_zones()
    users = pp.get_unique_users()
    problem = SeatingProblem(users, zones)
    run_moead_and_log(
        problem,
        args.n_neighbors,
        args.prob_neighbor_mating,
        args.generations,
        args.seed,
        args.summary_path,
        args.benchmark_path
    )
