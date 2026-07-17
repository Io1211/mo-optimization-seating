"""
A neural policy network scores every (user, zone) pair from learnable user/zone
embeddings passed through an MLP; the assignment is decoded greedily (masked
argmax over the score matrix). The network weights are trained *without
backprop* using an Evolution Strategy (OpenAI-ES: antithetic Gaussian
perturbations + rank-normalized reward-weighted updates).

Objective: same scalar fitness as the other single-objective methods
(-energy + lux - distance)
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


class DeepPolicy:
    """MLP over user/zone embeddings -> (n_users x n_zones) score matrix."""

    def __init__(self, n, embed_dim, hidden, rng):
        self.n = n
        self.d = embed_dim
        self.h = hidden
        scale = 0.5
        # parameter blocks
        self.shapes = {
            "U": (n, embed_dim),
            "V": (n, embed_dim),
            "W1": (2 * embed_dim, hidden),
            "b1": (hidden,),
            "w2": (hidden,),
            "b2": (1,),
        }
        self.sizes = {k: int(np.prod(s)) for k, s in self.shapes.items()}
        self.dim = sum(self.sizes.values())
        self.theta = rng.normal(0, scale, size=self.dim)

        # precompute all (i,j) index pairs for vectorized scoring
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        self.ii = ii.ravel()
        self.jj = jj.ravel()

    def _unpack(self, theta):
        out, off = {}, 0
        for k, s in self.shapes.items():
            sz = self.sizes[k]
            out[k] = theta[off:off + sz].reshape(s)
            off += sz
        return out

    def score_matrix(self, theta):
        p = self._unpack(theta)
        # pair features: concat(U_i, V_j) for all i,j
        x = np.concatenate([p["U"][self.ii], p["V"][self.jj]], axis=1)  # (n*n, 2d)
        hpre = x @ p["W1"] + p["b1"]
        hact = np.tanh(hpre)
        s = hact @ p["w2"] + p["b2"][0]
        return s.reshape(self.n, self.n)  # rows=users, cols=zones

    def decode(self, theta):
        """Greedy masked assignment: for each zone (col) pick best free user."""
        S = self.score_matrix(theta)
        n = self.n
        assigned_user = -np.ones(n, dtype=int)
        free = np.ones(n, dtype=bool)
        # assign zones in descending order of their best available score
        col_order = np.argsort(-S.max(axis=0))
        for j in col_order:
            col = S[:, j].copy()
            col[~free] = -np.inf
            i = int(np.argmax(col))
            assigned_user[j] = i
            free[i] = False
        # perm[j] = user index for zone j
        return assigned_user


class DeepOptimizer:
    def __init__(self, users, zones, evaluator, embed_dim, hidden,
                 population, iterations, sigma, learning_rate, objective="scalar"):
        self.users = list(users)
        self.zones = list(zones)
        self.n = len(self.users)
        self.ev = evaluator
        self.objective = objective
        self.reward_fn = evaluator.reward_bench if objective == "bench" else evaluator.reward
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.population = population if population % 2 == 0 else population + 1
        self.iterations = iterations
        self.sigma = sigma
        self.learning_rate = learning_rate
        self._start_time = 0.0
        self._benchmark_hit = False

    def _assignment(self, perm):
        return {self.users[perm[j]]: self.zones[j] for j in range(self.n)}

    def _reward(self, perm):
        return self.reward_fn(self._assignment(perm))

    @staticmethod
    def _rank_normalize(rewards):
        # centered ranks in [-0.5, 0.5] -> robust ES updates
        ranks = np.empty_like(rewards)
        ranks[np.argsort(rewards)] = np.arange(len(rewards))
        ranks = ranks / (len(rewards) - 1) - 0.5
        return ranks

    def run(self, seed, benchmark_path=None, hp=None):
        rng = np.random.default_rng(seed)
        policy = DeepPolicy(self.n, self.embed_dim, self.hidden, rng)
        self._start_time = time.time()

        init_perm = policy.decode(policy.theta)
        initial_reward = float(self._reward(init_perm))
        best_reward = initial_reward
        best_perm = init_perm.copy()
        self._maybe_log_benchmark(best_perm, best_reward, initial_reward, benchmark_path, hp)

        half = self.population // 2
        for it in range(self.iterations):
            eps = rng.normal(0, 1, size=(half, policy.dim))
            rewards = np.empty(self.population)
            perms = [None] * self.population
            for k in range(half):
                tp = policy.theta + self.sigma * eps[k]
                tm = policy.theta - self.sigma * eps[k]
                pp_perm = policy.decode(tp)
                pm_perm = policy.decode(tm)
                rewards[2 * k] = self._reward(pp_perm)
                rewards[2 * k + 1] = self._reward(pm_perm)
                perms[2 * k] = pp_perm
                perms[2 * k + 1] = pm_perm

            # track best across this generation
            gbest = int(np.argmax(rewards))
            if rewards[gbest] > best_reward:
                best_reward = float(rewards[gbest])
                best_perm = perms[gbest].copy()
            self._maybe_log_benchmark(best_perm, best_reward, initial_reward, benchmark_path, hp)

            # ES gradient estimate (antithetic + rank shaping)
            shaped = self._rank_normalize(rewards)
            grad = np.zeros(policy.dim)
            for k in range(half):
                grad += (shaped[2 * k] - shaped[2 * k + 1]) * eps[k]
            grad /= (self.population * self.sigma)
            policy.theta = policy.theta + self.learning_rate * grad

        execution_time = time.time() - self._start_time
        best_ass = self._assignment(best_perm)
        return best_ass, best_reward, initial_reward, execution_time

    def _maybe_log_benchmark(self, perm, best_reward, initial_reward, benchmark_path, hp):
        if not benchmark_path or self._benchmark_hit:
            return
        ass = self._assignment(perm)
        en, lux, dist = self.ev.evaluate(ass)
        if self.ev.benchmark(ass):
            write_benchmark(hp, initial_reward, best_reward,
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
            w.writerow(["iterations", "population", "sigma", "learning_rate",
                        "embed_dim", "hidden", "objective", "seed",
                        "initial_reward", "final_reward", "execution_time",
                        "final_assignment", "final_energy", "final_lux",
                        "final_weighted_distance"])
        w.writerow([hp["iterations"], hp["population"], hp["sigma"], hp["learning_rate"],
                    hp["embed_dim"], hp["hidden"], hp["objective"], seed,
                    initial_reward, final_reward, execution_time,
                    json.dumps(assignment), energy, lux, dist])


def write_benchmark(hp, initial_reward, best_reward, execution_time, assignment,
                    energy, lux, dist, filename):
    _ensure_dir(filename)
    exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if not exists:
            w.writerow(["iterations", "population", "sigma", "learning_rate",
                        "embed_dim", "hidden", "objective",
                        "initial_reward", "best_reward", "execution_time", "assignment",
                        "final_energy", "final_lux", "final_weighted_distance"])
        w.writerow([hp["iterations"], hp["population"], hp["sigma"], hp["learning_rate"],
                    hp["embed_dim"], hp["hidden"], hp["objective"],
                    initial_reward, best_reward, execution_time,
                    json.dumps(assignment), energy, lux, dist])


def main():
    parser = argparse.ArgumentParser(description="Deep (neural + evolution-strategy) optimization")
    parser.add_argument("--summary_path", type=str, default="result/summary_deep.csv")
    parser.add_argument("--benchmark_path", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--embed_dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--objective", choices=["scalar", "bench"], default="scalar",
                        help="scalar = -E+L-D (like GA/MC); bench = penalize benchmark-threshold violations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    zones = pp.get_unique_zones()
    users = pp.get_unique_users()
    ev = FastEvaluator()

    hp = {
        "iterations": args.iterations,
        "population": args.population,
        "sigma": args.sigma,
        "learning_rate": args.learning_rate,
        "embed_dim": args.embed_dim,
        "hidden": args.hidden,
        "objective": args.objective,
    }

    algo = DeepOptimizer(users, zones, ev,
                         embed_dim=args.embed_dim, hidden=args.hidden,
                         population=args.population, iterations=args.iterations,
                         sigma=args.sigma, learning_rate=args.learning_rate,
                         objective=args.objective)
    best_ass, best_reward, initial_reward, execution_time = algo.run(
        args.seed, benchmark_path=args.benchmark_path, hp=hp)

    energy, lux, dist = ev.evaluate(best_ass)
    write_summary(hp, args.seed, initial_reward, best_reward, execution_time,
                  best_ass, energy, lux, dist, args.summary_path)
    print(f"[DEEP] seed={args.seed} best_reward={best_reward:.4f} "
          f"E={energy:.3f} L={lux:.3f} D={dist:.4f} time={execution_time:.1f}s "
          f"benchmark={'yes' if algo._benchmark_hit else 'no'}")


if __name__ == "__main__":
    main()
