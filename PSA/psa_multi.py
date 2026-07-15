# file: psa_multi.py
import os, csv, json, time, sys
import numpy as np
from typing import List, Tuple, Dict

if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import common.Preprocessing as pp
from common.reward import total_reward, benchmark_reached  # -> (energy, lux, distance)

def perm_from_random_keys(x: np.ndarray) -> np.ndarray:
    return np.argsort(x)

def decode(users: List[str], zones: List[str], x: np.ndarray) -> Dict[str, str]:
    p = perm_from_random_keys(x)
    return { users[i]: zones[j] for j, i in enumerate(p) }

def objectives(users, zones, x: np.ndarray) -> np.ndarray:
    en, lux, dist = total_reward(decode(users, zones, x))
    # We **minimize** all three by flipping lux:
    return np.array([en, dist, -lux], dtype=float)  # f = [E, D, -Lux]

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a strictly dominates b (all <= and at least one <)."""
    return np.all(a <= b) and np.any(a < b)

def pareto_update(archive: List[Tuple[np.ndarray, np.ndarray]], cand: Tuple[np.ndarray, np.ndarray]):
    """Insert cand=(f,x) if nondominated; remove any dominated members."""
    f_new, x_new = cand
    # discard if dominated by existing
    for f_old, _ in archive:
        if dominates(f_old, f_new):
            return False
    # remove members dominated by cand
    keep = []
    for f_old, x_old in archive:
        if not dominates(f_new, f_old):
            keep.append((f_old, x_old))
    keep.append((f_new, x_new))
    archive[:] = keep
    return True

def estimate_scales(users, zones, n=256, seed=123):
    """Estimate per-objective scale (ideal/nadir) for normalization."""
    rng = np.random.default_rng(seed)
    xs = rng.random((n, len(users)))
    F = np.array([objectives(users, zones, x) for x in xs])
    z_min = F.min(axis=0)
    z_max = F.max(axis=0)
    scale = np.where(z_max > z_min, z_max - z_min, 1.0)
    return z_min, z_max, scale

def norm_delta(f_new, f_old, z_min, scale):
    """Sum of positive normalized regressions (worse objectives only)."""
    # smaller is better, so regression where f_new > f_old
    diff = (f_new - f_old)
    worse = np.maximum(diff, 0.0) / scale
    # tiny epsilon to avoid zero-energy traps
    return float(np.sum(worse) + 1e-12)

# ------------------ Multi-objective PSA kernel ------------------
def psa_multi(users, zones, steps=4000, T0=1.0, Tend=1e-3, sigma=0.05, seed=0, keep_every=25):
    """
    Simple PSA:
      - state: random-keys vector in [0,1]^n
      - objectives: f=[E, D, -Lux] (all minimized)
      - archive: nondominated set
      - acceptance:
          * if f' dominates f  -> accept
          * elif f dominates f'-> accept with prob exp(-Δ/T) where Δ is normalized regression
          * else (mutual nondom)-> accept with prob exp(-Δc/T) vs current, and add to archive
    """
    rng = np.random.default_rng(seed)
    n = len(users)

    # init
    x = rng.random(n)
    f = objectives(users, zones, x)

    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    # track benchmark hits similar to MOSA: record first time a benchmark is reached
    bench_hits: List[Dict] = []
    bench_reached = False

    inserted = pareto_update(archive, (f.copy(), x.copy()))
    if inserted and (not bench_reached):
        # f = [E, D, -Lux]
        en, dist, neg_lux = f.tolist()
        lux = -neg_lux
        if benchmark_reached(en, lux, dist):
            bench_hits.append({"x": x.copy(), "en": float(en), "lux": float(lux), "dist": float(dist), "eval_idx": 0, "time": time.time()})
            bench_reached = True

    # normalization (cheap and stable)
    z_min, z_max, scale = estimate_scales(users, zones, n=256, seed=seed+7)

    # cooling schedule (geometric)
    def temp(t):
        return max(Tend, T0 * (Tend / T0) ** (t / steps))

    accept_cnt = 0
    for t in range(1, steps + 1):
        T = temp(t)

        # neighbor: perturb a small subset
        k = max(1, n // 10)
        idx = rng.choice(n, size=k, replace=False)
        x_new = x.copy()
        x_new[idx] = np.clip(x_new[idx] + rng.normal(0, sigma, size=k), 0.0, 1.0)

        f_new = objectives(users, zones, x_new)

        if dominates(f_new, f):
            # strictly better → accept
            x, f = x_new, f_new
            accept_cnt += 1
            inserted = pareto_update(archive, (f.copy(), x.copy()))
            if inserted and (not bench_reached):
                en, dist, neg_lux = f.tolist()
                lux = -neg_lux
                if benchmark_reached(en, lux, dist):
                    bench_hits.append({"x": x.copy(), "en": float(en), "lux": float(lux), "dist": float(dist), "eval_idx": t, "time": time.time()})
                    bench_reached = True
        elif dominates(f, f_new):
            # strictly worse → Metropolis on regression
            dE = norm_delta(f_new, f, z_min, scale)
            if rng.random() < np.exp(-dE / T):
                x, f = x_new, f_new
                accept_cnt += 1
                # usually worse than current, but may still be nondominated vs archive
                inserted = pareto_update(archive, (f.copy(), x.copy()))
                if inserted and (not bench_reached):
                    en, dist, neg_lux = f.tolist()
                    lux = -neg_lux
                    if benchmark_reached(en, lux, dist):
                        bench_hits.append({"x": x.copy(), "en": float(en), "lux": float(lux), "dist": float(dist), "eval_idx": t, "time": time.time()})
                        bench_reached = True
        else:
            # mutually nondominated → encourage diversity; soft accept
            # use regression vs current to keep some pressure
            dE = norm_delta(f_new, f, z_min, scale)
            if rng.random() < np.exp(-dE / T):
                x, f = x_new, f_new
                accept_cnt += 1
            # update archive regardless (it’s nondominated wrt current)
            inserted = pareto_update(archive, (f_new.copy(), x_new.copy()))
            if inserted and (not bench_reached):
                en, dist, neg_lux = f_new.tolist()
                lux = -neg_lux
                if benchmark_reached(en, lux, dist):
                    bench_hits.append({"x": x_new.copy(), "en": float(en), "lux": float(lux), "dist": float(dist), "eval_idx": t, "time": time.time()})
                    bench_reached = True

        # optional: refresh scales sparsely for stability
        if (t % 500) == 0:
            z_min, z_max, scale = estimate_scales(users, zones, n=128, seed=seed+9)

    return archive, accept_cnt, bench_hits

def save_archive_csv(users, zones, archive, path="result/psa_multi_summary.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f, delimiter=';')
        if not file_exists:
            w.writerow(["energy","lux","distance","perm","assignment_json"])
        for fvec, x in archive:
            energy, distance, neg_lux = fvec.tolist()
            lux = -neg_lux
            p = perm_from_random_keys(x)
            # compact permutation as zone order (string)
            perm_zones = [str(zones[j]) for j in range(len(p))]
            # keep JSON small; omit if you prefer
            assignment = decode(users, zones, x)
            w.writerow([
                f"{energy:.6g}",
                f"{lux:.6g}",
                f"{distance:.6g}",
                ",".join(perm_zones),
                json.dumps(assignment, ensure_ascii=False)
            ])

def write_summary(steps, seed, solution_index, execution_time, best_energy, best_lux, best_dist, assignment,
                  filename=os.path.join("results", "summary_psa_multi.csv")):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        w = csv.writer(f, delimiter=';')
        if not file_exists:
            w.writerow(["steps", "seed", "solution_index", "execution_time", "final_energy",
                        "final_lux", "final_weighted_distance", "assignment"])
        w.writerow([steps, seed, solution_index, execution_time, best_energy, best_lux, best_dist,
                    json.dumps(assignment)])

def write_benchmark(steps, seed, solution_index, execution_time, best_energy, best_lux, best_dist, assignment,
                    filename=os.path.join("results", "benchmark_psa_multi_hpo.csv")):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        w = csv.writer(f, delimiter=';')
        if not file_exists:
            w.writerow(["steps", "seed", "solution_index", "execution_time", "final_energy",
                        "final_lux", "final_weighted_distance", "assignment"])
        w.writerow([steps, seed, solution_index, execution_time, best_energy, best_lux, best_dist,
                    json.dumps(assignment)])

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run Bayesian Optimization.")
    # logging / run control
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of optimization steps.")
    args = parser.parse_args()

    users = pp.get_unique_users()
    zones  = pp.get_unique_zones()
    steps = args.steps
    seed = args.seed
    t0 = time.time()
    archive, acc, bench_hits = psa_multi(users, zones, steps=steps, T0=1.0, Tend=1e-3, sigma=0.06, seed=seed)
    dt = time.time() - t0
    print(f"Archive size: {len(archive)} | accepted moves: {acc} | time: {dt:.2f}s")
    #save_archive_csv(users, zones, archive, "result/psa_multi_summary_HPO.csv")
    # write mosa-like summary per archive entry
    for idx, (fvec, x) in enumerate(archive):
        energy, distance, neg_lux = fvec.tolist()
        lux = -neg_lux
        assignment = decode(users, zones, x)
        write_summary(steps, seed, idx, dt, energy, lux, distance, assignment)

    # write benchmark hits recorded during the run (if any)
    for bidx, hit in enumerate(bench_hits):
        assignment = decode(users, zones, hit["x"]) if isinstance(hit.get("x"), (list, np.ndarray)) else hit.get("x")
        exec_time = hit.get("time", t0) - t0
        write_benchmark(steps, seed, bidx, exec_time, hit["en"], hit["lux"], hit["dist"], assignment)
