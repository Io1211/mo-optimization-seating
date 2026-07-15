import os
import sys

if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import csv
from skopt import gp_minimize
from skopt.space import Real
import common.Preprocessing as pp
from common.reward import reward_function, total_reward, benchmark_reached


class BayesianOptimizer:
    def __init__(self, csv_path, n_random_starts,benchmark_path=None):
        self.csv_path = csv_path
        self.zones = pp.get_unique_zones()
        self.users = pp.get_unique_users()
        self.n_random_starts = n_random_starts
        if len(self.users) != len(self.zones):
            raise ValueError("Users and zones counts must match.")
        self.zones_sorted = sorted(self.zones)
        # Build a search space in [0,1]^n_users
        self.space = [Real(0.0, 1.0, name=u) for u in self.users]
        self.historic_metrics = []
        self.eval_count = 0
        self._start_time = 0.0
        self.benchmark_hit = False
        self.benchmark_path = benchmark_path

    def _decode_to_assignment(self, vector):
        """
        Decode a continuous vector x into a valid permutation assignment.
        """
        # get indices of x in ascending order
        sorted_indices = sorted(range(len(vector)), key=lambda i: vector[i])
        assignment = {}
        for rank, user_idx in enumerate(sorted_indices):
            user = self.users[user_idx]
            zone = self.zones_sorted[rank]
            assignment[user] = zone
        return assignment

    def _objective(self, vector):
        self.eval_count += 1
        assignment = self._decode_to_assignment(vector)
        # compute negative reward because minimize
        r, total_energy, avg_lux_value, total_weighted_distance, benchmark = reward_function(
            assignment
        )
        metrics = (total_energy, avg_lux_value, total_weighted_distance)

        if benchmark and not self.benchmark_hit and self.benchmark_path:
            print(f"Benchmark reached with assignment: {assignment} and reward: {r}")
            elapsed = time.time() - self._start_time
            write_benchmark_results(
                n_evals=self.eval_count,
                n_random_starts=self.n_random_starts,
                initial_reward=self.historic_metrics[0][3] if self.historic_metrics else r,
                best_reward=r,
                execution_time=elapsed,
                assignment=assignment,
                energy=total_energy,
                lux=avg_lux_value,
                dist=total_weighted_distance,
                filename=self.benchmark_path

            )
            self.benchmark_hit = True

        if self.csv_path:
            write_run_csv(assignment, r, metrics[0], metrics[1], metrics[2], self.csv_path)
        self.historic_metrics.append((metrics[0], metrics[1], metrics[2], r))
        return -r

    def run(self, n_calls=200, n_random_starts=20, random_state=42,
            summary_path=None, checkpoint_every=50):
        """
        Run Bayesian optimization with gp_minimize over the continuous space.
        n_calls: total function evaluations.
        n_random_starts: initial random points.

        Returns best assignment dict and best reward (positive).
        """
        self._start_time = time.time()

        def _checkpoint(res):
            calls_done = len(res.func_vals)
            if summary_path and calls_done % checkpoint_every == 0:
                best_assignment = self._decode_to_assignment(res.x)
                best_reward = -res.fun
                init = self.historic_metrics[0][3] if self.historic_metrics else best_reward
                write_parameter_results(
                    calls_done, n_random_starts, random_state, init, best_reward,
                    time.time() - self._start_time, best_assignment, summary_path,
                )

        result = gp_minimize(
            func=self._objective,
            dimensions=self.space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            acq_func="EI",
            random_state=random_state,
            callback=_checkpoint,
        )

        execution_time = time.time() - self._start_time
        best_x = result.x
        best_assignment = self._decode_to_assignment(best_x)
        best_reward = -result.fun
        return best_assignment, best_reward, execution_time, self.historic_metrics


def write_run_csv(assignment, reward, energy, lux, dist, filename="history.csv"):
    """
    Write the assignment and reward to a CSV file.
    :param dist:
    :param lux:
    :param energy:
    :param assignment: The assignment dict {user: zone}
    :param reward: The reward value
    :param filename: The name of the CSV file
    """
    # create directory if not exists
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(["assignment", "reward", "energy_consumption", "lux_values", "weighted_distance"])
        writer.writerow([assignment, reward, energy, lux, dist])


def write_parameter_results(n_calls, n_random_starts, seed, initial_reward, final_reward, execution_time,
                            final_assignment, filename=r"result\summary_results_parameter.csv"):
    """
    Write the summary of the optimization run to a CSV file.
    :param seed: The random seed used
    :param n_random_starts: The number of random starts
    :param n_calls: The number of calls
    :param initial_reward: The initial reward value
    :param final_reward: The final reward value
    :param execution_time: The total execution time
    :param final_assignment: The final assignment dict {user: zone}
    :param filename: The name of the CSV file
    """
    # create directory if not exists
    energy, lux, dist = total_reward(final_assignment)
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(["n_calls", "n_random_starts", "seed", "initial_reward", "final_reward", "execution_time",
                             "final_assignment", "final_energy", "final_lux", "final_weighted_distance"])
        writer.writerow(
            [n_calls, n_random_starts, seed, initial_reward, final_reward, execution_time, final_assignment, energy, lux, dist])

def write_benchmark_results(n_evals, n_random_starts, initial_reward, best_reward, execution_time, assignment, energy, lux, dist,
                            filename="result/benchmark_bo.csv"):
    """Log the benchmark hit event (similar to GA)."""
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if not file_exists:
            w.writerow(["n_evaluations", "n_random_starts", "initial_reward", "best_reward", "execution_time", "assignment", "final_energy", "final_lux", "final_weighted_distance"])
        w.writerow([n_evals, n_random_starts, initial_reward, best_reward, execution_time, assignment, energy, lux, dist])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Bayesian Optimization.")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file for logging results.", default=None)
    parser.add_argument("--summary_path", type=str, help="Path to the CSV file for logging results.",
                        default=r"result\summary.csv")
    parser.add_argument("--benchmark_path", type=str, help="Path to the benchmark CSV file.", default=None)
    parser.add_argument("--n_calls", type=int, help="Number of function evaluations.")
    parser.add_argument("--n_random_starts", type=int, help="Number of initial random points.")
    parser.add_argument("--seed", type=int, help="Random seed.", default=42)
    args = parser.parse_args()

    bo = BayesianOptimizer(args.csv_path, args.n_random_starts, args.benchmark_path)
    best_assign, best_r, exe_time, metric_hist = bo.run(n_calls=args.n_calls, n_random_starts=args.n_random_starts,
                                                        random_state=args.seed,
                                                        summary_path=args.summary_path)
    write_parameter_results(
        args.n_calls,
        args.n_random_starts,
        args.seed,
        metric_hist[0][3],
        best_r,
        exe_time,
        best_assign,
        args.summary_path,
    )

# Source: https://scikit-optimize.github.io/dev/auto_examples/bayesian-optimization.html#sphx-glr-auto-examples-bayesian-optimization-py
