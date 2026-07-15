import csv
import os
import sys
import random, copy, math
import time

if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.reward import total_reward, check_assignment_duplicates, reward_function, benchmark_reached
import common.Preprocessing as pp


class MarkovChainMonteCarlo:
    """Simple Metropolis-style MCMC over permutation assignments using swap proposals.
    Keeps the same external interface as the existing SimulatedAnnealer for ease of swapping.
    """
    def __init__(self, temperature=1.0):
        self.users = []
        self.zones = []
        self.state = {}
        self.temperature = temperature
        self.best_state = {}
        self.best_reward = -float('inf')
        self.historic_metrics = []
        self._benchmark_hit = False
        self._start_time = 0.0

    def preprocess_data(self):
        self.zones = pp.get_unique_zones()
        self.users = pp.get_unique_users()
        if len(self.users) != len(self.zones):
            raise ValueError("The number of users and zones must be equal.")
        self.state = {user: zone for user, zone in zip(self.users, self.zones)}
        self.best_state = copy.deepcopy(self.state)
        self.best_reward = self.reward(self.state)
        return self.zones, self.users

    def reward(self, assignment):
        r, total_energy, avg_lux_value, total_weighted_distance, benchmark = reward_function(assignment)
        return r

    def proposal(self, state):
        # swap two random user zones
        a, b = random.sample(self.users, 2)
        new = state.copy()
        new[a], new[b] = new[b], new[a]
        return new

    def accept_prob(self, r_old, r_new):
        # Metropolis acceptance with temperature
        if r_new > r_old:
            return 1.0
        try:
            return math.exp((r_new - r_old) / max(self.temperature, 1e-12))
        except OverflowError:
            return 0.0

    def step(self):
        cand = self.proposal(self.state)
        if check_assignment_duplicates(cand):
            return
        r_old = self.reward(self.state)
        r_new = self.reward(cand)
        ap = self.accept_prob(r_old, r_new)
        if random.random() < ap:
            self.state = cand
            if r_new > self.best_reward:
                self.best_reward = r_new
                self.best_state = copy.deepcopy(cand)

    def run(self, iterations=1000, csv_path=None, benchmark_path=None):
        self._start_time = time.time()
        for i in range(iterations):
            self.step()
            r, energy, lux, dist, benchmark = reward_function(self.state)
            #if csv_path and i % 10 == 0:
            #write_run_csv(self.state, self.best_reward, energy, lux, dist, csv_path)

            # benchmark logging
            if benchmark_path and not self._benchmark_hit and benchmark:
                elapse = time.time() - self._start_time
                write_benchmark_results(
                    n_steps=i + 1,
                    temperature=self.temperature,
                    initial_reward=self.historic_metrics[0][3] if self.historic_metrics else self.best_reward,
                    best_reward=self.best_reward,
                    assignment=self.best_state,
                    energy=energy,
                    lux=lux,
                    dist=dist,
                    execution_time=elapse,
                    filename=benchmark_path,
                )
                self._benchmark_hit = True
            self.historic_metrics.append((energy, lux, dist, self.best_reward))
        return self.best_state, self.best_reward, self.historic_metrics, time.time() - self._start_time


def write_run_csv(assignment, reward, energy, lux, dist, filename="history_mcmc.csv"):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(["assignment", "reward", "energy_consumption", "lux_values", "weighted_distance"])
        writer.writerow([assignment, reward, energy, lux, dist])


def write_parameter_results(iterations, seed, temperature, initial_reward, final_reward, execution_time,
                            final_assignment, filename=r"result\summary_mcmc.csv"):
    energy, lux, dist = total_reward(final_assignment)
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(["iterations", "seed", "temperature", "initial_reward", "final_reward", "execution_time",
                             "final_assignment", "final_energy", "final_lux", "final_weighted_distance"])
        writer.writerow([iterations, seed, temperature, initial_reward, final_reward, execution_time,
                         final_assignment, energy, lux, dist])


def write_benchmark_results(n_steps, temperature, initial_reward, best_reward, assignment, energy, lux, dist, execution_time, filename="result/benchmark_mcmc.csv"):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if not file_exists:
            w.writerow(["n_steps", "temperature", "initial_reward", "best_reward", "execution_time", "assignment",
                        "final_energy", "final_lux", "final_weighted_distance"])
        w.writerow([n_steps, temperature, initial_reward, best_reward, execution_time, assignment, energy, lux, dist])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run MCMC sampler.")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file for logging results.", default=None)
    parser.add_argument("--summary_path", type=str, help="Path to the CSV file for logging summary.",
                        default=r"result\summary_mcmc_results.csv")
    parser.add_argument("--benchmark_path", type=str, help="Path to the benchmark CSV file.", default=None)
    parser.add_argument("--iterations", type=int, help="Number of MCMC iterations.")
    parser.add_argument("--seed", type=int, help="Random seed.", default=42)
    parser.add_argument("--temperature", type=float, help="Temperature for Metropolis acceptance.", default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)

    mcmc = MarkovChainMonteCarlo(temperature=args.temperature)
    mcmc.preprocess_data()

    _start = time.time()
    best_state, best_reward, metric_hist, execution_time = mcmc.run(iterations=args.iterations, csv_path=args.csv_path,
                                                                    benchmark_path=args.benchmark_path)

    write_parameter_results(
        args.iterations,
        args.seed,
        args.temperature,
        metric_hist[0][3] if metric_hist else best_reward,
        best_reward,
        execution_time,
        best_state,
        args.summary_path,
    )


if __name__ == "__main__":
    main()
