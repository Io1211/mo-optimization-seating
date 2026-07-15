import os
import sys

if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import os
import random
import time
from pyeasyga.pyeasyga import GeneticAlgorithm
import common.Preprocessing as pp
from common.reward import total_reward, check_assignment_duplicates, reward_function, benchmark_reached

class RandomKeyGA:
    def __init__(self, population_size, generations,
                 crossover_probability, mutation_probability):
        self.users = []
        self.zones = []
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.ga = GeneticAlgorithm(seed_data=[])
        self._benchmark_hit = False
        self._start_time = 0.0

    def preprocess_data(self):
        self.zones = pp.get_unique_zones()
        self.users = pp.get_unique_users()
        return self.zones, self.users

    def create_individual(self, seed_data):
        num_keys = len(self.users)
        individual = [random.random() for _ in range(num_keys)]
        return individual

    def decode_individual(self, individual):
        """
        Decode a random-key vector into an assignment dict {user: zone}.
        The sorted order of the keys gives the permutation.
        """
        sorted_indices = sorted(range(len(individual)), key=lambda i: individual[i])
        assignment = {}
        for i in range(len(self.zones)):
            idx = sorted_indices[i]
            assignment[self.users[idx]] = self.zones[i]
        return assignment

    def crossover(self, parent_1, parent_2):
        child1, child2 = [], []
        for gene1, gene2 in zip(parent_1, parent_2):
            if random.random() < 0.5:
                child1.append(gene1)
                child2.append(gene2)
            else:
                child1.append(gene2)
                child2.append(gene1)
        return child1, child2

    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_probability:
                individual[i] = random.random()
        return individual

    def fitness(self, individual, data):
        assignment = self.decode_individual(individual)
        if check_assignment_duplicates(assignment):
            return -1e10
        return reward_function(assignment)[0]

    def run(self, csv_path=None, benchmark_path=None):
        self.ga.create_individual = self.create_individual
        self.ga.crossover_function = self.crossover
        self.ga.mutate_function = self.mutate
        self.ga.fitness_function = self.fitness
        self.ga.population_size = self.population_size
        self.ga.generations = self.generations
        self.ga.crossover_probability = self.crossover_probability
        self.ga.mutate_probability = self.mutation_probability
        self._start_time = time.time()
        self.ga.create_initial_population()
        self.ga.calculate_population_fitness()
        self.ga.rank_population()
        initial_individual = self.ga.best_individual()[1]
        initial_reward = self.fitness(initial_individual, None)
        print("Initial reward:", initial_reward)

        for gen in range(1, self.generations):
            self.ga.create_next_generation()
            self.ga.calculate_population_fitness()
            self.ga.rank_population()
            best_reward = self.ga.best_individual()[0]
            best_ass = self.decode_individual(self.ga.best_individual()[1])
            best_en, best_lux, best_dist = total_reward(best_ass)
            if csv_path:
                write_run_csv(best_ass, best_reward, best_en, best_lux, best_dist, csv_path)
            if benchmark_path and not self._benchmark_hit and benchmark_reached(best_en, best_lux, best_dist):
                elapsed = time.time() - self._start_time
                write_benchmark_results(
                    pop_size=self.population_size,
                    n_gens=gen + 1,
                    crossover_prob=self.crossover_probability,
                    mutation_prob=self.mutation_probability,
                    initial_reward=initial_reward,
                    best_reward=best_reward,
                    execution_time=elapsed,
                    assignment=best_ass,
                    energy=best_en,
                    lux=best_lux,
                    dist=best_dist,
                    filename=benchmark_path,
                )
                self._benchmark_hit = True
        execution_time = time.time() - self._start_time
        best_individual = self.ga.best_individual()[1]
        final_assignment = self.decode_individual(best_individual)
        final_reward = reward_function(final_assignment)[0]
        return final_assignment, final_reward, initial_reward, execution_time


def write_parameter_results(population_size, generations, seed, crossover_probability, mutation_probability,
                            initial_reward, final_reward, execution_time, final_assignment,
                            filename=r"result\summary_ga.csv"):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    energy, lux, dist = total_reward(final_assignment)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(["population_size", "generations", "seed", "crossover_probability", "mutation_probability",
                             "initial_reward", "final_reward", "execution_time",
                             "final_assignment", "final_energy", "final_lux", "final_weighted_distance"])
        writer.writerow([population_size, generations, seed, crossover_probability, mutation_probability,
                         initial_reward, final_reward, execution_time,
                         final_assignment, energy, lux, dist])


def write_run_csv(assignment, reward, energy, lux, dist, filename="history.csv"):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(["assignment", "reward", "energy_consumption", "lux_values", "weighted_distance"])
        writer.writerow([assignment, reward, energy, lux, dist])


def write_benchmark_results(pop_size, n_gens, initial_reward, best_reward, execution_time, assignment, energy, lux, dist,
                            crossover_prob, mutation_prob, filename="result/benchmark_ga.csv"):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if not file_exists:
            w.writerow(["population_size", "n_generations",  "crossover_probability", "mutation_probability", "initial_reward", "best_reward", "execution_time", "assignment",
                        "final_energy", "final_lux", "final_weighted_distance"])
        w.writerow([pop_size, n_gens, crossover_prob, mutation_prob, initial_reward, best_reward, execution_time, assignment, energy, lux, dist])

def run_normal(args):
    ga = RandomKeyGA(
        population_size=args.population_size,
        generations=args.generations,
        crossover_probability=args.crossover_probability,
        mutation_probability=args.mutation_probability,
    )
    ga.preprocess_data()
    final_assignment, final_reward, initial_reward, execution_time = ga.run(
        csv_path=args.csv_path,
        benchmark_path=args.benchmark_path,
    )
    write_parameter_results(
        args.population_size,
        args.generations,
        args.seed,
        args.crossover_probability,
        args.mutation_probability,
        initial_reward,
        final_reward,
        execution_time,
        final_assignment,
        args.summary_path,
    )
    print(f"Best reward: {final_reward}")


def main():
    parser = argparse.ArgumentParser(description="Run the Random-Key GA optimization.")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file for logging results.", default=None)
    parser.add_argument("--summary_path", type=str, help="Path to the CSV file for logging summary.",
                        default=r"result\summary_ga.csv")
    parser.add_argument("--benchmark_path", type=str, help="Path to the benchmark CSV file.", default=None)
    parser.add_argument("--population_size", type=int, default=100,
                        help="Population size for the genetic algorithm")
    parser.add_argument("--generations", type=int, default=1000,
                        help="Number of generations for the genetic algorithm")
    parser.add_argument("--crossover_probability", type=float, default=0.8,
                        help="Crossover probability")
    parser.add_argument("--mutation_probability", type=float, default=0.4,
                        help="Mutation probability")
    parser.add_argument("--seed", type=int, help="Random seed.", default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    run_normal(args)


if __name__ == "__main__":
    main()
