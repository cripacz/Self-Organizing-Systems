import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def score(jobs, num_machines, solution):
    machines_usage = [0 for _ in range(num_machines)]
    for i in range(len(solution)):
        machines_usage[solution[i]] += jobs[i]
    if any(availability < 0 for availability in machines_usage):
        return -1
    return max(machines_usage)


def mutation(solution, valueRange, numberBits=1):
    positions = random.sample(range(0, len(solution) - 1), numberBits)

    for position in positions:
        possible_values = [x for x in valueRange if x != solution[position]]
        solution[position] = random.choice(possible_values)

    return solution


def mutationGeneticAlgorithm(seconds, num_solutions, jobs, num_machines):
    solutions = [[random.randint(0, num_machines - 1)
                  for _ in range(len(jobs))] for _ in range(num_solutions)]
    best_result = None
    best_score = float("inf")
    best_scores = []

    end = datetime.today() + timedelta(seconds=seconds)
    while datetime.today() < end:
        for i in range(len(solutions)):
            child = mutation(solutions[i], range(0, num_machines), 1)
            solutions.append(child)

        solutions = solutionsSelection(jobs, num_machines, solutions)
        newScore = score(jobs, num_machines, solutions[0])
        if (newScore < best_score):
            best_result = solutions[0].copy()
            best_score = newScore
        best_scores.append(best_score)

    return best_result, best_scores


def crossover(solution1, solution2):
    start = random.randint(0, len(solution1) - 2)
    end = random.randint(start + 1, len(solution1) - 1)
    child1 = solution1[0:start] + solution2[start:end] + solution1[end:]
    child2 = solution2[0:start] + solution1[start:end] + solution2[end:]
    return child1, child2


def solutionsSelection(jobs, num_machines, solutions):
    weights = [(score(jobs, num_machines, solution), solution)
               for solution in solutions]
    weights.sort(key=lambda x: x[0])
    return [weight[1] for weight in weights[0: int(len(weights) / 2)]]
    # random.choices(
    #    [weight[1] for weight in weights],
    #    [weight[0] for weight in weights],
    #    k=int(len(solutions)/2))


def crossoverGeneticAlgorithm(seconds, num_solutions, jobs, num_machines):
    solutions = [[random.randint(0, num_machines - 1)
                  for _ in range(len(jobs))] for _ in range(num_solutions)]
    best_solution = None
    best_score = float("inf")
    best_scores = []

    end = datetime.today() + timedelta(seconds=seconds)
    while datetime.today() < end:
        for _ in range(0, len(solutions), 2):
            child1, child2 = crossover(random.choice(solutions),
                                       random.choice(solutions))
            solutions.append(child1)
            solutions.append(child2)

        solutions = solutionsSelection(jobs, num_machines, solutions)
        new_best_score = score(jobs, num_machines, solutions[0])
        if (new_best_score < best_score):
            best_solution = solutions[0].copy()
            best_score = new_best_score

        best_scores.append(best_score)

    return best_solution, best_scores


def plot_average(num_runs, label, func, *arguments):
    best_scores = []
    for i in range(num_runs):
        _, best_score = func(*arguments)
        best_scores.append(best_score)
        print("Finished iteration ", i+1, "/", num_runs)

    average_best_score = []
    for i in range(len(best_scores[0])):
        average = []
        for j in range(len(best_scores)):
            if len(best_scores[j]) > i:
                average.append(best_scores[j][i])
        average_best_score.append(min(sum(average)/len(average),
                                      average_best_score[-1] if len(average_best_score) > 0 else float('inf')))

    plt.plot([SECONDS * i / len(average_best_score)
              for i in range(len(average_best_score))], average_best_score, label=label)


NUM_JOBS = 100
NUM_MACHINES = 10
MIN_JOB_DURATION = 1
MAX_JOB_DURATION = 10
SECONDS = 10

jobs = [random.randint(MIN_JOB_DURATION, MAX_JOB_DURATION)
        for _ in range(NUM_JOBS)]
jobs = [5, 5, 6, 3, 8, 3, 1, 7, 4, 1, 4, 7, 5, 3, 10, 6, 8, 10, 7, 9, 7, 9, 2, 2, 2, 2, 3, 10, 9, 7, 8, 2, 8, 2, 4, 3, 1, 8, 10, 1, 3, 1, 3, 1, 2, 10, 1, 5, 5, 3, 2, 8, 7, 7, 2, 7, 10, 1, 7, 3, 10, 2, 7, 10, 7, 3, 1, 1, 8, 8, 6, 2, 2, 1, 9, 1, 10, 8, 10, 3, 9, 4, 8, 8, 4, 7, 7, 10, 9, 4, 7, 6, 3, 5, 3, 9, 9, 4, 6, 1]

print("jobs = ", jobs)
print("number of machines = ", NUM_MACHINES)

print("\n\n---------- MUTATION ----------")

plot_average(5, "Mutation", mutationGeneticAlgorithm,
             SECONDS, 20, jobs, NUM_MACHINES)

print("\n\n---------- CROSSOVER ----------")

plot_average(5, "Crossover", crossoverGeneticAlgorithm,
             SECONDS, 20, jobs, NUM_MACHINES)

plt.xlabel("time")
plt.ylabel("Minimum Machine Time")
plt.legend()
plt.show()
