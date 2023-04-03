from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt


def score(jobs, solution):
    machines_usage = [0 for _ in range(NUM_MACHINES)]
    for i in range(len(solution)):
        machines_usage[solution[i]] += jobs[i]
    if any(availability < 0 for availability in machines_usage):
        return -1
    return max(machines_usage)


def bestSolution(solutions, scores):
    best_score = float("inf")
    for i in range(len(scores)):
        if (scores[i] < best_score):
            best_score = scores[i]
            best_solution = solutions[i]

    return best_solution, best_score


def antColonyOptimization(seconds, jobs, num_ants, pheromone_evaporation):
    best_solution = None
    best_score = float("inf")
    best_scores = []
    pheromones = [[1 for _ in range(NUM_MACHINES)] for _ in range(len(jobs))]
    ants = [[None for _ in range(len(jobs))] for _ in range(num_ants)]
    scores = [0 for _ in range(num_ants)]

    end = datetime.today() + timedelta(seconds=seconds)
    while datetime.today() < end:
        for i in range(len(ants)):
            for machine in range(len(jobs)):
                ants[i][machine] = move_ant(pheromones[machine])
            scores[i] = score(jobs, ants[i])
        new_best_solution, new_best_score = bestSolution(ants, scores)
        pheromones = updatePheromones(pheromones, ants, pheromone_evaporation)
        if (new_best_score < best_score):
            best_solution = new_best_solution.copy()
            best_score = new_best_score
        best_scores.append(best_score)
    return best_solution, best_scores


def updatePheromones(pheromones, solutions, pheromone_evaporation):
    ants_pheromone = [[0 for _ in range(len(pheromones[0]))]
                      for _ in range(len(pheromones))]
    for solution in range(len(solutions)):
        for machine in range(len(solutions[solution])):
            ants_pheromone[machine][solutions[solution][machine]] += 1
    for i in range(len(pheromones)):
        for j in range(len(pheromones[i])):
            pheromones[i][j] = (1 - pheromone_evaporation) * \
                pheromones[i][j] + ants_pheromone[i][j]
    return pheromones


def move_ant(pheromones):
    weights = [(1 * pheromones[i]) / sum([1 * pheromones[x]
                                          for x in range(NUM_MACHINES)]) for i in range(NUM_MACHINES)]
    choice = random.choices(
        [i for i in range(NUM_MACHINES)], weights=weights, k=1)
    return choice[0]


NUM_MACHINES = 10
NUM_JOBS = 100
MIN_JOB_DURATION = 1
MAX_JOB_DURATION = 10
SECONDS = 10

jobs = [random.randint(MIN_JOB_DURATION, MAX_JOB_DURATION)
        for _ in range(NUM_JOBS)]
jobs = [5, 5, 6, 3, 8, 3, 1, 7, 4, 1, 4, 7, 5, 3, 10, 6, 8, 10, 7, 9, 7, 9, 2, 2, 2, 2, 3, 10, 9, 7, 8, 2, 8, 2, 4, 3, 1, 8, 10, 1, 3, 1, 3, 1, 2, 10, 1, 5, 5, 3, 2, 8, 7, 7, 2, 7, 10, 1, 7, 3, 10, 2, 7, 10, 7, 3, 1, 1, 8, 8, 6, 2, 2, 1, 9, 1, 10, 8, 10, 3, 9, 4, 8, 8, 4, 7, 7, 10, 9, 4, 7, 6, 3, 5, 3, 9, 9, 4, 6, 1]

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
        average_best_score.append(min(sum(average)/len(average), average_best_score[-1] if len(average_best_score) > 0 else float('inf')))

    plt.plot([SECONDS * i / len(average_best_score)
              for i in range(len(average_best_score))], average_best_score, label=label)

# for i in range(5):
#     best_solution, best_scores = antColonyOptimization(SECONDS, jobs)
#     PHEROMONE_EVAPORATION += 0.1
#     plt.plot([SECONDS * j / len(best_scores)
#              for j in range(len(best_scores))], best_scores, label="Pheromone evaporation " + str(PHEROMONE_EVAPORATION))
#     print("Finished iteration ", i)

# plt.xlabel("time")
# plt.ylabel("Minimum Machine Time")
# plt.legend()
# plt.show()


# for pheromone_evaporation in [0.0001, 0.001, 0.01, 0.1, 0.3]:
#     plot_average(5, "Pheromone_evaporation " + str(pheromone_evaporation),
#                  antColonyOptimization, SECONDS, jobs, 10, pheromone_evaporation)

for num_ants in [2, 5, 10, 20, 50]:
    plot_average(5, "No. Ants " + str(num_ants),
                 antColonyOptimization, SECONDS, jobs, num_ants, 0.01)

# for i in range(5):
#     best_solution, best_scores = antColonyOptimization(SECONDS, jobs)
#     plt.plot([SECONDS * j / len(best_scores)
#              for j in range(len(best_scores))], best_scores, label="No. ants " + str(NUM_ANTS))
#     print("Finished iteration ", i)
#     print(best_scores)
#     NUM_ANTS += 5

plt.xlabel("Algorithm Execution Time (s)")
plt.ylabel("Minimum Machine Time")
plt.legend()
plt.show()
