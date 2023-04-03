from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import numpy as np


def score(weights, solution, c, values):
    x = np.sum(np.multiply(solution,weights))
    if x > c:
        p = 0
    if x < c or x == c:
        p = np.sum(np.multiply(solution,values))
    return p

def antColonyOptimization(seconds, weights, num_ants, pheromone_evaporation, constraint, values):
    best_solution = None
    new_best_solution = None
    #best_score = float("inf")
    best_score = 0
    best_scores = []
    pheromones = [[1 for _ in range(options)] for _ in range(len(weights))]
    ants = [[None for _ in range(len(weights))] for _ in range(num_ants)]
    scores = [0 for _ in range(num_ants)]

    end = datetime.today() + timedelta(seconds=seconds)
    while datetime.today() < end:
        for i in range(len(ants)):
            for op in range(len(weights)):
                ants[i][op] = move_ant(pheromones[op])
            scores[i] = score(weights, ants[i], constraint, values) 
        pheromones = updatePheromones(pheromones, ants, pheromone_evaporation)
        new_best_score = max(scores)
        max_index = scores.index(new_best_score)
        new_best_solution = ants[max_index]
        if (new_best_score > best_score):
            best_solution = new_best_solution.copy()
            best_score = new_best_score
        best_scores.append(best_score)
    return best_solution, best_scores


def updatePheromones(pheromones, solutions, pheromone_evaporation):
    ants_pheromone = [[0 for _ in range(len(pheromones[0]))]
                      for _ in range(len(pheromones))]
    for solution in range(len(solutions)):
        for op in range(len(solutions[solution])):
            ants_pheromone[op][solutions[solution][op]] += 1
    for i in range(len(pheromones)):
        for j in range(len(pheromones[i])):
            pheromones[i][j] = (1 - pheromone_evaporation) * \
                pheromones[i][j] + ants_pheromone[i][j]
    return pheromones


def move_ant(pheromones):
    weights = [(1 * pheromones[i]) / sum([1 * pheromones[x]
                                          for x in range(options)]) for i in range(options)]
    choice = random.choices(
        [i for i in range(options)], weights=weights, k=1)
    return choice[0]


options = 2
num_items = 1000

MIN_WEIGHT = 1
MAX_WEIGHT = 100

MIN_VALUE = 10
MAX_VALUE = 50

SECONDS = 10

weights = [random.randint(MIN_WEIGHT, MAX_WEIGHT)
        for _ in range(num_items)]

values = [random.randint(MIN_VALUE, MAX_VALUE)
        for _ in range(num_items)]

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

    #plt.plot([SECONDS * i / len(average_best_score)
    #          for i in range(len(average_best_score))], average_best_score, label=label)
    
    plt.plot([SECONDS * i / len(best_score)
              for i in range(len(best_score))], best_score, label=label)

n_iterations = 1

# Varying number of ants

#for num_ants in [1,5,10,15,20]:
#    print('Number of ants ', num_ants)
#    plot_average(n_iterations, "No. Ants " + str(num_ants),
#                 antColonyOptimization, SECONDS, weights, num_ants, 0.01, constr, values)

#constr = np.sum(weights)//2
#print('Maximum weight ', constr)

# Varying constraint

somma =[np.sum(weights) for _ in range(5)]
divisors = [1.5, 1.75, 2, 2.25, 2.5]
lista = np.round(np.divide(somma, divisors))
num_ants = 20

for c in lista:
    print('Maximum weight ', c)
    plot_average(n_iterations, str(c),
                 antColonyOptimization, SECONDS, weights, num_ants, 0.01, c, values)
    
plt.xlabel("Algorithm Execution Time (s)")
plt.ylabel("Optimized total value")
plt.legend(title='Max weight')
plt.show()
