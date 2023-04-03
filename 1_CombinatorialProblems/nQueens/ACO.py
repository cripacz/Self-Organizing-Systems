import random
import numpy as np 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from functools import reduce
import matplotlib.dates
import pandas as pd

generation = 1

pheromone_evaporation = 0.1
alpha = 2
beta = 0.1
list_nb_ants = [5, 10, 20, 50]
list_nb_queens = [4, 5, 6, 7, 8]

def fitness_score(seq) : 
    row_score = sum(seq.count(queen)-1 for queen in seq if queen != 0 and queen != None ) / 2 
    diag_score = 0 
    for j in range(0,len(seq))  :
        for i in range(0, len(seq)) : 
            if (i != j) : 
                x = abs(i-j)
                if (seq[i] != 0 and seq[j] != 0 and seq[i] != None and seq[j] !=None ) : 
                    y = abs(seq[i] - seq[j])
                    if(x==y) : 
                        diag_score = diag_score + 1 
    return row_score + diag_score  

def updatePheromones(pheromones, solutions,  pheromone_evaporation):
    ants_pheromone = [[0 for _ in range(len(pheromones[0]))]
                      for _ in range(len(pheromones))]
    for solution in range(len(solutions)):
        for queen in range(len(pheromones[0])):
            ants_pheromone[queen][solutions[solution][queen]-1] += 1
    for i in range(len(pheromones)):
        for j in range(len(pheromones[i])):
            pheromones[i][j] = (1 -  pheromone_evaporation) * \
                pheromones[i][j] + ants_pheromone[i][j]
    return pheromones

def add_queen(solution, queen, col) : 
    solution[col]=queen
    return solution

def move_ant(pheromones, solution, col, alpha, beta):
    if set(solution) == {None} : 
        weights = [1/(len(solution))for x in range(len(solution))]
    else : 
        weights = []
        for i in range(len(solution)) : 
            weights.append((pow((1/(fitness_score(add_queen(solution, i +1 , col))+1)),alpha) * pow((pheromones[i]),beta)) / sum([(pow ((1/(fitness_score(add_queen(solution, x + 1, col))+1)),alpha)) * pow(pheromones[x],beta)
                                                for x in range(len(solution))]))
    choice = random.choices(
        [i for i in range (1, len(solution) +1 )], weights=weights, k=1)
    return choice[0]

def bestSolution(solutions, scores):
    best_score = float("inf")
    for i in range(len(scores)):
        if (scores[i] < best_score):
            best_score = scores[i]
            best_solution = solutions[i]

    return best_solution, best_score

def antColonyOptimization(num_ants, n_queens, alpha, beta, pheromone_evaporation):
    max_score = n_queens * (n_queens - 1)
    iteration = 0 
    best_solution = None
    best_score = float("inf")
    pheromones = [[1 for _ in range(n_queens)] for _ in range(n_queens)]
    ants = [[None for _ in range(n_queens)] for _ in range(num_ants)]
    scores = [max_score for _ in range(num_ants)]
    begin = datetime.today()
    while 0 not in scores : 
        for i in range(len(ants)):
            for col in range(n_queens):
                ants[i][col] = move_ant(pheromones[col], ants[i], col, alpha, beta)
            scores[i] = fitness_score(ants[i])
        new_best_solution, new_best_score = bestSolution(ants, scores)
        pheromones = updatePheromones(pheromones, ants,  pheromone_evaporation)
        iteration = iteration + 1
        if (new_best_score < best_score):
            best_solution = new_best_solution.copy()
            best_score = new_best_score
            print("Iteration : "+ str(iteration))
            print("Best score : " +str(best_score))
            print("Time : "+ str(datetime.today() - begin))
            if(best_score==0): 
                end = datetime.today() - begin
                print("Solution found : ")
                print(best_solution)
                print("Iteration " + str(iteration))
                print("Time : " + str(end))
                break 

    return best_solution, best_score, end


def plot_average(num_runs, list_nb_ants, func, list_nb_queens, alpha, beta, pheromone_evaporation):
    for n_queens in list_nb_queens : 
        average_time = []
        for nb in list_nb_ants : 
            best_scores = []
            times = []
            for i in range(num_runs):
                best_solution, best_score, time = func(nb, n_queens, alpha, beta, pheromone_evaporation)
                best_scores.append(best_score)
                times.append(time.microseconds)
                print("Finished iteration ", i+1, "/", num_runs)
                print("Best solution found : ")

            average = []
            for i in range(len(times)):
                average.append(times[i])
            
            sum = reduce(lambda a,b:a+b,average)
            average_time.append((sum/len(average)))

        plt.plot(list_nb_ants, average_time, label = str(n_queens)+" queens")     

plot_average(10, list_nb_ants, antColonyOptimization, list_nb_queens, alpha, beta, pheromone_evaporation)
    

plt.xlabel("Number of ants")
plt.ylabel("Average time to find a solution (ms)")
plt.title("pheromone_evaporation = 0.1, alpha = 2, beta = 0.1")
plt.legend()
plt.show()