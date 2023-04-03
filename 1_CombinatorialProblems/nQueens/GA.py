import random 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
from functools import reduce

mutation_probability = 0.2
list_nb_pop = [10, 50, 100, 500, 1000]
list_nb_queens = [4, 5, 6, 7, 8]


def population_generation(n_queens, n_population) : 
    return [[random.randint(1, n_queens)
                for _ in range(n_queens)] 
                for _ in range(n_population)]

def fitness_score(pop) : 
    row_score = sum(pop.count(queen)-1 for queen in pop) / 2 
    diag_score = 0 
    for j in range(0,len(pop))  :
        for i in range(0, len(pop)) : 
            if (i != j) : 
                x = abs(i-j)
                y = abs(pop[i] - pop[j])
                if(x==y) : 
                   diag_score = diag_score + 1 
    return row_score + diag_score  

def selectionHalfPop(population) : 
    list_score = [(fitness_score(pop),pop) for pop in population]
    list_score.sort(key=lambda a: a[0])
    new_size = int(len(list_score)/2)
    return [x[1] for x in list_score[0:new_size]]

def selectionTwoPop(population) : 
    list_score = [(fitness_score(pop),pop) for pop in population]
    list_score.sort(key=lambda a: a[0])
    return [x[1] for x in list_score[0:2]]

def duplicateList(bestPopulation): 
    size = len(bestPopulation)
    for i in range(size) : 
        bestPopulation.append(bestPopulation[i])
    return bestPopulation

def crossover1(p1, p2) : 
    child = [0 for i in range(len(p1))]
    for i in range(len(p1)) : 
        if (p1[i]==p2[i]) : 
            child[i] = p1[i]
    for j in range(len(child)) : 
        if(child[j]==0) : 
            child[j] = random.randint(1, len(child))
    return child

def crossover2(p1,p2) : 
    a = random.randint(1, len(p1)-1)
    b = random.randint(a, len(p1))
    child = [p1[i] if i in range(a,b) else p2[i] for i in range(0, len(p1))]
    return child

def mutation(x):
    a = random.randint(0, len(x)-1)
    b = random.randint(0,  len(x)-1)
    elem = x[a]
    x[a] = x[b]
    x[b] = elem
    return x 

def random_select(population) : 
    a = random.randint(0, len(population) -1 )
    return population[a]

def genetic_queen(nb_queens, nb_population):
    generation = 1
    begin = datetime.today()
    population = population_generation(nb_queens, nb_population)
    solution = [seq for seq in population if fitness_score(seq) == 0 ] 
    end = datetime.today() - begin 
    while not 0.0 in [fitness_score(queen) for queen in population]:
        new_population = []
        solution = [] 
        best_population = selectionHalfPop(population)
        generation += 1
        for n in range(nb_population-1) : 
            child = crossover1(random_select(best_population),random_select(best_population))
            if random.random() < mutation_probability:
                child = mutation(child)
            new_population.append(child)
            if fitness_score(child) == 0 :
                end = datetime.today() - begin 
                print("Solution found : ")
                print(child)
                solution = child 
                return solution, end
            
    return solution, end

def plot_average(num_runs, list_nb_pop, func, list_nb_queens):
    for n_queens in list_nb_queens : 
        average_time = []
        for nb in list_nb_pop : 
            times = []
            for i in range(num_runs):
                result = func(n_queens, nb)
                best_solution = result[0]
                time = result[1]
                times.append(time.microseconds)
                print("Finished iteration ", i+1, "/", num_runs)

            average = []
            for i in range(len(times)):
                average.append(times[i])
            
            sum = reduce(lambda a,b:a+b,average)
            average_time.append((sum/len(average)))

        plt.plot(list_nb_pop, average_time, label = str(n_queens)+" queens")     

plot_average(10, list_nb_pop, genetic_queen, list_nb_queens)
    

plt.xlabel("Number of ants")
plt.ylabel("Average time to find a solution (ms)")
plt.title("Mutation probability = 0.2")
plt.legend()
plt.show()