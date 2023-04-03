from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

def mutation(pop):
    row = len(pop[:,0])
    col = len(pop[0,:])-4
    rand_r = int(np.random.randint(0,row,(1,1))) 
    rand_c = int(np.random.randint(0,col,(1,1))) 
    pop[rand_r,rand_c] = 1-int(pop[rand_r,rand_c])
    #return pop

def crossover(rand_pop, rand_popTemp):
    dup = np.array([])
    while 1:
        ranIndex = np.random.randint(low=0, high=row, size=2)
        u, c = np.unique(ranIndex, return_counts=True)
        dup = u[c > 1]
        if dup.size == 0:
            break

    c = 0
    for i in ranIndex:
        rand_popTemp[c] = rand_pop[i,0:col]
        c += 1
        
    k = np.random.randint(low=1, high=col, size=1) 
    
    a = []
    b =[]
    a = rand_popTemp[0,int(k):col].tolist()
    b = rand_popTemp[1,int(k):col].tolist()
    rand_popTemp[1,int(k):col] = a
    rand_popTemp[0,int(k):col] = b

    c = 0
    for i in ranIndex:
        rand_pop[i,0:col] = rand_popTemp[c]
        c += 1

def next_gen(rand_pop, rand_popTemp):
        count = 0
        c = 0
        for i in range(row):
            noc = rand_pop[i,7]
            count +=noc
            if count>row:
                noc -=1
                
            for j in range(int(noc)):
                rand_popTemp[c] = rand_pop[i,0:col]
                c +=1
        rand_pop[:,0:col] = rand_popTemp


#for col in [100, 200, 300, 400, 500]:
#    row = 5 # Total number of individuals
#    lab = f"{col}"
#    print('Number of items ', col)

for row in [5,10,15,20,25]:
    col = 20 # Total number of individuals
    lab = f"{row}"
    print('Number of individuals ', row)

    w = np.random.randint(1, 20, col)
    v = np.random.randint(1, 20, col)

    weight = np.sum(w)//2

    rand_pop = np.random.randint(0,2,(row,col)) 
    rand_popTemp = np.random.randint(0,2,(row,col))

    addZeros = np.zeros((row,4))
    rand_pop = np.append(rand_pop, addZeros, axis=1)

    # Main iteration starts from here
    maxVal = 0
    capIndividual = []
    n_iterations = 1000
    best_history = []
    seconds = 10

    end = datetime.today() + timedelta(seconds=seconds)
    while datetime.today() < end:        
        for i in range(row):
            sumWeight = sum(np.multiply(w, rand_pop[i,0:col])) # Total weight calculation
            rand_pop[i,col] = sumWeight
            sumValue = sum(np.multiply(v, rand_pop[i,0:col])) # Total value calculation
            
            if sumWeight > weight: # Constraint checking
                sumValue = 0
                rand_pop[i,col+1] = sumValue
                continue
            
            rand_pop[i,col+1] = sumValue
            
            if maxVal < sumValue:
                maxVal = sumValue
                capIndividual = rand_pop[i,0:col]
                
        # Fitness(i) calculation
        for i in range(row):
            rand_pop[i,col+2] = rand_pop[i,col+1]/np.average(rand_pop[:,col+1])

        # Next generation formation
        next_gen(rand_pop, rand_popTemp)
            
        # Crossover starts 
        crossover(rand_pop, rand_popTemp)

        # Mutation starts
        mutation(rand_pop)

        #print("The individual is: ",capIndividual,"and the best value is",maxVal) 
        best_history.append(maxVal)

    plt.plot(np.linspace(0,seconds,len(best_history)), best_history, label = lab)

plt.xlabel("Algorithm Execution Time (s)")
plt.ylabel("Optimized total value")

#plt.legend(title = 'Number of items')
plt.legend(title = 'Number of individuals')

plt.show()
