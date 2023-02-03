# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 15:23:57 2022

@author: smirp

Генетическая оптимизация распределения витков спиральной катушки для БПЭ
"""

from deap import base, algorithms
from deap import creator
from deap import tools

import random
import matplotlib.pyplot as plt
import numpy as np

import Field_functions as ff
#plt.style.use(['science', 'ieee'])

#%% Setup

#Genetic
no_of_generations = 50    # максимальное количество поколений
population_size = 50   # количество индивидуумов в популяции
size_of_individual = 100    # длина подлежащей оптимизации битовой строки
probability_of_mutation = 0.1        # вероятность мутации индивидуума
tournSel_k = 4
no_of_variables = 10 # Количество переменных (радиусов витков)
CXPB, MUTPB = 0.4, 0.04 # вероятность мутации и срещивания

# Geometrical & Electrical parameters
a_max = 0.5  # [m] Max coil radius
a_min = 0.05 # [m] Min coil radius
I = 1 # [A] Current
spacing = 1.5 # spacing for calculation domain
cp = 30 # Calculation domain points

# one tuple or pair of lower bound and upper bound for each variable
#bounds = [(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max)] 

bounds = []
minimal_gap = 0.002
for i in range(no_of_variables):
    bounds.append((a_min,a_max-i*minimal_gap))


#%%

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, size_of_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#%%


def decode_all_x(individual, no_of_variables, bounds):
       
    len_chromosome = len(individual)
    len_chromosome_one_var = int(len_chromosome/no_of_variables)
    bound_index = 0
    x = []
    
    for i in range(0,len_chromosome,len_chromosome_one_var):
        # converts binary to decimial using 2**place_value
        chromosome_string = ''.join((str(xi) for xi in individual[i:i+len_chromosome_one_var]))
        binary_to_decimal = int(chromosome_string,2)
        
        lb = bounds[bound_index][0]
        ub = bounds[bound_index][1]
        precision = (ub-lb)/((2**len_chromosome_one_var)-1)
        decoded = (binary_to_decimal*precision)+lb
        x.append(decoded)
        bound_index += 1
    
    return x


def objective_fxn(individual):     
    
    # decoding chromosome to get decoded x in a list
    r_i = decode_all_x(individual, no_of_variables, bounds)
    Bz = ff.Bz(a_max, a_min, no_of_variables, I, spacing, cp, r_i)
    
    height = 0.015 # [m]
    COV = ff.COV_circ(Bz, a_max, height, spacing)

    obj_function_value = COV
    return [obj_function_value] 


def check_feasiblity(individual):
    '''
    Feasibility function for the individual. 
    Returns True if individual is feasible (or constraint not violated),
    False otherwise
    '''
    var_list = decode_all_x(individual, no_of_variables, bounds)
    if sum(var_list) < 0:
        return True
    else:
        return True #################False


def penalty_fxn(individual):
    '''
    Penalty function to be implemented if individual is not feasible or violates constraint
    It is assumed that if the output of this function is added to the objective function fitness values,
    the individual has violated the constraint.
    '''
    var_list = decode_all_x(individual,no_of_variables,bounds)
    return sum(var_list)**2

# registering objetive function with constraint
toolbox.register("evaluate", objective_fxn) # privide the objective function here
toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 1000, penalty_fxn)) # constraint on our objective function

# registering basic processes using bulit in functions in DEAP
toolbox.register("mate", tools.cxTwoPoint) # strategy for crossover, this classic two point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=probability_of_mutation) # mutation strategy with probability of mutation
toolbox.register("select", tools.selTournament, tournsize=tournSel_k) # selection startegy

hall_of_fame = tools.HallOfFame(1)

stats = tools.Statistics()

# registering the functions to which we will pass the list of fitness's of a gneration's offspring
# to ge the results
stats.register('Min', np.min)
stats.register('Max', np.max)
stats.register('Avg', np.mean)
stats.register('Std', np.std)

logbook = tools.Logbook()

#%%

pop = toolbox.population(n=population_size)
fitnesses = list(map(toolbox.evaluate, pop)) 

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit


g = 0
hall_of_fame.clear()

# Begin the evolution
while g < no_of_generations:
    # A new generation
    g = g + 1


    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values


    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


    this_gen_fitness = [] # this list will have fitness value of all the offspring
    for ind in offspring:
        this_gen_fitness.append(ind.fitness.values[0])


    hall_of_fame.update(offspring)

    stats_of_this_gen = stats.compile(this_gen_fitness)

    stats_of_this_gen['Generation'] = g

    print(stats_of_this_gen)

    logbook.append(stats_of_this_gen)


    pop[:] = offspring
    
    
for best_indi in hall_of_fame:
    # using values to return the value and
    # not a deap.creator.FitnessMin object
    best_obj_val_overall = best_indi.fitness.values[0]
    print('Minimum value for function: ',best_obj_val_overall)
    print('Optimum Solution: ',decode_all_x(best_indi,no_of_variables,bounds))


best_obj_val_convergence = logbook.select('Min')[-1]


#%%
# plotting Generations vs Min to see convergence for each generation

plt.figure(figsize=(20, 10))

# using select method in logbook object to extract the argument/key as list
plt.plot(logbook.select('Generation'), logbook.select('Min'))

plt.title("Minimum values of f(x,y) Reached Through Generations",fontsize=20,fontweight='bold')
plt.xlabel("Generations",fontsize=18,fontweight='bold')
plt.ylabel("Value of Himmelblau's Function",fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')


plt.show()