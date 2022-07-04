import numpy as np


def base_fitness(Population, problem, intercept):
    fitness = []
    for indi in Population:
        fitness.append(np.array(problem.evaluate(indi) - np.array(intercept)))
    return fitness


def group_individual(Dim, group, individual):
    part_individual = np.zeros(1000)
    for element in group:
        part_individual[element] = individual[element]
    return part_individual


def groups_fitness(Dim, M, groups, Population, problem, cost, intercept):
    fitness = []
    for indi in Population:
        indi_fitness = [0] * M
        for group in groups:
            indi_fitness += (problem.evaluate(np.array(group_individual(Dim, group, indi))) - intercept)
            cost += 1
        fitness.append(indi_fitness)
    return fitness, cost


# outer interface
# opt_fitness is calculated by groups_fitness
def object_function(base_fitness, groups_fitness, M, group_size):

    weight = 1 / M
    error = 0
    pop_size = len(base_fitness)
    base_fitness = np.array(base_fitness)
    groups_fitness = np.array(groups_fitness)
    for i in range(pop_size):
        for j in range(M):
            error += (weight * ((groups_fitness[i][j] - base_fitness[i][j]) / group_size) ** 2)
    return error

















