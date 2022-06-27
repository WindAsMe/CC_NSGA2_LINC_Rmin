import numpy as np


def base_fitness(Population, problem, intercept):
    fitness = []
    for indi in Population:
        fitness.append(np.array(problem.evalVars(np.array([indi]))[0]) - np.array(intercept))
    return fitness


def group_individual(group, individual):
    part_individual = np.zeros(2000)
    for element in group:
        part_individual[element] = individual[element]
    return part_individual


def groups_fitness(groups, Population, problem, cost, intercept):
    fitness = []
    for indi in Population:
        indi_fitness = [0] * problem.M
        for group in groups:
            indi_fitness += (problem.evalVars(np.array([group_individual(group, indi)]))[0] - intercept)
            cost += 1
        fitness.append(indi_fitness)
    return fitness, cost


# outer interface
# opt_fitness is calculated by groups_fitness
def object_function(base_fitness, groups_fitness, M, group_size):
    weight = 1 / M
    error = 0
    base_fitness = np.array(base_fitness)
    groups_fitness = np.array(groups_fitness)
    for i in range(M):
        error += (weight * abs(1 - sum(groups_fitness[:, i]) / sum(base_fitness[:, i])))
    return error / group_size

















