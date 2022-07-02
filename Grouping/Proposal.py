import numpy as np
from Grouping.Comparison import CCDE
from util import help
from Grouping import optimizer
from bench import benchmark


def EGALINC_Rmin(Dim, Gene_len, problem, pop_size, scale_range, cost):

    base = np.array([np.zeros(Dim)])
    intercept = problem.evalVars(base)[0]

    """
    Algorithm initialization
    """
    NIND = 20
    Max_iter = 20
    random_Pop = help.random_Population(scale_range, Dim, pop_size)
    stop_threshold = 0.1
    final_groups = CCDE(Dim)

    base_fitness = benchmark.base_fitness(random_Pop, problem, intercept)

    cost += len(random_Pop)
    groups_fitness, cost = benchmark.groups_fitness(final_groups, random_Pop, problem, cost, intercept)
    current_best_obj = benchmark.object_function(base_fitness, groups_fitness, problem.M, len(final_groups)) * 0.95

    """
    Apply GA
    """
    if current_best_obj > stop_threshold:
        Groups, obj, temp_cost = optimizer.EGA_optmize(Dim, Gene_len, problem, random_Pop, NIND, Max_iter, intercept)
        cost += temp_cost
        if obj < current_best_obj:
            final_groups = Groups
    return final_groups, cost

