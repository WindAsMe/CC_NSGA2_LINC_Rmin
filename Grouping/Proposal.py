import numpy as np

from util import help
from Grouping import optimizer


def EGALINC_Rmin(Dim, Gene_len, problem, pop_size, scale_range, cost):

    base = np.array([np.zeros(Dim)])
    intercept = problem.evalVars(base)[0]
    """
    Algorithm initialization
    """
    NIND = 20
    Max_iter = 20
    random_Pop = help.random_Population(scale_range, Dim, pop_size)
    """
    Apply GA
    """
    Groups, obj, temp_cost = optimizer.EGA_optmize(Dim, Gene_len, problem, random_Pop, NIND, Max_iter, intercept)
    cost += temp_cost
    return Groups, cost
