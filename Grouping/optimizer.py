import copy
import random
import numpy as np

from util import help
from Grouping import DProblem
import geatpy as ea
from bench import benchmark
from Grouping.soea_EGA_templet import soea_EGA_templet


def EGA_optmize(Dim, gene_len, multi_problem, random_Pop, NIND, Max_iter, intercept):
    base_fitness = benchmark.base_fitness(random_Pop, multi_problem, intercept)
    problem = DProblem.MyProblem(Dim, gene_len, multi_problem, random_Pop, intercept, base_fitness)
    algorithm = ea.soea_EGA_templet(problem, ea.Population(Encoding='BG', NIND=NIND), MAXGEN=Max_iter, logTras=0)
    solution = ea.optimize(algorithm, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
    groups, obj = solution['Vars'][0], solution['ObjV']
    EGA_groups = help.Phen_Groups(groups, help.empty_groups(2 ** gene_len))

    return EGA_groups, obj, problem.cost
