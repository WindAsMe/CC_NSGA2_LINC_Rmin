import numpy as np
from optproblems import wfg


problem = wfg.WFG1(3, 2000, 4)
index = np.zeros((2000))
print(problem.rand_optimal_solution(k=4,l=3).objective_values)