import numpy as np
from pymoo.problems.many import wfg

Dim = 1000
M = 3
wfg = wfg.WFG1(n_var=Dim, n_obj=M)
print(wfg.xl)
print(wfg.xu)
a = np.zeros(1000)
a[10] = 0.1
print(wfg.evaluate(a))