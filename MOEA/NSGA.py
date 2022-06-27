import geatpy as ea
from MOEA.NSGA3_templet import moea_NSGA3_templet


def CC_NSGA(problem, init_Pop, groups, std_iter):

    Max_len = 1
    for group in groups:
        Max_len = max(Max_len, len(group))
    Max_iter = std_iter * Max_len

    for real_iter in range(Max_iter):
        for group in groups:
            if real_iter > len(group) * std_iter:
                continue
            else:
                algorithm = moea_NSGA3_templet(problem, init_Pop, group, MAXGEN=2, logTras=0)
                algorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
                algorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
                # 求解
                res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
                init_Pop = res["lastPop"]
        real_iter += 1

    problem.evalVars(init_Pop.Phen)
    [levels, criLevel] = ea.ndsortESS(init_Pop.ObjV)
    ObjV = []
    for i in range(len(levels)):
        if levels[i] == 1:
            ObjV.append(init_Pop.ObjV[i])
    return ObjV



