import geatpy as ea
import numpy as np

from MOEA.NSGA3_templet import moea_NSGA3_templet


def Pareto_update(group, Pareto_font, Chrom):
    Chrom = Chrom.tolist()
    for i in range(len(Pareto_font) - len(Chrom)):
        Chrom.append(Chrom[i])
    Chrom = np.array(Chrom)
    for i in range(len(group)):
        Pareto_font[:, group[i]] = Chrom[:, i]
    return Pareto_font


def Pareto_index_initial(Max_NIND, problem):
    Field = ea.crtfld("RI", problem.varTypes, problem.ranges, problem.borders)
    pop = ea.Population("RI", Field, Max_NIND)
    pop.initChrom(Max_NIND)
    return pop.Chrom



def CC_NSGA(problem, NIND, groups, Max_iter):
    Max_NIND = 0
    for group in groups:
        Max_NIND = max(len(group) * NIND, Max_NIND)
    Pareto_font_index = Pareto_index_initial(Max_NIND, problem)
    for i in range(len(groups)):
        # print("pop size: ", NIND * len(groups[i]))
        algorithm = moea_NSGA3_templet(problem,
                                       ea.Population(Encoding="RI", NIND=NIND * len(groups[i])),
                                       groups[i],
                                       Pareto_font_index,
                                       MAXGEN=Max_iter,
                                       logTras=0)
        # algorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
        # algorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
        # 求解
        res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
        Pareto_font_index = Pareto_update(groups[i], Pareto_font_index, res["optPop"].Chrom)
    Pareto_font = problem.evalVars(Pareto_font_index)

    [levels, criLevel] = ea.ndsortESS(Pareto_font)
    ObjV = []
    for i in range(len(levels)):
        if levels[i] == 1:
            ObjV.append(Pareto_font[i])
    return np.array(ObjV)


def simple_NSGA(problem, init_Pop):
    algorithm = ea.moea_NSGA2_templet(problem, init_Pop, MAXGEN=300, logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    algorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
    algorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
    # 求解
    res = ea.optimize(algorithm, verbose=False, drawing=1, outputMsg=False, drawLog=False, saveFlag=False)
    print(res["optPop"].ObjV)

