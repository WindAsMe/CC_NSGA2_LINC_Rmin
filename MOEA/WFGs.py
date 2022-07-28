# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
from pymoo.problems.many import wfg


class WFG6(ea.Problem):  # 继承Problem父类
    def __init__(self, M=3, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'WFG6'  # 初始化name（函数名称，可以随意设置）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        if Dim is None:
            Dim = M + 9  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = list(range(2, 2 * Dim + 1, 2))  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.wfg = wfg.WFG6(n_var=Dim, n_obj=M)
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)


    def evalVars(self, Vars):  # 目标函数
        f = self.wfg.evaluate(Vars)
        return f

