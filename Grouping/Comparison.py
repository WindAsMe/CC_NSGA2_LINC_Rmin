import numpy as np
import random
import geatpy as ea


def CCDE(N):
    groups = []
    for i in range(N):
        groups.append([i])
    return groups


def DECC_DG(Dim, problem):
    cost = 2
    groups = CCDE(Dim)
    intercept = problem.evalVars(np.zeros((1, Dim)))[0]
    for i in range(len(groups)-1):
        if i < len(groups) - 1:
            cost += 2
            index1 = np.zeros((1, Dim))
            index1[0][groups[i][0]] = 1
            delta1 = problem.evalVars(index1)[0] - intercept
            for j in range(i+1, len(groups)):
                cost += 2
                if i < len(groups)-1 and j < len(groups) and not DG_Differential(Dim, groups[i][0], groups[j][0], delta1, problem, intercept):
                    groups[i].extend(groups.pop(j))
                    j -= 1
    return groups, cost


def DG_Differential(Dim, e1, e2, a, problem, intercept):

    index1 = np.zeros((1, Dim))
    index2 = np.zeros((1, Dim))
    index1[0][e2] = 1
    index2[0][e1] = 1
    index2[0][e2] = 1

    b = problem.evalVars(index1)[0] - intercept
    c = problem.evalVars(index2)[0] - intercept

    delta = np.abs(c - (a + b))
    for d in delta:
        if d > 0.001:
            return False
    return True


def DECC_G(Dim, groups_num=20, max_number=50):
    return k_s(Dim, groups_num, max_number)


def NoGroup(Dim):
    return [list(range(0, Dim))]


def k_s(Dim, groups_num=20, max_number=50):
    groups = []
    groups_index = list(range(Dim))
    random.shuffle(groups_index)
    for i in range(groups_num):
        group = groups_index[i * max_number: (i+1) * max_number]
        groups.append(group)
    return groups


def DECC_D(Dim, func, scale_range, groups_num=20, max_number=50):

    NIND = Dim * 10
    delta = OptTool(Dim, NIND, func, scale_range)
    groups_index = list(np.argsort(delta))
    groups = []
    for i in range(groups_num):
        group = groups_index[i * max_number: (i + 1) * max_number]
        groups.append(group)
    return groups


class MyProblem(ea.Problem):
    def __init__(self, Dim, benchmark, scale_range):
        name = 'MyProblem'
        M = 1
        self.Dim = Dim
        self.benchmark = benchmark
        maxormins = [-1]
        varTypes = [0] * self.Dim
        lb = [scale_range[0]] * self.Dim
        ub = [scale_range[1]] * self.Dim
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        result = []
        for p in pop.Phen:
            result.append([self.benchmark(p) * (1 + np.random.normal(loc=0, scale=0.01, size=None))])
        pop.ObjV = np.array(result)


def OptTool(Dim, NIND, f, scale_range):
    problem = MyProblem(Dim, f, scale_range)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    population.initChrom(NIND)
    population.Phen = population.Chrom
    problem.aimFunc(population)
    """===========================算法参数设置=========================="""
    Initial_Chrom = population.Chrom
    myAlgorithm = soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = 2
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    solution = ea.optimize(myAlgorithm, verbose=False, outputMsg=False, drawLog=False, saveFlag=False)
    Optimized_Chrom = solution["lastPop"].Chrom
    delta = []
    for i in range(Dim):
        delta.append(abs(sum(Optimized_Chrom[:, i]) - sum(Initial_Chrom[:, i])))
    return delta


def LIMD(Dim, problem):
    cost = 2
    groups = CCDE(Dim)
    f0 = problem.evalVars(np.zeros((1, Dim)))[0]
    for i in range(len(groups)-1):
        if i < len(groups) - 1:
            cost += 2
            index1 = np.zeros((1, Dim))
            index1[0][groups[i][0]] = 1
            fi = problem.evalVars(index1)[0]
            for j in range(i+1, len(groups)):
                cost += 2
                if i < len(groups)-1 and j < len(groups) and not Monotonicity_Check(Dim, groups[i][0], groups[j][0],
                                                                                    fi, problem, f0):
                    groups[i].extend(groups.pop(j))
                    j -= 1
    return groups, cost


def Monotonicity_Check(Dim, e1, e2, fi, problem, f0):

    M = problem.M
    index1 = np.zeros((1, Dim))
    index2 = np.zeros((1, Dim))
    index1[0][e2] = 1
    index2[0][e1] = 1
    index2[0][e2] = 1

    fj = problem.evalVars(index1)[0]
    fij = problem.evalVars(index2)[0]
    for i in range(M):
        if not (fij[i] > fj[i] > f0[i] and fij[i] > fi[i] > f0[i]) or (fij[i] < fj[i] < f0[i] and fij[i] < fi[i] < f0[i]):
            return False
    return True



class soea_DE_currentToBest_1_L_templet(ea.SoeaAlgorithm):
    """
soea_DE_currentToBest_1_L_templet : class - 差分进化DE/current-to-best/1/bin算法类

算法描述:
    为了实现矩阵化计算，本算法类采用打乱个体顺序来代替随机选择差分向量。算法流程如下：
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 采用current-to-best的方法选择差分变异的各个向量，对当前种群进行差分变异，得到变异个体。
    5) 将当前种群和变异个体合并，采用指数交叉方法得到试验种群。
    6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
    7) 回到第2步。

参考文献:
    [1] Das, Swagatam & Suganthan, Ponnuthurai. (2011). Differential Evolution:
        A Survey of the State-of-the-Art.. IEEE Trans. Evolutionary Computation. 15. 4-31.

"""

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing, trappedValue, maxTrappedCount, dirName)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'DE/current-to-best/1/L'
        if population.Encoding == 'RI':
            self.mutOper = ea.Mutde(F=0.5)  # 生成差分变异算子对象
            self.recOper = ea.Xovexp(XOVR=0.5, Half_N=True)  # 生成指数交叉算子对象，这里的XOVR即为DE中的Cr
        else:
            raise RuntimeError('编码方式必须为''RI''.')

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法类的一些动态参数
        # ===========================准备进化============================
        # population.initChrom(NIND)  # 初始化种群染色体矩阵
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        # self.call_aimFunc(population)  # 计算种群的目标函数值
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        # ===========================开始进化============================
        while not self.terminated(population):
            # 进行差分进化操作
            r0 = np.arange(NIND)
            r_best = ea.selecting('ecs', population.FitnV, NIND)  # 执行'ecs'精英复制选择
            experimentPop = ea.Population(population.Encoding, population.Field, NIND)  # 存储试验个体
            experimentPop.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field,
                                                  [r0, None, None, r_best, r0])  # 变异
            experimentPop.Chrom = self.recOper.do(np.vstack([population.Chrom, experimentPop.Chrom]))  # 重组
            self.call_aimFunc(experimentPop)  # 计算目标函数值
            tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
            population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
