from geatpy.benchmarks.mops import zdt, dtlz, uf
from MOEA import NSGA
from Grouping import Comparison, Proposal
import geatpy as ea
import numpy as np
from os import path
import matplotlib.pyplot as plt
from optproblems import wfg


def initial_population(NIND, problem):
    Encoding = 'RI'  # 编码方式
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    pop = ea.Population(Encoding, Field, NIND)
    pop.initChrom(NIND)
    pop.Phen = pop.Chrom
    pop.ObjV = problem.evalVars(pop.Phen)
    return pop


def draw_pareto(No_obj, CC_obj, G_obj):
    font_title = {'size': 18}
    font = {'size': 16}

    plt.title("Pareto Front", font_title)
    plt.xlabel("$f_1$", font)
    plt.ylabel("$f_2$", font)
    plt.scatter(CC_obj[:, 0], CC_obj[:, 1], marker="o", label="CC_NSGA3")
    plt.scatter(G_obj[:, 0], G_obj[:, 1], marker="v", label="CC_NSGA3_G")
    plt.scatter(No_obj[:, 0], No_obj[:, 1], marker="*", label="NSGA3")

    plt.legend()
    plt.show()


def write_obj(data, path):
    with open(path, 'a+') as f:
        for l in data:
            f.write('[')
            for i in range(len(l)):
                if i == len(l) - 1:
                    f.write(str(l[i]))
                else:
                    f.write(str(l[i]) + ', ')
            f.write('],')
        f.close()


def write_cost(data, path):
    with open(path, 'a+') as f:
        f.write(str(data) + ", ")
        f.close()


if __name__ == '__main__':
    Dim = 2000
    FEs = 3000000
    NIND = 100
    Gene_len = 8
    trial_run = 1
    this_path = path.dirname(path.realpath(__file__))

    Problems = [zdt.ZDT1.ZDT1(Dim=Dim), zdt.ZDT2.ZDT2(Dim=Dim), zdt.ZDT3.ZDT3(Dim=Dim), zdt.ZDT4.ZDT4(Dim=Dim),
                zdt.ZDT5.ZDT5(Dim=Dim), zdt.ZDT6.ZDT6(Dim=Dim), dtlz.DTLZ1.DTLZ1(Dim=Dim), dtlz.DTLZ2.DTLZ2(Dim=Dim),
                dtlz.DTLZ3.DTLZ3(Dim=Dim), dtlz.DTLZ4.DTLZ4(Dim=Dim), dtlz.DTLZ5.DTLZ5(Dim=Dim),
                dtlz.DTLZ6.DTLZ6(Dim=Dim), dtlz.DTLZ7.DTLZ7(Dim=Dim), uf.UF1.UF1(Dim=Dim), uf.UF2.UF2(Dim=Dim)]
    # Problems = [wfg.WFG1()]
    for func_num in range(0, len(Problems)):
        problem = Problems[func_num]
        print("func: ", problem.name)
        #
        G_obj_path = this_path + "/Data/obj/G/" + problem.name
        DG_obj_path = this_path + "/Data/obj/DG/" + problem.name
        LIMD_obj_path = this_path + "/Data/obj/LIMD/" + problem.name
        Proposal_obj_path = this_path + "/Data/obj/Proposal/" + problem.name

        DG_cost_path = this_path + "/Data/cost/DG/" + problem.name
        LIMD_cost_path = this_path + "/Data/cost/LIMD/" + problem.name
        Proposal_cost_path = this_path + "/Data/cost/Proposal/" + problem.name

        DG_groups, DG_cost = Comparison.DECC_DG(Dim, problem)
        LIMD_groups, LIMD_cost = Comparison.LIMD(Dim, problem)

        write_cost(DG_cost, DG_cost_path)
        write_cost(LIMD_cost, LIMD_cost_path)

        for i in range(trial_run):
            """Decomposition"""

            G_groups = Comparison.DECC_G(Dim, 20, 100)
            Proposal_groups, Proposal_cost = Proposal.EGALINC_Rmin(Dim, Gene_len, problem, 5, problem.ranges, 0)

            write_cost(Proposal_cost, Proposal_cost_path)

            G_std_iter = int(FEs / NIND / Dim)
            DG_std_iter = int((FEs - DG_cost) / NIND / Dim)
            LIMD_std_iter = int((FEs - LIMD_cost) / NIND / Dim)
            Proposal_std_iter = int((FEs - Proposal_cost) / NIND / Dim)

            init_Pop = initial_population(NIND, problem)

            G_ObjV = NSGA.CC_NSGA(problem, init_Pop, G_groups, G_std_iter)
            write_obj(G_ObjV, G_obj_path)

            DG_ObjV = NSGA.CC_NSGA(problem, init_Pop, DG_groups, DG_std_iter)
            write_obj(DG_ObjV, DG_obj_path)

            LIMD_ObjV = NSGA.CC_NSGA(problem, init_Pop, LIMD_groups, LIMD_std_iter)
            write_obj(LIMD_ObjV, LIMD_obj_path)

            Proposal_ObjV = NSGA.CC_NSGA(problem, init_Pop, Proposal_groups, Proposal_std_iter)
            write_obj(Proposal_ObjV, Proposal_obj_path)



