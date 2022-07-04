from geatpy.benchmarks.mops import zdt, dtlz, uf

from pymoo.problems.many import wfg
from MOEA import NSGA
from Grouping import Comparison, Proposal
import geatpy as ea
import numpy as np
from os import path
import matplotlib.pyplot as plt


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
        f.write('[')
        for i in range(len(data)):
            f.write('[')
            for j in range(len(data[i])):
                if j == len(data[i]) - 1:
                    f.write(str(data[i][j]))
                else:
                    f.write(str(data[i][j]) + ', ')
            if i == len(data) - 1:
                f.write(']')
            else:
                f.write('],')
        f.write("],\n")
        f.close()


def write_cost(data, path):
    with open(path, 'a+') as f:
        f.write(str(data) + ", ")
        f.close()


def draw_pareto_2D(name, CC_obj, G_obj, DG_obj, LIMD_obj, Proposal_obj, ref):

    font_title = {'size': 18}
    font = {'size': 16}

    plt.title("Pareto Front of " + name, font_title)
    plt.xlabel("$f_1$", font)
    plt.ylabel("$f_2$", font)
    plt.scatter(G_obj[:, 0], G_obj[:, 1], marker="v", label="CC_NSGA3_G")
    plt.scatter(DG_obj[:, 0], DG_obj[:, 1], marker="^", label="CC_NSGA3_DG")
    plt.scatter(CC_obj[:, 0], CC_obj[:, 1], marker="+", label="CC_NSGA3")
    plt.scatter(LIMD_obj[:, 0], LIMD_obj[:, 1], marker="o", label="CC_NSGA3_LIMD")
    plt.scatter(Proposal_obj[:, 0], Proposal_obj[:, 1], marker="*", label="CC_NSGA3_LINC-Rmin")
    plt.scatter(ref[:, 0], ref[:, 1], marker="<", label="Reference")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    Dim = 1000
    FEs = 1500000
    NIND = 50
    Gene_len = 7
    trial_run = 10
    M = 3
    this_path = path.dirname(path.realpath(__file__))

    # Problems = [zdt.ZDT1.ZDT1(Dim=Dim), zdt.ZDT2.ZDT2(Dim=Dim), zdt.ZDT3.ZDT3(Dim=Dim), zdt.ZDT4.ZDT4(Dim=Dim),
    #             zdt.ZDT5.ZDT5(Dim=Dim), zdt.ZDT6.ZDT6(Dim=Dim), dtlz.DTLZ1.DTLZ1(Dim=Dim), dtlz.DTLZ2.DTLZ2(Dim=Dim),
    #             dtlz.DTLZ3.DTLZ3(Dim=Dim), dtlz.DTLZ4.DTLZ4(Dim=Dim), dtlz.DTLZ5.DTLZ5(Dim=Dim),
    #             dtlz.DTLZ6.DTLZ6(Dim=Dim), dtlz.DTLZ7.DTLZ7(Dim=Dim), uf.UF1.UF1(Dim=Dim), uf.UF2.UF2(Dim=Dim)]

    Problems = [wfg.WFG1(n_var=Dim, n_obj=M), wfg.WFG2(n_var=Dim, n_obj=M), wfg.WFG3(n_var=Dim, n_obj=M),
                wfg.WFG4(n_var=Dim, n_obj=M), wfg.WFG5(n_var=Dim, n_obj=M), wfg.WFG6(n_var=Dim, n_obj=M),
                wfg.WFG7(n_var=Dim, n_obj=M), wfg.WFG8(n_var=Dim, n_obj=M), wfg.WFG9(n_var=Dim, n_obj=M)]

    for func_num in range(len(Problems)):
        print("WFG" + str(func_num+1))
        problem = Problems[func_num]
        Proposal_obj_path = this_path + "/Data/obj/Proposal/WFG" + str(func_num+1)
        ranges = [problem.xl, problem.xu]
        for i in range(1):
            # Proposal_groups, Proposal_cost = Proposal.EGALINC_Rmin(Dim, M, Gene_len, problem, 5, ranges, 0)
            DG_groups, DG_cost = Comparison.DECC_DG(Dim, problem)

            print(len(DG_groups), DG_groups)



            # Proposal_Max_iter = int((FEs - Proposal_cost) / NIND / Dim)
            # Proposal_ObjV = NSGA.CC_NSGA(problem, NIND, Proposal_groups, Proposal_Max_iter)
            # write_obj(Proposal_ObjV, Proposal_obj_path)
