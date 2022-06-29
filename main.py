from geatpy.benchmarks.mops import zdt, dtlz, uf

from MOEA import NSGA
from Grouping import Comparison, Proposal
import geatpy as ea
import numpy as np
from os import path
import matplotlib.pyplot as plt
from optproblems import wfg





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
    trial_run = 1
    this_path = path.dirname(path.realpath(__file__))

    Problems = [zdt.ZDT1.ZDT1(Dim=Dim), zdt.ZDT2.ZDT2(Dim=Dim), zdt.ZDT3.ZDT3(Dim=Dim),
                zdt.ZDT5.ZDT5(Dim=Dim), zdt.ZDT6.ZDT6(Dim=Dim), dtlz.DTLZ1.DTLZ1(Dim=Dim), dtlz.DTLZ2.DTLZ2(Dim=Dim),
                dtlz.DTLZ3.DTLZ3(Dim=Dim), dtlz.DTLZ4.DTLZ4(Dim=Dim), dtlz.DTLZ5.DTLZ5(Dim=Dim),
                dtlz.DTLZ6.DTLZ6(Dim=Dim), dtlz.DTLZ7.DTLZ7(Dim=Dim), uf.UF1.UF1(Dim=Dim), uf.UF2.UF2(Dim=Dim)]
    # Problems = [wfg.WFG1()]
    for func_num in range(0, len(Problems)):
        problem = Problems[func_num]
        print("problem: ", problem.name)

        CC_obj_path = this_path + "/Data/obj/CC/" + problem.name
        G_obj_path = this_path + "/Data/obj/G/" + problem.name
        DG_obj_path = this_path + "/Data/obj/DG/" + problem.name
        LIMD_obj_path = this_path + "/Data/obj/LIMD/" + problem.name
        Proposal_obj_path = this_path + "/Data/obj/Proposal/" + problem.name

        DG_cost_path = this_path + "/Data/cost/DG/" + problem.name
        LIMD_cost_path = this_path + "/Data/cost/LIMD/" + problem.name
        Proposal_cost_path = this_path + "/Data/cost/Proposal/" + problem.name

        CC_groups = Comparison.CCDE(Dim)
        DG_groups, DG_cost = Comparison.DECC_DG(Dim, problem)
        LIMD_groups, LIMD_cost = Comparison.LIMD(Dim, problem)

        write_cost(DG_cost, DG_cost_path)
        write_cost(LIMD_cost, LIMD_cost_path)

        for i in range(trial_run):
            """Decomposition"""
            G_groups = Comparison.DECC_G(Dim, 10, 100)
            Proposal_groups, Proposal_cost = Proposal.EGALINC_Rmin(Dim, Gene_len, problem, 5, problem.ranges, 0)
            write_cost(Proposal_cost, Proposal_cost_path)

            CC_Max_iter = int(FEs / NIND / Dim)
            G_Max_iter = int(FEs / NIND / Dim)
            DG_Max_iter = int((FEs - DG_cost) / NIND / Dim)
            LIMD_Max_iter = int((FEs - LIMD_cost) / NIND / Dim)
            Proposal_Max_iter = int((FEs - Proposal_cost) / NIND / Dim)

            CC_ObjV = NSGA.CC_NSGA(problem, NIND, CC_groups, CC_Max_iter)
            write_obj(CC_ObjV, CC_obj_path)
            print("    CC finish")

            G_ObjV = NSGA.CC_NSGA(problem, NIND, G_groups, G_Max_iter)
            write_obj(G_ObjV, G_obj_path)
            print("    G finish")

            DG_ObjV = NSGA.CC_NSGA(problem, NIND, DG_groups, DG_Max_iter)
            write_obj(DG_ObjV, DG_obj_path)
            print("    DG finish")

            LIMD_ObjV = NSGA.CC_NSGA(problem, NIND, LIMD_groups, LIMD_Max_iter)
            write_obj(LIMD_ObjV, LIMD_obj_path)
            print("    LIMD finish")

            Proposal_ObjV = NSGA.CC_NSGA(problem, NIND, Proposal_groups, Proposal_Max_iter)
            write_obj(Proposal_ObjV, Proposal_obj_path)
            print("    Proposal finish")

            # draw_pareto_2D(problem.name, CC_ObjV, G_ObjV, DG_ObjV, LIMD_ObjV, Proposal_ObjV, problem.calReferObjV())


