from MOEA import NSGA, WFGs
from Grouping import Comparison, Proposal

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
    Dim = 500
    FEs = 750000
    NIND = 50
    Gene_len = 7
    trial_run = 10
    this_path = path.dirname(path.realpath(__file__))

    # Problems = [zdt.ZDT1.ZDT1(Dim=Dim), zdt.ZDT2.ZDT2(Dim=Dim), zdt.ZDT3.ZDT3(Dim=Dim), zdt.ZDT4.ZDT4(Dim=Dim),
    #             zdt.ZDT5.ZDT5(Dim=Dim), zdt.ZDT6.ZDT6(Dim=Dim), dtlz.DTLZ1.DTLZ1(Dim=Dim), dtlz.DTLZ2.DTLZ2(Dim=Dim),
    #             dtlz.DTLZ3.DTLZ3(Dim=Dim), dtlz.DTLZ4.DTLZ4(Dim=Dim), dtlz.DTLZ5.DTLZ5(Dim=Dim),
    #             dtlz.DTLZ6.DTLZ6(Dim=Dim), dtlz.DTLZ7.DTLZ7(Dim=Dim), uf.UF1.UF1(Dim=Dim), uf.UF2.UF2(Dim=Dim)]
    # Problems = [zdt.ZDT4.ZDT4(Dim=Dim),
    #             zdt.ZDT5.ZDT5(Dim=Dim), zdt.ZDT6.ZDT6(Dim=Dim), dtlz.DTLZ1.DTLZ1(Dim=Dim), dtlz.DTLZ2.DTLZ2(Dim=Dim),
    #             dtlz.DTLZ3.DTLZ3(Dim=Dim), dtlz.DTLZ4.DTLZ4(Dim=Dim), dtlz.DTLZ5.DTLZ5(Dim=Dim),
    #             dtlz.DTLZ6.DTLZ6(Dim=Dim), dtlz.DTLZ7.DTLZ7(Dim=Dim), uf.UF1.UF1(Dim=Dim), uf.UF2.UF2(Dim=Dim)]
    Problems = [WFGs.WFG6(M=3, Dim=Dim), WFGs.WFG7(M=3, Dim=Dim), WFGs.WFG8(M=3, Dim=Dim), WFGs.WFG9(M=3, Dim=Dim)]
    for func_num in range(len(Problems)):
        problem = Problems[func_num]
        print("problem: ", problem.name)

        G_obj_path = this_path + "/Data/obj/G_500/" + problem.name
        DG_obj_path = this_path + "/Data/obj/DG_500/" + problem.name
        LIMD_obj_path = this_path + "/Data/obj/LIMD_500/" + problem.name
        Proposal_obj_path = this_path + "/Data/obj/Proposal_500/" + problem.name

        DG_cost_path = this_path + "/Data/cost/DG_500/" + problem.name
        LIMD_cost_path = this_path + "/Data/cost/LIMD_500/" + problem.name
        Proposal_cost_path = this_path + "/Data/cost/Proposal_500/" + problem.name

        DG_groups, DG_cost = Comparison.DECC_DG(Dim, problem)
        LIMD_groups, LIMD_cost = Comparison.LIMD(Dim, problem)

        write_cost(DG_cost, DG_cost_path)
        write_cost(LIMD_cost, LIMD_cost_path)

        for i in range(trial_run):
            """Decomposition"""
            G_groups = Comparison.DECC_G(Dim, 5, 100)
            # Proposal_groups, Proposal_cost = Proposal_500.EGALINC_Rmin(Dim, Gene_len, problem, 3, problem.ranges, 0)
            # write_cost(Proposal_cost, Proposal_cost_path)

            G_Max_iter = int(FEs / NIND / Dim)
            DG_Max_iter = int((FEs - DG_cost) / NIND / Dim)
            LIMD_Max_iter = int((FEs - LIMD_cost) / NIND / Dim)
            # Proposal_Max_iter = int((FEs - Proposal_cost) / NIND / Dim)

            G_ObjV = NSGA.CC_NSGA(problem, NIND, G_groups, G_Max_iter)
            print("    G_500 finish")
            #
            DG_ObjV = NSGA.CC_NSGA(problem, NIND, DG_groups, DG_Max_iter)
            write_obj(DG_ObjV, DG_obj_path)
            print("    DG_500 finish")

            LIMD_ObjV = NSGA.CC_NSGA(problem, NIND, LIMD_groups, LIMD_Max_iter)
            write_obj(LIMD_ObjV, LIMD_obj_path)
            print("    LIMD_500 finish")

            # Proposal_ObjV = NSGA.CC_NSGA(problem, NIND, Proposal_groups, Proposal_Max_iter)
            # write_obj(Proposal_ObjV, Proposal_obj_path)
            # print("    Proposal_500 finish")

