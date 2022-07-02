from geatpy.benchmarks.mops import dtlz
from Grouping import Proposal

Gene_len = 8
Dim = 1000
Problems = [dtlz.DTLZ1.DTLZ1(Dim=Dim), dtlz.DTLZ2.DTLZ2(Dim=Dim),
                dtlz.DTLZ3.DTLZ3(Dim=Dim), dtlz.DTLZ4.DTLZ4(Dim=Dim), dtlz.DTLZ5.DTLZ5(Dim=Dim),
                dtlz.DTLZ6.DTLZ6(Dim=Dim)]

for problem in Problems:
    Proposal_groups, Proposal_cost = Proposal.EGALINC_Rmin(Dim, Gene_len, problem, 5, problem.ranges, 0)
    print("group len: ", len(Proposal_groups))
    print(Proposal_groups)