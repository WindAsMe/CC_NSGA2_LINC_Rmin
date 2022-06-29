import numpy as np
from optproblems import wfg
from os import path


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

a = [[1,2], [3,4], [5,6]]

this_path = path.dirname(path.realpath(__file__))
CC_obj_path = this_path + "/Data/obj/CC/1"

write_obj(a, CC_obj_path)