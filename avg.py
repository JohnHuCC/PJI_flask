import pandas as pd
import json
import numpy as np


# def sort_and_avg(data, name):
#     data.sort(reverse=True)
#     print(str(name)+':')
#     print(data[0:Tree_num])
#     print(np.mean(data[0:Tree_num]))
#     print('')


fid_dict_top3 = {1: [0.98, 0.961, 1.0, 0.941, 0.98],
                 2: [1.0, 0.314, 1.0, 0.941, 0.941],
                 3: [1.0, 0.863, 0.961, 0.922, 1.0],
                 4: [1.0, 0.902,  0.98, 0.98, 0.922],
                 5: [1.0, 0.941, 1.0, 0.98, 0.902],
                 6: [0.98, 0.961, 0.961, 0.863, 0.961],
                 7: [0.961, 0.941, 0.98, 0.941, 0.902],
                 8: [0.98, 0.922, 0.98, 1.0, 0.941],
                 9: [0.941, 1.0, 0.98, 0.98, 1.0],
                 10: [1.0, 0.941, 0.98, 0.941, 0.98],
                 }

fid_dict_relax = {1: [1.0, 0.961, 1.0, 0.941, 1.0],
                  2: [0.98, 0.314, 1.0, 0.961, 0.941],
                  3: [1.0, 0.863, 0.961, 0.922, 1.0],
                  4: [1.0, 0.902,  0.961, 0.98, 0.922],
                  5: [0.961, 0.941, 1.0, 0.941, 0.902],
                  6: [0.98, 0.961, 0.922, 0.843, 0.882],
                  7: [0.941, 0.941, 0.922, 0.941, 0.902],
                  8: [0.98, 0.922, 0.98, 1.0, 0.941],
                  9: [0.941, 1.0, 0.98, 0.98, 1.0],
                  10: [1.0, 0.922, 0.98, 0.941, 0.98],
                  }

fid_dict_strict = {1: [0.941, 0.843, 0.863, 0.804, 0.922],
                   2: [0.824, 0.392, 1.0, 0.745, 0.98],
                   3: [1.0, 0.51, 0.765, 0.843, 1.0],
                   4: [0.961, 0.765, 0.961, 0.902, 0.902],
                   5: [0.941, 0.824, 0.765, 0.471, 0.902],
                   6: [0.765, 0.706, 0.882, 0.569, 0.922],
                   7: [0.824, 0.902, 0.647, 0.765, 0.863],
                   8: [0.98, 0.784, 0.902, 0.922, 0.882],
                   9: [0.588, 0.922, 0.725, 0.941, 1.0],
                   10: [0.961, 0.922, 0.882, 0.765, 0.98],
                   }


def kfold_avg(fid_dict):
    _fid_dict = fid_dict.copy()
    for item in fid_dict.items():
        _fid_dict[item[0]] = round(np.mean(item[1]), 3)
    return _fid_dict


print(kfold_avg(fid_dict_relax))