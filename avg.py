import pandas as pd
import json
import numpy as np


# def sort_and_avg(data, name):
#     data.sort(reverse=True)
#     print(str(name)+':')
#     print(data[0:Tree_num])
#     print(np.mean(data[0:Tree_num]))
#     print('')


fid_dict_top3 = {3: [0.7851510416666667, 0.7451435897435896, 0.8005743589743589, 0.8408125, 0.7711216931216931], 4: [0.7055729166666667, 0.7519692307692307,
                                                                                                                     0.7296923076923076, 0.73278125, 0.7058306878306877], 5: [0.7233541666666666, 0.7463179487179485, 0.7263846153846154, 0.7821666666666667, 0.6687883597883598],
                 6: [0.694953125, 0.7133538461538462, 0.7355025641025641, 0.7563958333333333, 0.7213915343915344]}
fid_dict_top5 =


def kfold_avg(fid_dict):
    _fid_dict = fid_dict.copy()
    for item in fid_dict.items():
        _fid_dict[item[0]] = round(np.mean(item[1]), 3)
    return _fid_dict


print(kfold_avg(fid_dict_top3))
