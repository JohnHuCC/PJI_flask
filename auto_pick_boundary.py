from sympy.logic.boolalg import to_dnf
from sympy.logic import simplify_logic, SOPform
from sympy import symbols
from subprocess import call
import pandas as pd
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import six
import sys
from sklearn.tree import export_graphviz
import personal_DecisionPath2
import json
import joblib
import numpy as np
import random
from sympy import Symbol
from dump_function import *
sys.modules['sklearn.externals.six'] = six

# In[16]: Declare the function for clasification with ICM & nonICM
# 原始推導流程參閱掃描筆記檔案


def Non_ICM_1_morethan_Q(x):
    return {
        'mu(N)_delta_Q_mu(I)': ['>=', '>=', 'delta', 'delta', 'nonICM_1_morethan_s1'],
        'mu(N)_delta_mu(I)_Q': ['>=', '>=', 'delta', 'delta', 'nonICM_1_morethan_s2'],
        'mu(N)_Q_delta_mu(I)': ['>', '>=', 'Q', 'delta', 'nonICM_1_morethan_s3'],
        'mu(N)_Q_mu(I)_delta': ['>', '>=', 'Q', 'delta', 'nonICM_1_morethan_s4'],
        'mu(N)_mu(I)_delta_Q': ['>', '>=', 'mu(I)', 'delta', 'nonICM_1_morethan_s5'],
        'mu(N)_mu(I)_Q_delta': ['>', '>=', 'Q', 'delta', 'nonICM_1_morethan_s6'],
        'delta_mu(N)_Q_mu(I)': ['NotUsed', 'nonICM_1_morethan_s7'],
        'delta_mu(N)_mu(I)_Q': ['NotUsed', 'nonICM_1_morethan_s8'],
        'delta_Q_mu(N)_mu(I)': ['NotUsed', 'nonICM_1_morethan_s9'],
        'delta_mu(I)_Q_mu(N)': ['>=', '>=', 'delta', 'delta', 'nonICM_1_morethan_s10'],
        'delta_mu(I)_mu(N)_Q': ['>=', '>=', 'delta', 'delta', 'nonICM_1_morethan_s11'],
        'delta_Q_mu(I)_mu(N)': ['>=', '>=', 'delta', 'delta', 'nonICM_1_morethan_s12'],
        'Q_mu(N)_delta_mu(I)': ['>', '>=', 'Q', 'delta', 'nonICM_1_morethan_s13'],
        'Q_mu(N)_mu(I)_delta': ['>', '>=', 'Q', 'delta', 'nonICM_1_morethan_s14'],
        'Q_delta_mu(N)_mu(I)': ['>', '>=', 'Q', 'delta', 'nonICM_1_morethan_s15'],
        'Q_delta_mu(I)_mu(N)': ['>=', '>', 'delta', 'Q', 'nonICM_1_morethan_s16'],
        'Q_mu(I)_mu(N)_delta': ['NotUsed', 'nonICM_1_morethan_s17'],
        'Q_mu(I)_delta_mu(N)': ['>=', '>', 'delta', 'Q', 'nonICM_1_morethan_s18'],
        'mu(I)_mu(N)_delta_Q': ['NotUsed', 'nonICM_1_morethan_s19'],
        'mu(I)_mu(N)_Q_delta': ['NotUsed', 'nonICM_1_morethan_s20'],
        'mu(I)_delta_mu(N)_Q': ['>=', '>=', 'delta', 'delta', 'nonICM_1_morethan_s21'],
        'mu(I)_delta_Q_mu(N)': ['>=', '>=', 'delta', 'delta', 'nonICM_1_morethan_s22'],
        'mu(I)_Q_mu(N)_delta': ['NotUsed', 'nonICM_1_morethan_s23'],
        'mu(I)_Q_delta_mu(N)': ['>', '>', 'Q', 'Q', 'nonICM_1_morethan_s24'],
    }.get(x, 'Error')


def Non_ICM_1_lessthan_Q(x):
    return {
        'mu(N)_delta_Q_mu(I)': ['<', '<', 'Q', 'Q', 'nonICM_1_lessthan_s1'],
        'mu(N)_delta_mu(I)_Q': ['<', '<', 'Q', 'Q', 'nonICM_1_lessthan_s2'],
        'mu(N)_Q_delta_mu(I)': ['<=', '<=', 'delta', 'delta', 'nonICM_1_lessthan_s3'],
        'mu(N)_Q_mu(I)_delta': ['NotUsed', 'nonICM_1_lessthan_s4'],
        'mu(N)_mu(I)_delta_Q': ['<', '<=', 'Q', 'delta', 'nonICM_1_lessthan_s5'],
        'mu(N)_mu(I)_Q_delta': ['<=', '<=', 'delta', 'delta', 'nonICM_1_lessthan_s6'],
        'delta_mu(N)_Q_mu(I)': ['NotUsed', 'nonICM_1_lessthan_s7'],
        'delta_mu(N)_mu(I)_Q': ['NotUsed', 'nonICM_1_lessthan_s8'],
        'delta_Q_mu(N)_mu(I)': ['NotUsed', 'nonICM_1_lessthan_s9'],
        'delta_mu(I)_Q_mu(N)': ['<', '<=', 'Q', 'delta', 'nonICM_1_lessthan_s10'],
        'delta_mu(I)_mu(N)_Q': ['<', '<=', 'Q', 'delta', 'nonICM_1_lessthan_s11'],
        'delta_Q_mu(I)_mu(N)': ['<', '<=', 'Q', 'delta', 'nonICM_1_lessthan_s12'],
        'Q_mu(N)_delta_mu(I)': ['<=', '<=', 'delta', 'delta', 'nonICM_1_lessthan_s13'],
        'Q_mu(N)_mu(I)_delta': ['<=', '<=', 'delta', 'delta', 'nonICM_1_lessthan_s14'],
        'Q_delta_mu(N)_mu(I)': ['NotUsed', 'nonICM_1_lessthan_s15'],
        'Q_delta_mu(I)_mu(N)': ['<=', '<=', 'delta', 'delta', 'nonICM_1_lessthan_s16'],
        'Q_mu(I)_mu(N)_delta': ['NotUsed', 'nonICM_1_lessthan_s17'],
        'Q_mu(I)_delta_mu(N)': ['<=', '<=', 'delta', 'delta', 'nonICM_1_lessthan_s18'],
        'mu(I)_mu(N)_delta_Q': ['NotUsed', 'nonICM_1_lessthan_s19'],
        'mu(I)_mu(N)_Q_delta': ['NotUsed', 'nonICM_1_lessthan_s20'],
        'mu(I)_delta_mu(N)_Q': ['<=', '<=', 'delta', 'delta', 'nonICM_1_lessthan_s21'],
        'mu(I)_delta_Q_mu(N)': ['<', '<=', 'Q', 'delta', 'nonICM_1_lessthan_s22'],
        'mu(I)_Q_mu(N)_delta': ['NotUsed', 'nonICM_1_lessthan_s23'],
        'mu(I)_Q_delta_mu(N)': ['<=', '<=', 'delta', 'delta', 'nonICM_1_lessthan_s24'],
    }.get(x, 'Error')


def Non_ICM_0_morethan_Q(x):
    return {
        'mu(N)_delta_Q_mu(I)': ['>=', '>=', 'delta', 'delta', 'nonICM_0_morethan_s1'],
        'mu(N)_delta_mu(I)_Q': ['>=', '>=', 'delta', 'delta', 'nonICM_0_morethan_s2'],
        'mu(N)_Q_delta_mu(I)': ['>', '>', 'Q', 'Q', 'nonICM_1_morethan_s3'],
        'mu(N)_Q_mu(I)_delta': ['NotUsed', 'nonICM_1_morethan_s4'],
        'mu(N)_mu(I)_delta_Q': ['NotUsed', 'nonICM_1_morethan_s5'],
        'mu(N)_mu(I)_Q_delta': ['NotUsed', 'nonICM_1_morethan_s6'],
        'delta_mu(N)_Q_mu(I)': ['>=', '>=', 'delta', 'delta', 'nonICM_0_morethan_s7'],
        'delta_mu(N)_mu(I)_Q': ['>=', '>=', 'delta', 'delta', 'nonICM_0_morethan_s8'],
        'delta_Q_mu(N)_mu(I)': ['>=', '>=', 'delta', 'delta', 'nonICM_0_morethan_s9'],
        'delta_mu(I)_Q_mu(N)': ['NotUsed', 'nonICM_1_morethan_s10'],
        'delta_mu(I)_mu(N)_Q': ['NotUsed', 'nonICM_1_morethan_s11'],
        'delta_Q_mu(I)_mu(N)': ['NotUsed', 'nonICM_1_morethan_s12'],
        'Q_mu(N)_delta_mu(I)': ['>', '>=', 'Q', 'delta', 'nonICM_0_morethan_s13'],
        'Q_mu(N)_mu(I)_delta': ['NotUsed', 'nonICM_1_morethan_s14'],
        'Q_delta_mu(N)_mu(I)': ['>', '>=', 'Q', 'delta', 'nonICM_0_morethan_s15'],
        'Q_delta_mu(I)_mu(N)': ['NotUsed', 'nonICM_1_morethan_s16'],
        'Q_mu(I)_mu(N)_delta': ['>=', '>=', 'delta', 'delta', 'nonICM_0_morethan_s17'],
        'Q_mu(I)_delta_mu(N)': ['>=', '>=', 'delta', 'delta', 'nonICM_0_morethan_s18'],
        'mu(I)_mu(N)_delta_Q': ['>=', '>=', 'delta', 'delta', 'nonICM_0_morethan_s19'],
        'mu(I)_mu(N)_Q_delta': ['>', '>=', 'Q', 'delta', 'nonICM_0_morethan_s20'],
        'mu(I)_delta_mu(N)_Q': ['>=', '>=', 'delta', 'delta', 'nonICM_0_morethan_s21'],
        'mu(I)_delta_Q_mu(N)': ['>=', '>=', 'delta', 'delta', 'nonICM_0_morethan_s22'],
        'mu(I)_Q_mu(N)_delta': ['>', '>=', 'Q', 'delta', 'nonICM_0_morethan_s23'],
        'mu(I)_Q_delta_mu(N)': ['>', '>=', 'Q', 'delta', 'nonICM_0_morethan_s24'],
    }.get(x, 'Error')


def Non_ICM_0_lessthan_Q(x):
    return {
        'mu(N)_delta_Q_mu(I)': ['<', '<=', 'Q', 'delta', 'nonICM_0_lessthan_s1'],
        'mu(N)_delta_mu(I)_Q': ['<=', '<=', 'delta', 'delta', 'nonICM_0_lessthan_s2'],
        'mu(N)_Q_delta_mu(I)': ['<=', '<=', 'delta', 'delta', 'nonICM_0_lessthan_s3'],
        'mu(N)_Q_mu(I)_delta': ['NotUsed', 'nonICM_0_lessthan_s4'],
        'mu(N)_mu(I)_delta_Q': ['NotUsed', 'nonICM_0_lessthan_s5'],
        'mu(N)_mu(I)_Q_delta': ['NotUsed', 'nonICM_0_lessthan_s6'],
        'delta_mu(N)_Q_mu(I)': ['<', '<=', 'Q', 'delta', 'nonICM_0_lessthan_s7'],
        'delta_mu(N)_mu(I)_Q': ['<', '<=', 'mu(N)', 'delta', 'nonICM_0_lessthan_s8'],
        'delta_Q_mu(N)_mu(I)': ['<', '<=', 'Q', 'delta', 'nonICM_0_lessthan_s9'],
        'delta_mu(I)_Q_mu(N)': ['NotUsed', 'nonICM_0_lessthan_s10'],
        'delta_mu(I)_mu(N)_Q': ['NotUsed', 'nonICM_0_lessthan_s11'],
        'delta_Q_mu(I)_mu(N)': ['NotUsed', 'nonICM_0_lessthan_s12'],
        'Q_mu(N)_delta_mu(I)': ['<=', '<=', 'delta', 'delta', 'nonICM_0_lessthan_s13'],
        'Q_mu(N)_mu(I)_delta': ['NotUsed', 'nonICM_0_lessthan_s14'],
        'Q_delta_mu(N)_mu(I)': ['<=', '<=', 'delta', 'delta', 'nonICM_0_lessthan_s15'],
        'Q_delta_mu(I)_mu(N)': ['NotUsed', 'nonICM_0_lessthan_s16'],
        'Q_mu(I)_mu(N)_delta': ['<=', '<=', 'delta', 'delta', 'nonICM_0_lessthan_s17'],
        'Q_mu(I)_delta_mu(N)': ['NotUsed', 'nonICM_0_lessthan_s18'],
        'mu(I)_mu(N)_delta_Q': ['<', '<=', 'Q', 'delta', 'nonICM_0_lessthan_s19'],
        'mu(I)_mu(N)_Q_delta': ['<=', '<=', 'delta', 'delta', 'nonICM_0_lessthan_s20'],
        'mu(I)_delta_mu(N)_Q': ['<', '<', 'Q', 'Q', 'nonICM_0_lessthan_s21'],
        'mu(I)_delta_Q_mu(N)': ['<', '<', 'Q', 'Q', 'nonICM_0_lessthan_s22'],
        'mu(I)_Q_mu(N)_delta': ['<=', '<=', 'delta', 'delta', 'nonICM_0_lessthan_s23'],
        'mu(I)_Q_delta_mu(N)': ['<=', '<=', 'delta', 'delta', 'nonICM_0_lessthan_s24'],
    }.get(x, 'Error')

# OK


def ICM_1_morethan_Q(x):
    return {
        'ICM_Q_delta': ['>', '>=', 'Q', 'delta', 'ICM_1_morethan_s1'],
        'ICM_delta_Q': ['>', '>=', 'ICM', 'delta', 'ICM_1_morethan_s2'],
        'Q_ICM_delta': ['>', '>=', 'Q', 'delta', 'ICM_1_morethan_s3'],
        'Q_delta_ICM': ['>', '>=', 'Q', 'delta', 'ICM_1_morethan_s4'],
        'delta_ICM_Q': ['>=', '>=', 'delta', 'delta', 'ICM_1_morethan_s5'],
        'delta_Q_ICM': ['>=', '>=', 'delta', 'delta', 'ICM_1_morethan_s6'],
    }.get(x, 'Error')


def ICM_1_lessthan_Q(x):
    return {
        'ICM_Q_delta': ['<=', '<=', 'delta', 'delta', 'ICM_1_lessthan_s1'],
        'ICM_delta_Q': ['<', '<=', 'Q', 'delta', 'ICM_1_lessthan_s2'],
        'Q_ICM_delta': ['NotUsed', 'ICM_1_lessthan_s3'],
        'Q_delta_ICM': ['NotUsed', 'ICM_1_lessthan_s4'],
        'delta_ICM_Q': ['NotUsed', 'ICM_1_lessthan_s5'],
        'delta_Q_ICM': ['NotUsed', 'ICM_1_lessthan_s6'],

    }.get(x, 'Error')


def ICM_0_morethan_Q(x):
    return {
        'delta_Q_ICM': ['>=', '>=', 'delta', 'delta', 'ICM_0_morethan_s1'],
        'delta_ICM_Q': ['NotUsed', 'ICM_0_morethan_s2'],
        'ICM_delta_Q': ['NotUsed', 'ICM_0_morethan_s3'],
        'ICM_Q_delta': ['NotUsed', 'ICM_0_morethan_s4'],
        'Q_ICM_delta': ['NotUsed', 'ICM_0_morethan_s5'],
        'Q_delta_ICM': ['>', '>=', 'Q', 'delta', 'ICM_0_morethan_s6'],

    }.get(x, 'Error')


def ICM_0_lessthan_Q(x):
    return {
        'delta_Q_ICM': ['<', '<=', 'Q', 'delta', 'ICM_0_lessthan_s1'],
        'delta_ICM_Q': ['<', '<=', 'Q', 'delta', 'ICM_0_lessthan_s2'],
        'ICM_delta_Q': ['<', '<=', 'Q', 'delta', 'ICM_0_lessthan_s3'],
        'ICM_Q_delta': ['<=', '<', 'delta', 'Q', 'ICM_0_lessthan_s4'],
        'Q_ICM_delta': ['<=', '<', 'delta', 'Q', 'ICM_0_lessthan_s5'],
        'Q_delta_ICM': ['<=', '<', 'delta', 'Q', 'ICM_0_lessthan_s6'],

    }.get(x, 'Error')


PID_index = 1
New_data_X = pd.read_csv('PJI_Dataset/New_data_x.csv', encoding='utf-8')
X_test = New_data_X.iloc[PID_index:PID_index + 1]


def X_test_rename(X_test):
    X_test_ = X_test.rename(
        columns={
            '2X positive culture': 'two_positive_culture',
            'APTT': 'APTT',
            'ASA_2': 'ASA_2',
            'Age': 'Age',
            'HGB': 'Hb',
            'P.T': 'P_T',
            'PLATELET': 'PLATELET',
            'Positive Histology': 'Positive_Histology',
            'Primary, Revision\nnative hip': 'Surgery',
            'Pulurence': 'Purulence',
            'Segment (%)': 'Segment',
            'Serum CRP': 'Serum_CRP',
            'Serum ESR': 'Serum_ESR',
            'Serum WBC ': 'Serum_WBC_',
            'Single Positive culture': 'Single_Positive_culture',
            'Synovial WBC': 'Synovial_WBC',
            'Synovial_PMN': 'Synovial_PMN',
            'Total CCI': 'Total_CCI',
            'Total Elixhauser Groups per record': 'Total_Elixhauser_Groups_per_record',
        }
    )
    return X_test_


def dp_to_singleton(Candidate_DecisionPath):
    # In[19]: 讀取 rules_list, data
    data = Candidate_DecisionPath
    res_combined = []
    # In[20] 拆解所有的 decision path 元素
    # input: all_path
    # output: all_singleton
    for element in data:
        element = element.replace(') | (', ' and ')
        element = element.replace('(', '')
        element = element.replace(')', '')
        element = element.replace('&', 'and')

        res_combined = res_combined + \
            [x.strip(' ') for x in element.split('and')]

    _singleton_set = set()
    for atom in res_combined:
        _singleton_set.add(atom)

    return list(_singleton_set)


def singleton_opt(singleton_list, stacking_result, X_test, top, depth):
    ICM_threshold = json.load(open("auto_boundary/ICM_threshold.json"))
    _2018ICM_ = pd.DataFrame(list(ICM_threshold.items()),
                             columns=['variable', 'threshold'])
    non2018ICM_I = json.load(open("auto_boundary/nonICM_bound_I.json"))
    non2018ICM_N = json.load(open("auto_boundary/nonICM_bound_N.json"))
    non2018ICM_N = pd.DataFrame(list(non2018ICM_N.items()),
                                columns=['variable', 'mu(N)'])
    non2018ICM_I = pd.DataFrame(list(non2018ICM_I.items()),
                                columns=['variable', 'mu(I)'])
    non2018ICM_ = non2018ICM_N.merge(non2018ICM_I, how='outer')

    X_test_ = X_test_rename(X_test)

    decision_list_relax = {}
    decision_list_strict = {}
    for ele in singleton_list:
        eleList = ele.split(' ')
        variable = eleList[0]
        operator = eleList[1]
        NotUsed_flag = 0
        if operator == "==":
            continue
        else:
            if variable in list(non2018ICM_['variable']):  # non2018ICM List
                index = list(non2018ICM_['variable']).index(variable)
                mu_N = non2018ICM_['mu(N)'][index]
                mu_I = non2018ICM_['mu(I)'][index]
                delta = float(X_test_[variable])
                Q = float(eleList[2])
                nonICM = {"mu(N)": mu_N, "mu(I)": mu_I, "delta": delta, "Q": Q}
                nonICM_Sorting = sorted(nonICM, key=nonICM.get)
                concate_nonICM_Sorting = '_'.join(nonICM_Sorting)
                if operator == ">" or operator == ">=":
                    if stacking_result == 0:   # Meta(I) = 0
                        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
                    elif stacking_result == 1:  # Meta(I) = 1
                        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
                elif operator == "<" or operator == "<=":
                    if stacking_result == 0:   # Meta(I) = 0
                        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
                    elif stacking_result == 1:  # Meta(I) = 1
                        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
                if len(result) == 5:
                    operator_relax, operator_strict, _relax_value, _strict_value, situation_num = result
                    relax_value, strict_value = nonICM[_relax_value], nonICM[_strict_value]
                    # dump_nonICM_parameters(
                    #     top, depth, ele, mu_N, mu_I, delta, Q, situation_num)
                else:  # 不考慮
                    NotUsed_flag = 1
                    not_used_, situation_num = result
                    # dump_nonICM_notused_parameters(
                    #     top, depth, ele, mu_N, mu_I, delta, Q, situation_num)

            elif variable in list(_2018ICM_['variable']):  # 2018ICM List
                index = list(_2018ICM_['variable']).index(variable)
                ICM_threshold = _2018ICM_['threshold'][index]
                delta = float(X_test_[variable])
                Q = float(eleList[2])

                ICM = {"ICM": ICM_threshold, "Q": Q, "delta": delta}
                ICM_Sorting = sorted(ICM, key=ICM.get)
                concate_ICM_Sorting = '_'.join(ICM_Sorting)

                if operator == ">" or operator == ">=":
                    if stacking_result == 0:   # Meta(I) = 0
                        result = ICM_0_morethan_Q(concate_ICM_Sorting)
                    elif stacking_result == 1:  # Meta(I) = 1
                        result = ICM_1_morethan_Q(concate_ICM_Sorting)
                elif operator == "<" or operator == "<=":
                    if stacking_result == 0:   # Meta(I) = 0
                        result = ICM_0_lessthan_Q(concate_ICM_Sorting)
                    elif stacking_result == 1:  # Meta(I) = 1
                        result = ICM_1_lessthan_Q(concate_ICM_Sorting)
                if len(result) == 5:
                    operator_relax, operator_strict, _relax_value, _strict_value, situation_num = result
                    relax_value, strict_value = ICM[_relax_value], ICM[_strict_value]
                    # dump_ICM_parameters(top, depth, ele, ICM_threshold,
                    #                     delta, Q, situation_num)
                else:  # 不考慮
                    NotUsed_flag = 1
                    not_used_, situation_num = result
                    # dump_ICM_notused_parameters(top, depth, ele, ICM_threshold,
                    #                             delta, Q, situation_num)

            if NotUsed_flag == 1:
                continue
            else:
                decision_list_relax[ele] = str(
                    variable)+' '+str(operator_relax)+' '+str(relax_value)
                decision_list_strict[ele] = str(
                    variable)+' '+str(operator_strict)+' '+str(strict_value)
                # dump_final_singleton(
                #     top, depth, ele, decision_list_relax, decision_list_strict)

    return decision_list_relax, decision_list_strict


def refresh_final_singleton(dp_dict, final_singleton):
    final_singleton_ = final_singleton.copy()
    for item in final_singleton:
        if item in dp_dict.keys():
            final_singleton_.append(dp_dict[item])
            final_singleton_.remove(item)
    return final_singleton_


def to_POSform(decision_path, _singleton_list_, final_singleton):
    # In[24]: 宣告 Symbols by ns[] for symbols
    sym = "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z"
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z = symbols(
        sym)
    sym_array = [a, b, c, d, e, f, g, h, i, j,
                 k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z]
    ns = {}
    for s_, singleton_ in zip(sym_array, final_singleton):
        ns[s_] = Symbol(singleton_)

    # In[25]: decision path 同質性簡化
    data_ = decision_path
    for val in list(_singleton_list_.keys()):
        data_ = [w_.replace(val, _singleton_list_[val]) for w_ in data_]

    ironmen_dict = {"featureSet": data_}
    # 建立 data frame
    df = pd.DataFrame(ironmen_dict)

    # logic_path = set()
    POSForm = []
    for line in df['featureSet']:
        string = line
        for s_, singleton_ in zip(list(['ns[a]', 'ns[b]', 'ns[c]', 'ns[d]', 'ns[e]',
                                        'ns[f]', 'ns[g]', 'ns[h]', 'ns[i]', 'ns[j]',
                                        'ns[k]', 'ns[l]', 'ns[m]', 'ns[n]', 'ns[o]',
                                        'ns[p]', 'ns[q]', 'ns[r]', 'ns[s]', 'ns[t]',
                                        'ns[u]', 'ns[v]', 'ns[w]', 'ns[x]', 'ns[y]', 'ns[z]',
                                        'ns[A]', 'ns[B]', 'ns[C]',
                                        'ns[D]', 'ns[E]', 'ns[F]', 'ns[G]', 'ns[H]',
                                        'ns[I]', 'ns[J]', 'ns[K]', 'ns[L]', 'ns[M]',
                                        'ns[N]', 'ns[O]', 'ns[P]', 'ns[Q]', 'ns[R]',
                                        'ns[S]', 'ns[T]', 'ns[U]', 'ns[V]', 'ns[W]', 'ns[X]', 'ns[Y]', 'ns[Z]'
                                        ]), list(final_singleton)):

            string = string.replace(singleton_, s_)

        string = string.replace('and', '&')
        string = string.replace('or', '|')

        out = to_dnf(eval(string))
        POSForm.append(out)
    return list(set(POSForm))


def POS_Form_concat(POS_Form):
    _POS_Form = POS_Form.copy()
    for i, rules in enumerate(POS_Form):
        _POS_Form[i] = str(rules)
    _POS_Form = ' | '.join(_POS_Form)
    return _POS_Form


if __name__ == "__main__":
    dp = ['Segment > 65.5 and Synovial_WBC > 7003.0', 'Serum_ESR > 28.5 and Synovial_PMN > 51.5 and Synovial_WBC > 7003.0',
          'APTT > 26.5 and Serum_CRP > 13.5 and Synovial_PMN > 51.5 and Synovial_WBC > 7003.0', 'PLATELET > 213.5 and Serum_CRP > 13.5 and Serum_ESR > 28.5 and Synovial_WBC > 7003.0',
          'Serum_CRP > 13.5 and Serum_ESR > 28.5 and Synovial_WBC > 7003.0 and two_positive_culture == False',
          'Age <= 88.0 and PLATELET > 213.5 and Serum_CRP > 13.5 and Synovial_PMN > 51.5 and Synovial_WBC > 7003.0']
    final_singleton = dp_to_singleton(dp)
    # print(final_singleton)
    X_test_ = X_test_rename(X_test)
    singleton_relax, singleton_strict = singleton_opt(
        final_singleton, 1, X_test_)
    # print('singleton_relax:', singleton_relax)
    # print('singleton_strict:', singleton_strict)
    final_singleton_relax = refresh_final_singleton(
        singleton_relax, final_singleton)
    pos_form = to_POSform(dp, singleton_relax, final_singleton_relax)
    # print('pos_form', pos_form)
    # grid_search()

# %%
