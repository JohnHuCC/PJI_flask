from collections import Counter
from func_timeout import func_set_timeout
import func_timeout
import subprocess
from threading import Timer
import threading
import os
import re
import json
import itertools
from sympy.logic.boolalg import to_dnf
from sympy.logic import simplify_logic, SOPform
from sympy import symbols
from sympy import Symbol
import warnings
from IPython.display import Image
from subprocess import call
import joblib
from operator import itemgetter
import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from mixed_naive_bayes import MixedNB
from mlxtend.classifier import StackingClassifier
import six
import sys
from sklearn.tree import export_graphviz
import time
import eventlet
from imblearn.over_sampling import SMOTE
import personal_DecisionPath2
sys.modules['sklearn.externals.six'] = six
##
warnings.filterwarnings("ignore")

# In[6]: Main parameter setting
# 主體參數設定
debug_model = 0
pID_idx = 5
start_a = time.time()
explainers_counter = 5  # 找出 n 組候選 explainers
CONDITIOS_AvgFidelity = {}

feature_selection2 = [
    'Age',
    'Segment (%)',  # Neutrophil Segment'
    'HGB',          # Hemoglobin
    'PLATELET',
    'Serum WBC ',
    'P.T',          # Prothrombin Time
    'APTT',         # Activated Partial Thromboplastin Time
    'Total CCI',    # Charlson Comorbidity Index
    'Total Elixhauser Groups per record',  # Elixhauser Comorbidity Index
    # Surgery (primary/revision), category
    'Primary, Revision\nnative hip',
    'ASA_2',        # American Society of Anesthesiologists, category
    '2X positive culture',
    'Serum CRP',
    'Serum ESR',
    'Synovial WBC',
    'Single Positive culture',  # category
    'Synovial_PMN',  # 'Synovial Neutrophil',
    'Positive Histology',  # category
    'Purulence',    # 'Pulurence' , category
]


def personalDP(PID):
    start_b = time.time()
    # New data use SMOTE
    X_res = pd.read_csv('New_data_x.csv', encoding='utf-8')
    X_res_test = pd.read_csv('New_data_x_test.csv', encoding='utf-8')
    y_res = pd.read_csv('New_data_y.csv', encoding='utf-8')
    internal_X = pd.read_csv('internal_x.csv', encoding='utf-8')
    internal_y = pd.read_csv('internal_y.csv', encoding='utf-8')

    no_group = list(X_res_test['No.Group'])
    PID_index = 2 + no_group.index(PID)
    print('pid_index:')
    print(PID_index)
    # Old data for model
    X_train, y_train = internal_X, internal_y
    X_test, y_test = internal_X.iloc[PID_index:PID_index +
                                     1], internal_y.iloc[PID_index:PID_index+1]
    X_tr, X_val, y_tr, y_val = train_test_split(
        internal_X, internal_y, test_size=0.2, random_state=666, shuffle=True)

    # Stacking model
    stacking_model = joblib.load('Stacking_model')
    personal_result = stacking_model.predict(X_res)[0]

    if personal_result == 0:
        rule_1, rule_2 = 0, 1  # ground_truth: N, Meta: N
    else:
        rule_1, rule_2 = 1, 0  # ground_truth: I, Meta: I
    print("Stacking Prediction : {}".format(personal_result))

    # RF model
    Explainer_depth = 12
    explainers, tree_candidates = personal_DecisionPath2.getCandidate(internal_X, internal_y,
                                                                      X_res.iloc[PID_index:PID_index +
                                                                                 1], stacking_model,
                                                                      Explainer_depth, explainers_counter)

    # 10.1 Prepare the val_df (size = 10) for calculate fidelity scores
    VAL_SIZE = 1
    VAL_DATASET = []
    Y_VAL_DATASET = []
    for i in range(VAL_SIZE):
        VAL_DATASET.append(resample(X_val, n_samples=55,
                                    replace=False, random_state=i))
        Y_VAL_DATASET.append(resample(y_val, n_samples=55,
                                      replace=False, random_state=i))

    # 10.2 Calculate the fidelity by explain_i
    # explain_i generate the tree_candidates[explain_i]
    for explain_i in list(explainers.keys()):
        VAL_list = []
        rules = []
        top_n_rank = {}
        top_n_tree_idx = {}
        for val_df, y_val_df in zip(VAL_DATASET, Y_VAL_DATASET):
            ACC_list = []
            for tree_idx in tree_candidates[explain_i]:
                tree_pred = explainers[explain_i].estimators_[
                    tree_idx].predict(val_df)
                stack_pred = stacking_model.predict(val_df)
                ACC_list.append(accuracy_score(stack_pred, tree_pred))
            VAL_list.append(ACC_list)

        fidelity_scores = np.array(VAL_list).reshape(
            VAL_SIZE, -1).mean(axis=0)
        rank = np.argsort(-1 * fidelity_scores)
        top_n_rank[explain_i] = fidelity_scores[rank][:10]
        top_n_tree_idx[explain_i] = np.array(
            tree_candidates[explain_i])[rank][:10]

        # 10.3 Enumerate the decision path of the explain_i
        res_combined = []
        for num, (idx, score) in enumerate(zip(top_n_tree_idx[explain_i], top_n_rank[explain_i])):
            res = personal_DecisionPath2.interpret(X_test, explainers[explain_i].estimators_[
                idx], feature_selection2)
            rule = " and ".join([" ".join([str(w_) for w_ in r_])
                                for r_ in res['info']])
            rules.append(rule)
            res_combined = res_combined + \
                [" ".join([str(w_) for w_ in r_]) for r_ in res['info']]

        # 10.4 Fixed the decision path (rules) with condition
        rules_ = rules
        rules_ = [w_.replace('Positive Histology > 0.5',
                             'Positive_Histology == True') for w_ in rules_]
        rules_ = [w_.replace('Positive Histology <= 0.5',
                             'Positive_Histology == False') for w_ in rules_]
        rules_ = [w_.replace('Primary, Revision\nnative hip > 0.5',
                             'Surgery == True') for w_ in rules_]
        rules_ = [w_.replace('Primary, Revision\nnative hip <= 0.5',
                             'Surgery == False') for w_ in rules_]
        rules_ = [w_.replace('Purulence <= 0.5', 'Purulence == False')
                  for w_ in rules_]
        rules_ = [w_.replace('Purulence > 0.5', 'Purulence == True')
                  for w_ in rules_]
        rules_ = [w_.replace('Serum ESR', 'Serum_ESR') for w_ in rules_]
        rules_ = [w_.replace('Serum CRP', 'Serum_CRP') for w_ in rules_]
        rules_ = [w_.replace('BW (kg)', 'BW') for w_ in rules_]
        rules_ = [w_.replace('Segment (%)', 'Segment') for w_ in rules_]
        rules_ = [w_.replace('Synovial WBC', 'Synovial_WBC')
                  for w_ in rules_]
        rules_ = [w_.replace('Height (m)', 'Height') for w_ in rules_]
        rules_ = [w_.replace('2X positive culture <= 0.5',
                             'two_positive_culture == False') for w_ in rules_]
        rules_ = [w_.replace('2X positive culture > 0.5',
                             'two_positive_culture == True') for w_ in rules_]
        rules_ = [w_.replace('Synovial Neutrophil', 'Synovial_PMN')
                  for w_ in rules_]
        rules_ = [w_.replace('Serum WBC ', 'Serum_WBC_') for w_ in rules_]
        rules_ = [w_.replace('Single Positive culture',
                             'Single_Positive_culture') for w_ in rules_]
        rules_ = [w_.replace('P.T', 'P_T') for w_ in rules_]
        rules_ = [w_.replace('Total Elixhauser Groups per record',
                             'Total_Elixhauser_Groups_per_record') for w_ in rules_]
        rules_ = [w_.replace('Total CCI', 'Total_CCI') for w_ in rules_]
        rules_ = [w_.replace('HGB', 'Hb') for w_ in rules_]

        # Modify to suitable names for the parser
        d_path = {
            '2X positive culture': 'two_positive_culture',
            'APTT': 'APTT',
            'ASA_2': 'ASA_2',
            'Age': 'Age',
            'HGB': 'Hb',
            'P.T': 'P_T',
            'PLATELET': 'PLATELET',
            'Positive Histology': 'Positive_Histology',
            'Primary, Revision\nnative hip': 'Surgery',
            'Pulurence': 'Purulence',  # 膿:Purulence
            # 'Rheumatoid Arthritis/collagen': 'Rheumatoid_Arthritis/collagen',
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

        condition_i = 0
        AVG_FIDELITYS = []
        CONDITIOS = rules_
        print("Enumerate the decision path of the explain[{n}]".format(
            n=explain_i))
        for condition in rules_:
            fidelity = []
            for val_df in VAL_DATASET:
                # val_df = val_df.to_frame
                # print(val_df)
                stack_pred = stacking_model.predict(val_df)
                val_df = val_df.rename(columns=d_path)
                merge_pred = np.where(
                    val_df.eval(condition), rule_1, rule_2)
                for tree_idx in tree_candidates[explain_i]:
                    fidelity.append(accuracy_score(stack_pred, merge_pred))
            AVG_FIDELITYS.append(round(np.mean(fidelity), 3))
        # print(AVG_FIDELITYS)
        # validate fidelity
        CONDITIOS_AvgFidelity[explain_i, 'rules'] = CONDITIOS
        CONDITIOS_AvgFidelity[explain_i, 'fidelity'] = AVG_FIDELITYS
    end_b = time.time()
    trtime = end_b - start_b
    print("train model time = {}".format(trtime))
    start_c = time.time()
    # In[14]: Concatenate multi lists for CONDITIOS_AvgFidelity
    rules_list = personal_DecisionPath2.getTopN_Fidelity(
        CONDITIOS_AvgFidelity, list(explainers.keys()), 13)
    # print(rules_list)
    # In[26]: Call the transPOSForm func.
    POS_Form, final_singleton = personal_DecisionPath2.transPOSForm(rules_list)
    # print('POS_Form:')
    # print(POS_Form)

    # print('final_singleton:')
    # print(final_singleton)
    ## 20210531 ##
    # 套用演算法分類 ICM & non_ICM 分類
    # 原始推導資料請參閱文件
    ## singleton_opt == parser.py
    # In[27]: Call the singleton_opt function
    # opt_decision_list = personal_DecisionPath2.singleton_opt(X_test)
    # print('opt_decision_list:')
    # print(opt_decision_list)

    # In[28]: return the indices of the each rule with in truth table
    rule_i_ = []
    for i in range(len(POS_Form)):
        rule = list()
        rule_i = list()
        for j in range(len(final_singleton)):
            if final_singleton[j] in str(POS_Form[i]):
                rule.append(str(final_singleton[j]))
                rule_i.append(j)
        if (debug_model == 1):
            print(rule, rule_i)
        rule_i_.append(rule_i)

    end_c = time.time()
    ruletime = end_c - start_c
    print("get rule time = {}".format(ruletime))
    start_d = time.time()
    # Init the truth table
    init_data = np.array([], dtype=np.int64).reshape(
        0, len(final_singleton))
    lst_all = pd.DataFrame(init_data, columns=final_singleton)
    for index_rule in rule_i_:
        # print(i)
        M = len(final_singleton)-len(index_rule)
        lst = [list(a) for a in itertools.product([0, 1], repeat=M)]

        fs = final_singleton.copy()
        # transfer to the formate of pandas.dataframe
        for remove_ind in index_rule:
            fs.remove(final_singleton[remove_ind])

        df_fs = pd.DataFrame(lst, columns=fs)

        # for j in range(len(lst)):
        for i_ in index_rule:
            insert_column = pd.DataFrame([1]*2**M)
            df_fs.insert(i_, final_singleton[i_], insert_column)

        lst_all = pd.concat([df_fs, lst_all])
        lst_all = lst_all.reset_index(drop=True)

    # Get the unique elements of the truth table
    lst_all_ = set()
    for i in lst_all.values:
        lst_all_.add(tuple(i))

    # transfer the set of truth_tables to the narray
    minterms_ = []
    for i in list(lst_all_):
        minterms_.append(list(np.asarray(i).data))

    # In[29]: POS_minterm process
    sym = "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x"
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x = symbols(
        sym)
    sym_array = [a, b, c, d, e, f, g, h, i, j,
                    k, l, m, n, o, p, q, r, s, t, u, v, w, x]
    ns_all = list()

    for s_, singleton_ in zip(sym_array, final_singleton):
        ns_all.append(Symbol(singleton_))

    # print('ns_all:')
    # print(ns_all)

    # print('minterms:')
    # print(minterms_)
    # In[30]: call the simplify_logic and get the Simplify_decisionRules
    Simplify_DecisionRules = "{}".format(
        simplify_logic(SOPform(ns_all, minterms_), 'dnf'))

    rule_str = Simplify_DecisionRules
    # rule_str = rules_list

    # rule_str = ''.join(str(i)+' | ' for i in rules_list)

    end_d = time.time()
    simplifytime = end_d - start_d
    print("simplify rule time = {}".format(simplifytime))
    start_e = time.time()
    j = 65
    for i in range(len(final_singleton)):
        a = j + i
        rule_str = rule_str.replace(final_singleton[i], str(chr(a)))

    # print('rule_str:')
    # print(rule_str)
    regex = re.compile('[A-Z]')
    rule_str_sp = rule_str.split('|')

    decision_rule = {}
    for i in range(len(rule_str_sp)):
        temp = regex.findall(rule_str_sp[i])
        decision_rule.setdefault(i, temp)

    nodeCounter = {}
    for k, v in decision_rule.items():
        for node in v:
            if node not in nodeCounter:
                nodeCounter[node] = 1
            else:
                nodeCounter[node] += 1

    nodeCounter = dict(
        sorted(nodeCounter.items(), key=lambda x: x[1], reverse=True))
    nodeOrder = list(nodeCounter.keys())
    for k, v in decision_rule.items():
        decision_rule[k] = list(
            sorted(v, key=lambda x: nodeOrder.index(x)))
    with open("decision_rule_"+str(PID)+".json", "w") as decision_rule_file:
        json.dump(decision_rule, decision_rule_file, indent=4)

    decision_rule_map = {}
    for i in range(len(final_singleton)):
        a = j + i
        decision_rule_map.setdefault(str(chr(a)), final_singleton[i])
    with open("decision_rule_map_"+str(PID)+".json", "w") as decision_rule_file:
        json.dump(decision_rule_map, decision_rule_file, indent=4)

    rule_str_sp = rule_str
    rule_str_sp = rule_str_sp.split('|')
    for i in range(len(rule_str_sp)):
        rule_str_sp[i] = rule_str_sp[i].replace('&', 'and').replace(
            ' (', '').replace('(', '').replace(') ', '').replace(')', '')
    print(rule_str_sp)
    # AV_FIDELITYS = []
    # for condition in rule_str_sp:
    #     fidelity = []
    #     for val_df in VAL_DATASET:
    #         stack_pred = stacking_model.predict(val_df)
    #         val_df = val_df.rename(columns=d_path)
    #         merge_pred = np.where(val_df.eval(condition), rule_1, rule_2)
    #         for tree_idx in tree_candidates[explain_i]:
    #             fidelity.append(accuracy_score(stack_pred, merge_pred))
    #     AV_FIDELITYS.append(round(np.mean(fidelity), 3))
    end_e = time.time()
    translatetime = end_e - start_e
    print("translate rule time = {}".format(translatetime))
    return (personal_result)


# if __name__ == "__main__":
#     personalDP(1101)
