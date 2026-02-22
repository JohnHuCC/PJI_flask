#!/usr/bin/env python
###
# @1 Library 引用注意事項
#
# mlxtend.__version__, 0.13.0 家承 (0.18.0)
# sklearn. 0.21.3 家承 (0.23.2)
###
# @2 本檔將以下檔案進行合併。
#
# 2.1 ./PJI_20210128/main.1.20210103.case*.py
# 2.2 ./PJI_20210108/dnf_case1_trace.000_20210108.py
# ---------------------------------------------------
# 2.1.1 蒐集與 stacking 相同答案的 Decision Tree, τ。
# 2.1.1 計算 τ 的 fidelity_scores
# ---------------------------------------------------
# 2.2.1 計算 decision_path of τ 的 fidelity_scores
# 請注意，2.2.1 fidelity_scores of decision_path 與 fidelity_scores of τ 不同
###
#
#
#

# In[1]: Import Library
# 引用適當的 Library
##
import numpy
import pickle
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
import auto_pick_boundary
sys.modules['sklearn.externals.six'] = six
warnings.filterwarnings("ignore")

# In[2]: Collect Decision Tree with the same prediction as the stacking model
# Description: Collect Decision Tree with the same prediction as the stacking model
# In[18]: Transfer the decision_path to POS Form


def transPOSForm_ini(Candidate_DecisionPath):
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

    return res_combined


def transPOSForm(Candidate_DecisionPath):

    # In[19]: 讀取 rules_list, data
    # data = [
    # # '(Serum_ESR > 48.5 and Segment > 64.5) | (Serum_ESR > 49.0 and Serum_CRP > 13.5 and APTT > 29.5 and Synovial_PMN > 52.0) | (Serum_ESR > 39.5 and Synovial_PMN > 69.0 and Age > 32.5 and Synovial_WBC > 367.5) | (Serum_ESR > 55.0 and P_T > 5.0 and Age > 33.0 and Hb > 8.5) | (Serum_ESR > 41.0 and Synovial_PMN > 41.5 and P_T > 5.0 and APTT > 23.5 and Age > 31.5 and Segment > 64.5)',
    # '(Positive_Histology == True and Serum_CRP > 3.0 and Hb <= 13.5) | (Positive_Histology == True and Serum_CRP > 2.5 and APTT > 23.5 and Hb <= 13.5) | (Serum_ESR > 39.5 and APTT > 23.5 and Serum_CRP > 4.0 and Positive_Histology == True) | (Positive_Histology == True and Serum_CRP > 3.0) | (Positive_Histology == True and P_T > 5.0 and Segment > 65.5) | (Serum_ESR > 28.5 and Synovial_WBC > 2483.5 and APTT > 26.5)'
    # ]
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

    _singleton_list = list()
    for atom in _singleton_set:
        _singleton_list.append(atom)

    # In[21]: 列舉 all_singleton 的 features by Set()
    singleton_f = set()
    for i, (val) in enumerate(_singleton_list):

        if val.find("<=") > 0:
            singleton_f.add(val[:val.find("<=")-1])

        elif val.find("<") > 0:
            singleton_f.add(val[:val.find("<")-1])

        elif val.find("==") > 0:
            singleton_f.add(val[:val.find("==")-1])

        elif val.find(">=") > 0:
            singleton_f.add(val[:val.find(">=")-1])

        elif val.find(">") > 0:
            singleton_f.add(val[:val.find(">")-1])

    # In[22]: 列舉 all_singleton 的 features by List()
    _singleton_list_f = list()
    for i, (val) in enumerate(_singleton_list):

        if val.find("<=") > 0:
            _singleton_list_f.append(val[:val.find("<=")-1])
        elif val.find("<") > 0:
            _singleton_list_f.append(val[:val.find("<")-1])
        elif val.find("==") > 0:
            _singleton_list_f.append(val[:val.find("==")-1])
        elif val.find(">=") > 0:
            _singleton_list_f.append(val[:val.find(">=")-1])
        elif val.find(">") > 0:
            _singleton_list_f.append(val[:val.find(">")-1])

    # In[23]: 列舉所有 decision path的 singleton constraints
    _singleton_list_ = {}
    final_singleton = set()
    for i, (val) in enumerate(singleton_f):
        index = [n for n, x in enumerate(_singleton_list_f) if x == val]
        group_list, val_list = [], []
        if len(index) == 1:
            # print(_singleton_list[index[0]])
            final_singleton.add(_singleton_list[index[0]])
        else:
            if (_singleton_list[index[0]].find('<=') > 0):
                for val2 in index:  # index):
                    val_list.append(
                        float(_singleton_list[val2][_singleton_list[val2].find("<=")+3:]))
                min_value = min(val_list)
                min_index = val_list.index(min_value)
                final_singleton.add(_singleton_list[index[min_index]])

                for i in index:
                    _singleton_list_[_singleton_list[i]
                                     ] = _singleton_list[index[min_index]]

            elif (_singleton_list[index[0]].find('<') > 0):
                for val2 in index:  # index):
                    val_list.append(
                        float(_singleton_list[val2][_singleton_list[val2].find("<")+2:]))
                min_value = min(val_list)
                min_index = val_list.index(min_value)
                final_singleton.add(_singleton_list[index[min_index]])

                for i in index:
                    _singleton_list_[_singleton_list[i]
                                     ] = _singleton_list[index[min_index]]

            elif (_singleton_list[index[0]].find('>=') > 0):
                for val2 in index:
                    val_list.append(
                        float(_singleton_list[val2][_singleton_list[val2].find(">=")+3:]))
                max_value = max(val_list)
                max_index = val_list.index(max_value)
                final_singleton.add(_singleton_list[index[max_index]])

                for i in index:
                    _singleton_list_[_singleton_list[i]
                                     ] = _singleton_list[index[max_index]]

            elif (_singleton_list[index[0]].find('>') > 0):
                for val2 in index:
                    val_list.append(
                        float(_singleton_list[val2][_singleton_list[val2].find(">")+2:]))
                max_value = max(val_list)
                max_index = val_list.index(max_value)
                final_singleton.add(_singleton_list[index[max_index]])

                for i in index:
                    _singleton_list_[_singleton_list[i]
                                     ] = _singleton_list[index[max_index]]

    # In[24]: 宣告 Symbols by ns[] for symbols
    sym = "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x"
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x = symbols(
        sym)
    sym_array = [a, b, c, d, e, f, g, h, i, j,
                 k, l, m, n, o, p, q, r, s, t, u, v, w, x]
    ns = {}
    for s_, singleton_ in zip(sym_array, final_singleton):
        ns[s_] = Symbol(singleton_)

    # In[25]: decision path 同質性簡化
    data_ = data
    for val in list(_singleton_list_.keys()):
        data_ = [w_.replace(val, _singleton_list_[val]) for w_ in data_]

    ironmen_dict = {"featureSet": data_}
    # 建立 data frame
    df = pd.DataFrame(ironmen_dict)

    # logic_path = set()
    transPOSForm = []
    for line in df['featureSet']:
        string = line
        for s_, singleton_ in zip(list(['ns[a]', 'ns[b]', 'ns[c]', 'ns[d]', 'ns[e]',
                                        'ns[f]', 'ns[g]', 'ns[h]', 'ns[i]', 'ns[j]',
                                        'ns[k]', 'ns[l]', 'ns[m]', 'ns[n]', 'ns[o]',
                                        'ns[p]', 'ns[q]', 'ns[r]', 'ns[s]', 'ns[t]',
                                        'ns[u]', 'ns[v]', 'ns[w]', 'ns[x]'
                                        ]), list(final_singleton)):

            string = string.replace(singleton_, s_)

        string = string.replace('and', '&')
        string = string.replace('or', '|')

        out = to_dnf(eval(string))
        transPOSForm.append(out)

    return list(set(transPOSForm)), list(final_singleton)


def getCandidate(X_train, y_train, test_X, stacking_model, Explainer_depth,
                 explainer_count):
    """
    input:
        X_train, X of training data of the LOO
        y_train, y of training data of the LOO
        test_X, should be dataframe like. X of test data of the LOO
        stacking_model,
        Explainer_depth, the depth of the Explainer
        explainer_count, the number of randomly generated explainers
    output:
        explainers, five candidate explainers (RF)
        tree_candidates, the tree_candidates of the five candidate explainers
    """
    explainers = {}
    tree_candidates = {}
    while_flag = 0  # the flag of available explainer
    i = 0          # the index of explainer
    while (True):
        i = i + 1
        if while_flag >= explainer_count:
            break

        explainers[i-1] = RandomForestClassifier(max_depth=Explainer_depth,
                                                 n_estimators=100, random_state=i-1)
        explainers[i-1].fit(X_train.values, y_train.values)
        stacking_pred = stacking_model.predict(test_X)
        explainers_pred = explainers[i-1].predict(test_X)
        # print("train:"+str(i))
        if stacking_pred == explainers_pred:
            # print("(stacking_pred == explainers_pred) + 1")
            tree_candidate = []
            for tree_idx in range(explainers[i-1].n_estimators):
                tree_model = explainers[i-1].estimators_[tree_idx]
                tree_pred = tree_model.predict(test_X)
                # print (stacking_pred == tree_pred)
                if stacking_pred == tree_pred:
                    # print("(stacking_pred == tree_pred) + 1")
                    tree_candidate.append(tree_idx)
            tree_candidates[i-1] = tree_candidate
            while_flag = while_flag + 1
        else:
            del explainers[i-1]  # 不存在當下可解釋器，刪除指標
            continue

    return explainers, tree_candidates

# In[3]: Enumerate the decision path of the specific sample
# 列舉特定樣本的決策路徑


# In[8]: feature_selection2 by feature importance from PJI-PI-01-02-2021.docx (Table 4)
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


def interpret(sample, estimator, feature_names):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # Initialize
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    X_test = sample[feature_names].values.reshape(1, -1)
    result = {}
    result['prediction'] = estimator.predict(X_test)
    node_indicator = estimator.decision_path(X_test)
    leave_id = estimator.apply(X_test)
    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]
        :node_indicator.indptr[sample_id + 1]]
    result['info'] = []

    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        result['info'].append(
            [feature_names[feature[node_id]], threshold_sign, round(threshold[node_id], 2)])

    return result


decision_tree_fid = dict()


def getTopN_Fidelity(fidelity_list, top_N_indices, top_N, PID):
    # fidelity_list, candidate_decision_path & the fidelity
    # top_N_indices: the indices of top_N of the decision path.
    fidelity_list_ = []
    rules_list_ = []
    print('top_N_indices:', top_N_indices)

    for idx in top_N_indices:
        print('fidelity_list[idx, fidelity]:', fidelity_list[idx, 'fidelity'])
        fidelity_list_ = fidelity_list_ + \
            fidelity_list[idx, 'fidelity']
        rules_list_ = rules_list_ + fidelity_list[idx, 'rules']
    decision_tree_fid[PID] = fidelity_list_
    with open('final_result/decision_tree_v1_top'+str(top)+'_all_data.txt', 'a') as f:
        f.write('Decision_tree_fid:')
        f.write(str(decision_tree_fid))
        f.write('\n')
    f.close()

    top_n_fidelity_i = sorted(
        range(len(fidelity_list_)), key=lambda k: fidelity_list_[k])[-top_N:]

    return list(itemgetter(*top_n_fidelity_i)(rules_list_))


start_preprocess = time.time()

fid_list = list()
fid_all = list()
fid_problem = list()
problem_id = list()
final_singleton_num = list()


def run_test(PID, top_num, singleton_num):

    # In[6]: Main parameter setting
    # 主體參數設定
    debug_model = 0
    Explainer_depth = 12  # The depth of Explainer Model
    X_res_test = pd.read_csv(
        'PJI_Dataset/New_data_x_test.csv', encoding='utf-8')
    no_group = list(X_res_test['No.Group'])
    PID_index = no_group.index(PID)

    explainers_counter = 5  # 找出 n 組候選 explainers
    CONDITIOS_AvgFidelity = {}

    ### 補充:  iloc vs loc 功能說明 ###
    # iloc，即index locate 用index索引進行定位，所以引數是整型，如：df.iloc[10:20, 3:5]
    # loc，則可以使用column名和index名進行定位，如：df.loc[‘image1’:‘image10’, ‘age’:‘score’]
    # internal_X = pd.read_csv('internal_x_for_new_data.csv', encoding='utf-8')
    # internal_y = pd.read_csv(
    #     'internal_y_for_new_data.csv', encoding='utf-8')
    internal_X = pd.read_csv(
        'PJI_Dataset/internal_x_all.csv', encoding='utf-8')
    internal_y = pd.read_csv(
        'PJI_Dataset/internal_y_all.csv', encoding='utf-8')
    New_data_X = pd.read_csv('PJI_Dataset/New_data_x.csv', encoding='utf-8')
    New_data_y = pd.read_csv('PJI_Dataset/New_data_y.csv', encoding='utf-8')
    end_preprocess = time.time()
    preprocess_time = end_preprocess - start_preprocess
    print("Preprocess time = {}".format(preprocess_time))

    # 7.1 Get the specific patient profile by PID
    internal_x_test = pd.read_csv(
        'PJI_Dataset/internal_x_test.csv', encoding='utf-8')
    no_group_all = list(internal_x_test['No.Group'])
    PID_index_all = no_group_all.index(PID)
    # X_train, y_train = internal_X, internal_y
    X_train, y_train = internal_X.drop(
        index=PID_index_all), internal_y.drop(index=PID_index_all)

    print('PID_index:', PID_index_all)
    # X_test, y_test = New_data_X.iloc[PID_index_all:PID_index_all +
    #                                  1], New_data_y.iloc[PID_index_all:PID_index_all + 1]

    X_test, y_test = internal_X.iloc[PID_index_all:PID_index_all +
                                     1], internal_y.iloc[PID_index_all:PID_index_all + 1]
    print(X_test)
    # 7.2 Split dataset to tr (80%) and val (20%)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=666, shuffle=True)

    # 8.4 Explainer Modeling from 100% dataset   ?
    explainer = RandomForestClassifier(
        max_depth=Explainer_depth, n_estimators=100, random_state=123)
    explainer.fit(X_train.values, y_train.values)
    estimator = explainer.estimators_[5]

    loaded_model = joblib.load('Stacking_model_all_'+str(PID))
    # loaded_model = joblib.load('Stacking_model_new_data')
    stacking_result = loaded_model.predict(X_test)
    print('stacking_result:', stacking_result)
    # 根據ground_truth & Meta Learner 調節
    if stacking_result == 0:
        rule_1, rule_2 = 0, 1  # ground_truth: N, Meta: N
    else:
        rule_1, rule_2 = 1, 0  # ground_truth: I, Meta: I
    # In[11]: Randomly generate random forest and candidate tree
    explainers, tree_candidates = getCandidate(X_train, y_train,
                                               X_test, loaded_model,
                                               Explainer_depth, explainers_counter)
    start_tree_candidate = time.time()
    # 10.1 Prepare the val_df (size = 10) for calculate fidelity scores
    VAL_SIZE = 10
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
                stack_pred = loaded_model.predict(val_df)
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
            # if (debug_model == 1):
            #     print("Decision Path_{} : ".format(num+1))
            res = interpret(X_test, explainers[explain_i].estimators_[
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

        # rules_
        # In[12]: d_path by feature importance from PJI-PI-01-02-2021.docx (Table 4)
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
        if (debug_model == 1):
            print("Enumerate the decision path of the explain[{n}]".format(
                n=explain_i))
        print("Enumerate the decision path of the explain[{n}]".format(
            n=explain_i))
        print(rules_)
        for condition in rules_:
            fidelity = []
            for val_df in VAL_DATASET:
                # val_df = val_df.to_frame
                # print(val_df)
                stack_pred = loaded_model.predict(val_df)
                val_df = val_df.rename(columns=d_path)
                merge_pred = np.where(val_df.eval(condition), rule_1, rule_2)
                for tree_idx in tree_candidates[explain_i]:
                    fidelity.append(accuracy_score(stack_pred, merge_pred))
            AVG_FIDELITYS.append(round(np.mean(fidelity), 3))
        print(AVG_FIDELITYS)

        CONDITIOS_AvgFidelity[explain_i, 'rules'] = CONDITIOS
        CONDITIOS_AvgFidelity[explain_i, 'fidelity'] = AVG_FIDELITYS
    # with open("Tree_Candidate/Tree_candidate_new_data.pkl", "rb") as Tree_candidate_file:
    #     CONDITIOS_AvgFidelity = pickle.load(Tree_candidate_file)
    end_tree_candidate = time.time()
    tree_candiate_time = end_tree_candidate - start_tree_candidate
    print("tree candidate time = {}".format(tree_candiate_time))

    # In[14]: Concatenate multi lists for CONDITIOS_AvgFidelity
    rules_list = getTopN_Fidelity(
        CONDITIOS_AvgFidelity, list(explainers.keys()), top_num, PID)

    # print('rules_list_ini:')
    # print(rules_list)
    # print('')
    # # %%
    # # print('rules_list_fix_not_used:')
    # # final_singleton_ini = transPOSForm_ini(rules_list)
    # # print(final_singleton_ini)
    # # rules_list_temp = auto_pick_boundary.singleton_opt(PID, final_singleton_ini)
    # # print(rules_list_temp)
    # # print('')
    # rules_list = auto_pick_boundary.find_delta(
    #     rules_list, PID, singleton_num, top_num)

    # for rules in rules_list:
    #     if rules == str(''):
    #         print('deleted!!!')
    #         rules_list.remove(rules)
    # print('rules_list_tuned:')
    # print(rules_list)
    # print('')

    # # # In[26]: Call the transPOSForm func.
    # # POS_Form, final_singleton = transPOSForm(rules_list)
    # # print('POS_Form:')
    # # print(POS_Form)
    # # print('')

    # # print('final_singleton:')
    # # print(final_singleton)
    # # print('')
    # # final_singleton_num.append(len(final_singleton))
    # rules_list_p = rules_list
    # for i, rules in enumerate(rules_list_p):
    #     rules_list_p[i] = '(' + str(rules) + ')'
    # rules_list_p = ' | '.join(rules_list_p)
    # print('rules_list_linked:')
    # print(rules_list_p)
    # print('')
    # # res_combined = []
    # # for element in rules_list_p:
    # #     element = element.replace(') | (', ' and ')
    # #     element = element.replace('(', '')
    # #     element = element.replace(')', '')
    # #     element = element.replace('&', 'and')

    # #     res_combined = res_combined + \
    # #         [x.strip(' ') for x in element.split('and')]

    # # rules_list_p_singleton_set = set()
    # # for atom in res_combined:
    # #     rules_list_p_singleton_set.add(atom)
    # # print(rules_list_p_singleton_set)
    # ## 20210531 ##
    # # 套用演算法分類 ICM & non_ICM 分類
    # # 原始推導資料請參閱文件
    # # singleton_opt == parser.py
    # # In[27]: Call the singleton_opt function
    # # opt_decision_list = singleton_opt(X_test)

    # # In[28]: return the indices of the each rule with in truth table
    # # rule_i_ = []
    # # for i in range(len(POS_Form)):
    # #     rule = list()
    # #     rule_i = list()
    # #     for j in range(len(final_singleton)):
    # #         if final_singleton[j] in str(POS_Form[i]):
    # #             rule.append(str(final_singleton[j]))
    # #             rule_i.append(j)
    # #     if (debug_model == 1):
    # #         print(rule, rule_i)
    # #     rule_i_.append(rule_i)

    # # ###
    # # # Init the truth table
    # # init_data = np.array([], dtype=np.int64).reshape(0, len(final_singleton))
    # # lst_all = pd.DataFrame(init_data, columns=final_singleton)
    # # for index_rule in rule_i_:
    # #     # print(i)
    # #     M = len(final_singleton)-len(index_rule)
    # #     lst = [list(a) for a in itertools.product([0, 1], repeat=M)]

    # #     fs = final_singleton.copy()
    # #     # transfer to the formate of pandas.dataframe
    # #     for remove_ind in index_rule:
    # #         fs.remove(final_singleton[remove_ind])

    # #     df_fs = pd.DataFrame(lst, columns=fs)

    # #     # for j in range(len(lst)):
    # #     for i_ in index_rule:
    # #         insert_column = pd.DataFrame([1]*2**M)
    # #         df_fs.insert(i_, final_singleton[i_], insert_column)

    # #     lst_all = pd.concat([df_fs, lst_all])
    # #     lst_all = lst_all.reset_index(drop=True)

    # # # Get the unique elements of the truth table
    # # lst_all_ = set()
    # # for i in lst_all.values:
    # #     lst_all_.add(tuple(i))

    # # # transfer the set of truth_tables to the narray
    # # minterms_ = []
    # # for i in list(lst_all_):
    # #     minterms_.append(list(np.asarray(i).data))

    # # # In[29]: POS_minterm process
    # # sym = "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x"
    # # a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x = symbols(
    # #     sym)
    # # sym_array = [a, b, c, d, e, f, g, h, i, j,
    # #              k, l, m, n, o, p, q, r, s, t, u, v, w, x]
    # # ns_all = list()

    # # for s_, singleton_ in zip(sym_array, final_singleton):
    # #     ns_all.append(Symbol(singleton_))

    # # # In[30]: call the simplify_logic and get the Simplify_decisionRules
    # # Simplify_DecisionRules = "{}".format(
    # #     simplify_logic(SOPform(ns_all, minterms_), 'dnf'))

    # # rule_str = Simplify_DecisionRules
    # # print('simplified_rule:')
    # # print(rule_str)
    # # print('')

    # # j = 65
    # # for i in range(len(final_singleton)):
    # #     a = j + i
    # #     rule_str = rule_str.replace(final_singleton[i], str(chr(a)))

    # # regex = re.compile('[A-Z]')
    # # rule_str_sp = rule_str.split('|')

    # # decision_rule = {}
    # # for i in range(len(rule_str_sp)):
    # #     temp = regex.findall(rule_str_sp[i])
    # #     decision_rule.setdefault(i, temp)

    # # nodeCounter = {}
    # # for k, v in decision_rule.items():
    # #     for node in v:
    # #         if node not in nodeCounter:
    # #             nodeCounter[node] = 1
    # #         else:
    # #             nodeCounter[node] += 1

    # # nodeCounter = dict(
    # #     sorted(nodeCounter.items(), key=lambda x: x[1], reverse=True))
    # # nodeOrder = list(nodeCounter.keys())
    # # for k, v in decision_rule.items():
    # #     decision_rule[k] = list(sorted(v, key=lambda x: nodeOrder.index(x)))

    # # with open("decision_rule.json", "w") as decision_rule_file:
    # #     json.dump(decision_rule, decision_rule_file, indent=4)

    # # decision_rule_map = {}
    # # for i in range(len(final_singleton)):
    # #     a = j + i
    # #     decision_rule_map.setdefault(str(chr(a)), final_singleton[i])

    # # with open("decision_rule_map.json", "w") as decision_rule_file:
    # #     json.dump(decision_rule_map, decision_rule_file, indent=4)

    # # rule_str_sp = Simplify_DecisionRules
    # # rule_str_sp = rule_str_sp.split('|')
    # # for i in range(len(rule_str_sp)):
    # #     rule_str_sp[i] = rule_str_sp[i].replace('&', 'and').replace(
    # #         ' (', '').replace('(', '').replace(') ', '').replace(')', '')
    # # rule_str_bound = auto_pick_boundary.simplified_singleton_opt(
    # #     PID, rule_str_sp)
    # # print('boundary_rule:')
    # # print(rule_str_bound)
    # # print('')
    # # print('rule_str_sp:', rule_str_sp)
    # # AV_FIDELITYS = []
    # # problem_id = []
    # # for explain_i in list(explainers.keys()):
    # #     for condition in rules_list:
    # #         fidelity = []
    # #         for val_df in VAL_DATASET:
    # #             stack_pred = loaded_model.predict(val_df)
    # #             val_df = val_df.rename(columns=d_path)
    # #             merge_pred = np.where(val_df.eval(condition), rule_1, rule_2)
    # #             for tree_idx in tree_candidates[explain_i]:
    # #                 fidelity.append(accuracy_score(stack_pred, merge_pred))
    # #         AV_FIDELITYS.append(round(np.mean(fidelity), 3))
    # # avg = numpy.mean(AV_FIDELITYS)
    # # if avg < 0.5:
    # #     problem_id.append(PID)
    # #     avg = 1 - avg

    # # fid_list.append(avg)
    # AVG_FIDELITYS = []
    # condition = rules_list_p
    # for explain_i in list(explainers.keys()):
    #     fidelity = []
    #     for val_df in VAL_DATASET:
    #         stack_pred = loaded_model.predict(val_df)
    #         val_df = val_df.rename(columns=d_path)
    #         merge_pred = np.where(val_df.eval(condition), rule_1, rule_2)
    #         for tree_idx in tree_candidates[explain_i]:
    #             fidelity.append(accuracy_score(stack_pred, merge_pred))
    #     AVG_FIDELITYS.append(round(np.mean(fidelity), 3))
    # avg = np.mean(AVG_FIDELITYS)
    # print(np.mean(AVG_FIDELITYS))
    # fid_all.append(avg)
    # if avg < 0.5:
    #     problem_id.append(PID)
    #     fid_problem.append(avg)
    # else:
    #     fid_list.append(np.mean(avg))
    # with open('final_result/final_result_v1_top'+str(top)+'_LOO.txt', 'a') as f:
    #     # f.write('final singleton:')
    #     # f.write(str(final_singleton))
    #     # f.write('\n')
    #     # f.write('final singleton num:')
    #     # f.write(str(len(final_singleton)))
    #     # f.write('\n')
    #     f.write('final decision rule:')
    #     f.write(str(rules_list_p))
    #     f.write('\n')
    #     f.write('AVG fidelitys:')
    #     f.write(str(avg))
    #     f.write('\n')
    # f.close()


if __name__ == "__main__":
    rule_problem = [3971, 6101, 7331, 7451, 4562, ]
    rule_ok = [121, 271, 331, 531, 541, 581, 631, 671, 1101, 1121, 1291, 1311, 1331, 1721, 1861,
               2351, 2441, 2951, 3211, 3621, 3671, 4111, 4621, 5221, 5291, 5391, 42, 62, 1302, 1312, 1882,
               1912, 2102, 2502, 2892, 5082, 6012, 6142,  6512,
               6582, 6672, 6852, 6912, 7312]
    New_data = [1311, 151, 8201, 23012, 23752, 42, 331, 531, 541, 581, 631, 671, 1101, 1121, 1291, 1311,
                1331, 1721, 1861, 2351, 2441, 2951, 3211, 3621, 3671, 4111, 4621, 5221, 5291, 5391, 5541,
                1312, 1882, 1912, 2102, 2502, 2892, 5082, 6012, 6142, 6182, 6512,
                6582, 6672, 6852, 6912, 7312, 7332]
    top_num = [7, 8, 9, 10, 11, 12, 13]
    top_num_high = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    singleton_num = [10, 10, 10, 10, 10, 10, 12, 12, 12]
    top_num_high = [12]
    singleton_num = [10]
    for i, top in enumerate(top_num_high):
        for pid in New_data:
            print('PID:', pid)
            print('Top:', top)
            print('Singleton num:', singleton_num[i])
            with open('final_result/final_result_v1_top'+str(top)+'_LOO.txt', 'a') as f:
                f.write('Top:')
                f.write(str(top))
                f.write('\n')
                f.write('PID:')
                f.write(str(pid))
                f.write('\n')
                f.write('Singleton num:')
                f.write(str(singleton_num[i]))
                f.write('\n')
            f.close()
            run_test(pid, top, singleton_num[i])
        print('fid_list:', fid_list)
        avg_fix = numpy.mean(fid_list)
        avg_problem = numpy.mean(fid_problem)
        avg_all = numpy.mean(fid_all)
        # final_singleton_num_avg = numpy.mean(final_singleton_num)
        with open('final_result/test_out_v1_LOO.txt', 'a') as f:
            f.write('Top:')
            f.write(str(top))
            # f.write('\n')
            # f.write('Final_singleton_num_avg:')
            # f.write(str(final_singleton_num_avg))
            f.write('\n')
            f.write('AVG_FIDELITYS_all_fix:')
            f.write(str(avg_fix))
            f.write('\n')
            f.write('AVG_FIDELITYS_all_problem:')
            f.write(str(avg_problem))
            f.write('\n')
            f.write('AVG_FIDELITYS_all_all:')
            f.write(str(avg_all))
            f.write('\n')
            f.write('Problem_id:')
            f.write(str(problem_id))
            f.write('\n')
        fid_list = list()
        final_singleton_num = list()
    f.close()
