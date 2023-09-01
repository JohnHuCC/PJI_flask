from dump_function import *
from update_data_bound import *
from pyeda.boolalg.minimization import espresso_exprs
from pyeda.boolalg import boolfunc
from pyeda.boolalg.expr import expr, OrOp, AndOp
from pyeda.boolalg.expr import exprvar
from pyparsing import *
from rule_parser import *
from auto_pick_boundary import *
import time
from sklearn import tree
from sklearn.tree import export_graphviz
import sys
import six
from mlxtend.classifier import StackingClassifier
from mixed_naive_bayes import MixedNB
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from operator import itemgetter
import joblib
from subprocess import call
from IPython.display import Image
import warnings
from sympy import Symbol
from sympy import symbols
from sympy.logic import simplify_logic, SOPform
from sympy.logic.boolalg import to_dnf
import itertools
import json
import re
import pickle
sys.modules['sklearn.externals.six'] = six
warnings.filterwarnings("ignore")


def transPOSForm(Candidate_DecisionPath):

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
            for i in index:

                if (_singleton_list[i].find('<=') > 0 or _singleton_list[i].find('<') > 0):
                    for val2 in index:  # index):
                        singleton_temp = _singleton_list[val2].split(' ')
                        if (singleton_temp[1] == '<=' or singleton_temp[1] == '<'):
                            val_list.append(float(singleton_temp[2]))
                        else:
                            val_list.append(float('inf'))
                        # val_list.append(
                        #     float(_singleton_list[val2][_singleton_list[val2].find("<=")+3:]))
                    min_value = min(val_list)
                    min_index = val_list.index(min_value)

                    val_list = []
                    final_singleton.add(_singleton_list[index[min_index]])
                    # for i in index:

                    #     _singleton_list_[_singleton_list[i]
                    #                      ] = _singleton_list[index[max_index]]
                    for k in index:
                        singleton_temp = _singleton_list[k].split(' ')
                        if (singleton_temp[1] == '<=' or singleton_temp[1] == '<'):
                            _singleton_list_[_singleton_list[i]
                                             ] = _singleton_list[index[min_index]]

                elif (_singleton_list[i].find('>=') > 0 or _singleton_list[i].find('>') > 0):
                    for val2 in index:
                        singleton_temp = _singleton_list[val2].split(' ')
                        if (singleton_temp[1] == '>=' or singleton_temp[1] == '>'):
                            val_list.append(float(singleton_temp[2]))
                        else:
                            val_list.append(float('-inf'))
                        # val_list.append(
                        #     float(_singleton_list[val2][_singleton_list[val2].find(">=")+3:]))
                    max_value = max(val_list)
                    max_index = val_list.index(max_value)

                    val_list = []
                    final_singleton.add(_singleton_list[index[max_index]])
                    # for i in index:

                    #     _singleton_list_[_singleton_list[i]
                    #                      ] = _singleton_list[index[min_index]]
                    for k in index:
                        singleton_temp = _singleton_list[k].split(' ')
                        if (singleton_temp[1] == '>=' or singleton_temp[1] == '>'):
                            _singleton_list_[_singleton_list[i]
                                             ] = _singleton_list[index[max_index]]

    print('_singleton_list_:', _singleton_list_)
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
    data_ = data
    for val in list(_singleton_list_.keys()):
        data_ = [w_.replace(val, _singleton_list_[val]) for w_ in data_]

    ironmen_dict = {"featureSet": data_}
    # 建立 data frame
    df = pd.DataFrame(ironmen_dict)
    print('df:', df)
    # logic_path = set()
    transPOSForm = []
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
        transPOSForm.append(out)

    # return list(set(transPOSForm)), list(final_singleton)
    return list(set(transPOSForm)), list(set(final_singleton))


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
        if stacking_pred == explainers_pred:
            print("(stacking_pred == explainers_pred)")
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
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]                                        :node_indicator.indptr[sample_id + 1]]
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


def getTopN_Fidelity(fidelity_list, top_N_indices, top_N, PID, explainer_depth):
    # fidelity_list, candidate_decision_path & the fidelity
    # top_N_indices: the indices of top_N of the decision path.
    fidelity_list_ = []
    rules_list_ = []
    for idx in top_N_indices:
        print('fidelity_list[idx, fidelity]:', fidelity_list[idx, 'fidelity'])
        fidelity_list_ = fidelity_list_ + \
            fidelity_list[idx, 'fidelity']
        rules_list_ = rules_list_ + fidelity_list[idx, 'rules']
    decision_tree_fid[PID] = fidelity_list_

    top_n_fidelity_i = sorted(
        range(len(fidelity_list_)), key=lambda k: fidelity_list_[k])[-top_N:]

    return list(itemgetter(*top_n_fidelity_i)(rules_list_))


def kmap_minimize(final_singleton, POS_Form):
    # In[28]: return the indices of the each rule with in truth table
    rule_i_ = []
    for i in range(len(POS_Form)):
        rule = list()
        rule_i = list()
        for j in range(len(final_singleton)):
            if final_singleton[j] in str(POS_Form[i]):
                rule.append(str(final_singleton[j]))
                rule_i.append(j)
        rule_i_.append(rule_i)

    ###
    # Init the truth table
    init_data = np.array([], dtype=np.int64).reshape(0, len(final_singleton))
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

    # In[30]: call the simplify_logic and get the Simplify_decisionRules
    Simplify_DecisionRules = "{}".format(
        simplify_logic(SOPform(ns_all, minterms_), 'dnf'))

    return Simplify_DecisionRules


def dp_to_json(rule_str, final_singleton, PID):
    for i in range(len(final_singleton)):
        if i < 26:
            j = 65
            a = j + i
            rule_str = rule_str.replace(final_singleton[i], str(chr(a)))
        else:
            j = 71
            a = j + i
            rule_str = rule_str.replace(final_singleton[i], str(chr(a)))

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
    print("nodeOrder:")
    print(nodeOrder)
    with open("Decision_rule/decision_rule_"+str(PID)+".json", "w") as decision_rule_file:
        json.dump(decision_rule, decision_rule_file, indent=4)

    decision_rule_map = {}
    for i in range(len(final_singleton)):
        if i < 26:
            j = 65
            a = j + i
            decision_rule_map.setdefault(str(chr(a)), final_singleton[i])
        else:
            j = 71
            a = j + i
            decision_rule_map.setdefault(str(chr(a)), final_singleton[i])

    with open("Decision_rule/decision_rule_map_"+str(PID)+".json", "w") as decision_rule_file:
        json.dump(decision_rule_map, decision_rule_file, indent=4)

    return None


def dp_to_json_simplified(rule_str, final_singleton, PID):
    for i in range(len(final_singleton)):
        if i < 26:
            j = 65
            a = j + i
            rule_str = rule_str.replace(final_singleton[i], str(chr(a)))
        else:
            j = 71
            a = j + i
            rule_str = rule_str.replace(final_singleton[i], str(chr(a)))

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
    print("nodeOrder:")
    print(nodeOrder)
    with open("Decision_rule/simplified_decision_rule_"+str(PID)+".json", "w") as decision_rule_file:
        json.dump(decision_rule, decision_rule_file, indent=4)

    decision_rule_map = {}
    for i in range(len(final_singleton)):
        if i < 26:
            j = 65
            a = j + i
            decision_rule_map.setdefault(str(chr(a)), final_singleton[i])
        else:
            j = 71
            a = j + i
            decision_rule_map.setdefault(str(chr(a)), final_singleton[i])

    with open("Decision_rule/simplified_decision_rule_map_"+str(PID)+".json", "w") as decision_rule_file:
        json.dump(decision_rule_map, decision_rule_file, indent=4)

    return None


def kf_validation(PJI_Data_X, PJI_Data_y):
    skf = StratifiedKFold(n_splits=5)
    skf = skf.split(PJI_Data_X, PJI_Data_y)
    return skf


def kf_validation_random(PJI_Data_X, PJI_Data_y, PID):
    skf = StratifiedKFold(n_splits=5, random_state=PID, shuffle=True)
    skf = skf.split(PJI_Data_X, PJI_Data_y)
    return skf


def gen_stacking_model(X_train, y_train):
    xgb = xgboost.XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=50, scale_pos_weight=0.85,
                                enable_experimental_json_serialization=True, tree_method='hist', random_state=0)
    rf = RandomForestClassifier(n_estimators=50, random_state=123, max_depth=5)
    nb_pipe = MixedNB(categorical_features=[9, 10, 11, 15, 17, 18])
    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(
        solver='lbfgs', max_iter=500, random_state=123))
    svc_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, random_state=123, C=10, gamma=0.01))
    ])
    stacking_model = StackingClassifier(
        classifiers=[xgb, rf, lr_pipe],
        use_probas=True,
        average_probas=True,
        use_features_in_secondary=True,
        meta_classifier=svc_pipe
    )
    stacking_model.fit(X_train, y_train)
    joblib.dump(stacking_model, 'Stacking_model_0831')
    return stacking_model


def espresso_simplify(final_singleton, singleton_dict, rules_, PID):
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z = map(
        exprvar, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = map(
        exprvar, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    final_singleton_fix = refresh_final_singleton(
        singleton_dict, final_singleton)
    Pos_form_fix = to_POSform(
        rules_, singleton_dict, final_singleton_fix)
    _POS_Form_fix = POS_Form_concat(Pos_form_fix)
    singleton_map = dict()
    expr_map, singleton_map = map_to_var(
        final_singleton_fix, _POS_Form_fix, singleton_map)

    print('singleton_map:', singleton_map)
    print('map to var:', expr_map)
    f1 = eval(expr_map)
    f2 = ~a & ~b & ~c | ~a & ~b & c | a & ~b & c | a & b & c | a & b & ~c
    f3 = W & Q & S | P & E & W | P & N & T | P & D & T
    f1m, f2m, f3m = espresso_exprs(
        f1.to_dnf(), f2.to_dnf(), f3.to_dnf())
    print(f1m)
    print(f2m)
    simplified_rule = parse_ast(f1m)
    print('simplified_rule:', simplified_rule)
    simplified_rule = alphabet_to_singleton(
        simplified_rule, singleton_map)

    rule_splited = spilt_rule(simplified_rule)
    print('splited_rule:', rule_splited)
    rule_dict = split_singleton(rule_splited)
    rule_dict = filt_rule(rule_dict, PID)
    print('rule_dict:', rule_dict)
    rule_str = concat_rule(rule_dict)
    return rule_str, final_singleton_fix


def fix_dp(rules):
    # 10.4 Fixed the decision path(rules) with condition
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
    return rules_


def bootstrap_data(VAL_SIZE, X_data, y_data, n_samples, random_state):
    # 10.1 Prepare the val_df (size = 10) for calculate fidelity scores
    X_DATASET = []
    Y_DATASET = []
    for i in range(VAL_SIZE):
        X_DATASET.append(resample(X_data, n_samples=n_samples,
                                  replace=True, random_state=random_state))
        Y_DATASET.append(resample(y_data, n_samples=n_samples,
                                  replace=True, random_state=random_state))
    return X_DATASET, Y_DATASET


def merge_bootstrap_data(X_DATASET, Y_DATASET):
    merged_df_x = pd.concat(X_DATASET, ignore_index=True)
    merged_df_y = pd.concat(Y_DATASET, ignore_index=True)
    return merged_df_x, merged_df_y


fid_list = list()
fid_all = list()
fid_problem = list()
problem_id = list()
final_singleton_num = list()
rules_list = list()
consistent_tree_len_all = list()
tree_candiate_time_list = list()
kmap_time_list = list()


def run_test(PID, top_num, Explainer_depth):
    explainers_counter = 1  # 找出 n 組候選 explainers
    X_res_test = pd.read_csv(
        'PJI_Dataset/internal_x_test.csv', encoding='utf-8')
    internal_X = pd.read_csv(
        'PJI_Dataset/internal_x_all.csv', encoding='utf-8')
    internal_y = pd.read_csv(
        'PJI_Dataset/internal_y_all.csv', encoding='utf-8')
    no_group = list(X_res_test['No.Group'])
    PID_index = no_group.index(PID)

    # 7.1 Get the specific patient profile by PID
    X_train, y_train = internal_X.drop(
        index=PID_index), internal_y.drop(index=PID_index)
    # X_train, y_train = internal_X, internal_y

    X_test, y_test = internal_X.iloc[PID_index:PID_index +
                                     1], internal_y.iloc[PID_index:PID_index + 1]
    print('PID:', PID)
    print('PID_index:', PID_index)
    print(X_test)

    # 7.2 Split dataset to tr (80%) and val (20%) random seed diff
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=PID, shuffle=True)

    Train_Dataset, y_train_Dataset = bootstrap_data(10, X_tr, y_tr, 210, PID)
    Train_Dataset_merged, y_train_Dataset_merged = merge_bootstrap_data(
        Train_Dataset, y_train_Dataset)

    VAL_SIZE = 10
    VAL_DATASET, Y_VAL_DATASET = bootstrap_data(
        VAL_SIZE, X_val, y_val, 50, PID)

    loaded_model = gen_stacking_model(
        X_train, y_train)
    stacking_result = loaded_model.predict(X_test)
    print('stacking_result:', stacking_result)
    # 根據ground_truth & Meta Learner 調節
    if stacking_result == 0:
        rule_1, rule_2 = 0, 1  # ground_truth: N, Meta: N
    else:
        rule_1, rule_2 = 1, 0  # ground_truth: I, Meta: I

    # In[11]: Randomly generate random forest and candidate tree
    explainers, tree_candidates = getCandidate(Train_Dataset_merged, y_train_Dataset_merged,
                                               X_test, loaded_model,
                                               Explainer_depth, explainers_counter)
    # 10.2 Calculate the fidelity by explain_i
    # explain_i generate the tree_candidates[explain_i]

    for explain_i in list(explainers.keys()):
        start_tree_candidate = time.time()
        VAL_list = []
        rules = []
        top_n_rank = {}
        top_n_tree_idx = {}
        stack_pred_for_patient = loaded_model.predict(X_test)
        print('Num of consistent trees:', len(tree_candidates[explain_i]))
        for val_df, y_val_df in zip(VAL_DATASET, Y_VAL_DATASET):
            tree_index_list = []
            ACC_list = []
            ACC_G_list = []

            for tree_idx in tree_candidates[explain_i]:
                tree_pred_for_patient = explainers[explain_i].estimators_[
                    tree_idx].predict(X_test)
                tree_pred = explainers[explain_i].estimators_[
                    tree_idx].predict(val_df)
                stack_pred = loaded_model.predict(val_df)

                if tree_pred_for_patient == stack_pred_for_patient:
                    tree_index_list.append(tree_idx)
                    ACC_list.append(accuracy_score(stack_pred, tree_pred))
            VAL_list.append(ACC_list)

        fidelity_scores = np.array(VAL_list).reshape(
            VAL_SIZE, -1).mean(axis=0)
        rank = np.argsort(-1 * fidelity_scores)
        top_n_rank[explain_i] = fidelity_scores[rank][:top_num]
        top_n_tree_idx[explain_i] = np.array(
            tree_candidates[explain_i])[rank][:top_num]
        consistent_tree_len = list()
        consistent_tree_len.append(len(tree_index_list))
        consistent_tree_len_all.append(len(tree_index_list))
        # 10.3 Enumerate the decision path of the explain_i
        res_combined = []
        for idx in top_n_tree_idx[explain_i]:
            res = interpret(X_test, explainers[explain_i].estimators_[
                idx], feature_selection2)
            rule = " and ".join([" ".join([str(w_) for w_ in r_])
                                 for r_ in res['info']])
            rules.append(rule)
            res_combined = res_combined + \
                [" ".join([str(w_) for w_ in r_]) for r_ in res['info']]

        rules_ = fix_dp(rules)

        print('rules_list_ini:')
        print(rules_)

        CONDITIOS = rules_
        AV_FIDELITYS = []

        fidelity = []
        for condition in CONDITIOS:
            for val_df in VAL_DATASET:
                stack_pred = loaded_model.predict(val_df)
                val_df = change_column_name(val_df)
                merge_pred = np.where(
                    val_df.eval(condition), rule_1, rule_2)
                fidelity.append(accuracy_score(stack_pred, merge_pred))
            AV_FIDELITYS.append(round(np.mean(fidelity), 3))
        end_tree_candidate = time.time()
        tree_candiate_time = end_tree_candidate - start_tree_candidate
        tree_candiate_time_list.append(tree_candiate_time)
        print("tree candidate time = {}".format(tree_candiate_time))
        avg = np.mean(AV_FIDELITYS)
        print(avg)

        final_singleton = dp_to_singleton(rules_)
        print('final_singleton:', final_singleton)
        rule_str = ' | '.join(str(item) for item in rules_)
        dp_to_json(rule_str, final_singleton, PID)
        X_test_ = X_test_rename(X_test)
        singleton_relax, singleton_strict = singleton_opt(
            final_singleton, stacking_result, X_test_, top_num, Explainer_depth)
        print('singleton_relax:', singleton_relax)
        print('singleton_strict:', singleton_strict)
        start_kmap = time.time()
        rule_str_relax, final_singleton_relax = espresso_simplify(
            final_singleton, singleton_relax, rules_, PID)
        rule_str_strict, final_singleton_strict = espresso_simplify(
            final_singleton, singleton_strict, rules_, PID)
        print('rule_str_strict:', rule_str_strict)
        print('final_singleton_strict:', final_singleton_strict)
        dp_to_json_simplified(rule_str_strict, final_singleton_strict, PID)
        end_kmap = time.time()
        kmap_time = end_kmap - start_kmap
        kmap_time_list.append(kmap_time)


if __name__ == "__main__":
    run_ok = [121, 271, 331, 531, 541, 581, 631, 671, 1101, 1121, 1291, 1311, 1331, 1721, 1861, 2351, 2441, 2951, 3211, 3621, 3661, 3671, 3971, 4111, 4621, 5221, 5291, 5391, 5541, 5631, 5761, 5841, 5991, 6091, 6101, 6121, 6181, 6411, 6751, 6971, 7011, 7031, 7331, 7361, 7451, 7551, 7611, 7831, 8021, 8061, 8131, 8201, 8381, 8491, 8511, 8601, 8631, 8641, 8691, 8841, 8901, 9011, 9031, 9131, 9201, 9301, 9341, 9391, 9481, 9991, 10161, 10191, 10221, 11121, 11301, 12401, 12601, 12691, 12701, 42, 62, 1302, 1312, 1882, 1912, 2102, 2502, 2892, 4562, 5082, 6012, 6142, 6182, 6512, 6582, 6672, 6852, 6912, 7312, 7332, 7482, 8232, 8782, 8792, 8872, 8942, 11732, 12242, 12632, 12712, 12752, 12942, 13402, 13702, 13862, 14042, 14512, 14722, 16172, 16532, 16962, 17512, 17702, 17722, 17762, 18112, 18502, 18722, 18782, 18832, 18912, 18932, 19122, 19162, 19242, 19282, 19262, 19322, 19332, 19342, 19352, 19392, 20022, 20142, 20152, 20172, 20182, 20532, 20652, 20692, 20732, 20992, 21242, 21312, 21612, 21882, 22472, 22642, 22702, 23012, 23132,
              23142, 23152, 23172, 23182, 23202, 23212, 23222, 23242, 23292, 23362, 23372, 23452, 23512, 23522, 23602, 23632, 23642, 23672, 23392, 23682, 23742, 23812, 23852, 23892, 23952, 23982, 24002, 24032, 24042, 24082, 24092, 24102, 24112, 24162, 24182, 23802, 23862, 24302, 24352, 24362, 24392, 24402, 24062, 24462, 24472, 24152, 21802, 24372, 23752, 24122, 151, 171, 231, 491, 591, 681, 1041, 1081, 1091, 1181, 1191, 1201, 1301, 1411, 1851, 2001, 2151, 2321, 2841, 2851, 2971, 3141, 3391, 3431, 3501, 3521, 3871, 3881, 4481, 4631, 4701, 5071, 5251, 5461, 5521, 5911, 6031, 6141, 6171, 6381, 6641, 6661, 6711, 6721, 6731, 6771, 6781, 6811, 6871, 6951, 7171,
              7261, 9421, 9451, 10381, 12271, 12501, 12751, 12781, 2591, 3601, 4451, 6741, 6941, 7911, 9041, 9051, 12331, 12721, 551, 1071, 1261, 1471, 1741, 1941, 2731, 5281, 5331, 5811, 6071, 6291, 6391, 7751, 8701, 8861, 9771, 11281, 1371, 1551, 1931, 2081, 2211, 2511, 2601, 2931, 3001, 3081, 3551, 4441, 4681, 4941, 4981, 5021, 5481, 5921, 6001, 6021, 6131, 6261, 11291, 2401]
    run_id = [24422, 8841]
    top_num, Explainer_depth = 10, 1
    for i in range(len(run_id)):
        # if run_id[i] == 23752 or run_id[i] == 8841 or run_id[i] == 24422:
        #     continue
        # else:
        run_test(run_id[i], top_num, Explainer_depth)
