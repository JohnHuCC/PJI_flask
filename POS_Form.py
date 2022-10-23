#!/usr/bin/env python
###
# @1 原始資料來源: dnf_case4_2.py
# features: 將 fidelity_DecisionPath 產生的候選 decision path 轉換成 POS Form
###
# input: Candidate Decision Path
# outpu: POS form rules
#

# In[15]: Import Library
# 引用適當的 Library
import numpy as np
from sympy import Symbol
from sympy import symbols
from sympy.logic import simplify_logic, SOPform
from sympy.logic.boolalg import to_dnf
import itertools
import pandas as pd
# from sympy.logic import simplify_logic


# In[16]: Declare the function for clasification with ICM & nonICM
# 原始推導流程參閱掃描筆記檔案
def Non_ICM_1(x):
    return {
        'mu(I)_delta_mu(N)': ['<=', 'delta', 'mu(N)'],
        'mu(I)_mu(N)_delta': ['NotUsed'],
        'delta_mu(N)_mu(I)': ['NotUsed'],
        'delta_mu(I)_mu(N)': ['<=', 'delta', 'mu(N)'],
        'mu(N)_delta_mu(I)': ['>=', 'mu(N)', 'delta'],
        'mu(N)_mu(I)_delta': ['>=', 'mu(N)', 'delta'],

    }.get(x, 'Error')


def Non_ICM_0(x):
    return {
        'delta_mu(N)_mu(I)': ['<=', 'delta', 'mu(I)'],
        'delta_mu(I)_mu(N)': ['NotUsed'],
        'mu(N)_mu(I)_delta': ['NotUsed'],
        'mu(N)_delta_mu(I)': ['<=', 'delta', 'mu(I)'],
        'mu(I)_delta_mu(N)': ['>=', 'mu(I)', 'delta'],
        'mu(I)_mu(N)_delta': ['>=', 'mu(I)', 'delta'],
    }.get(x, 'Error')


def ICM_1(x):
    return {

        'ICM_Q_delta': ['>=', 'ICM', 'delta'],
        'ICM_delta_Q': ['>=', 'ICM', 'delta'],
        'Q_ICM_delta': ['>=', 'ICM', 'delta'],
        'Q_delta_ICM': ['>=', 'Q', 'delta'],
        'delta_ICM_Q': ['NotUsed'],
        'delta_Q_ICM': ['NotUsed'],

    }.get(x, 'Error')


def ICM_0(x):
    return {
        'delta_Q_ICM': ['<=', 'delta', 'ICM'],
        'delta_ICM_Q': ['<=', 'delta', 'ICM'],
        'ICM_delta_Q': ['<=', 'delta', 'Q'],
        'ICM_Q_delta': ['NotUsed'],
        'Q_ICM_delta': ['NotUsed'],
        'Q_delta_ICM': ['<=', 'delta', 'ICM'],

    }.get(x, 'Error')


# In[17]: Get the upperbound & lowerbound of singletons
# 注意：此處須匯入兩個檔案: Non2018ICM.xlsx, 2018ICM.xlsx
def singleton_opt(X_test):
    non2018ICM = pd.read_excel("./Non2018ICM.xlsx")
    _2018ICM = pd.read_excel("./2018ICM.xlsx")

    _2018ICM_ = _2018ICM[['variable', 'threshold']]
    non2018ICM_ = non2018ICM[['variable', 'mu(N)', 'mu(I)']]

    # non2018ICM_.iloc[0, 1]

    X_test_ = X_test.rename(
        columns={'Serum WBC ': 'Serum_WBC_',
                 'Segment (%)': 'Segment',
                 'HGB': 'Hb',
                 'P.T': 'P_T',
                 'Total CCI': 'Total_CCI',
                 'Total Elixhauser Groups per record': 'Total_Elixhauser_Groups_per_record',
                 'Serum CRP': 'Serum_CRP',
                 'Serum ESR': 'Serum_ESR',
                 'Synovial WBC': 'Synovial_WBC',

                 }
    )

    decision_list = {}

    for val in list(final_singleton):
        NotUsed_flag = 0

        if ("==") in val:
            continue
        elif ("<=") in val:
            variable = val[:val.find("<=")-1]
            operator = val[val.find("<="):val.find("<=")+2]
            Q = val[val.find("<=")+3:]
        elif ("< ") in val:
            variable = val[:val.find("<")-1]
            operator = val[val.find("<"):val.find("<")+1]
            Q = val[val.find("<")+2:]
        elif (">=") in val:
            variable = val[:val.find(">=")-1]
            operator = val[val.find(">="):val.find(">=")+2]
            Q = val[val.find(">=")+3:]
        elif ("> ") in val:
            variable = val[:val.find(">")-1]
            operator = val[val.find(">"):val.find(">")+1]
            Q = val[val.find(">")+2:]

        # print (variable, operator, Q)

        try:
            if variable in list(non2018ICM_['variable']):  # non2018ICM List
                index = list(non2018ICM_['variable']).index(variable)
                mu_N = non2018ICM_['mu(N)'][index]
                mu_I = non2018ICM_['mu(I)'][index]
                delta = float(X_test_[variable])

                nonICM = {"mu(N)": mu_N, "mu(I)": mu_I, "delta": delta}
                #nonICM_Sorting = (sorted(nonICM.items(), key=lambda x:x[1]))
                nonICM_Sorting = sorted(nonICM, key=nonICM.get)
                concate_nonICM_Sorting = '_'.join(nonICM_Sorting)

                if explainer.predict(X_test) == 0:   # Meta(I) = 0
                    result = Non_ICM_0(concate_nonICM_Sorting)
                    if len(result) == 3:
                        operator, _lower_bound, _upper_bound = result
                        lower_bound, upper_bound = nonICM[_lower_bound], nonICM[_upper_bound]

                    else:  # 不考慮
                        NotUsed_flag = 1

                elif explainer.predict(X_test) == 1:  # Meta(I) = 1
                    result = Non_ICM_1(concate_nonICM_Sorting)
                    if len(result) == 3:
                        operator, _lower_bound, _upper_bound = result
                        lower_bound, upper_bound = nonICM[_lower_bound], nonICM[_upper_bound]

                    else:  # 不考慮
                        NotUsed_flag = 1

            elif variable in list(_2018ICM_['variable']):  # 2018ICM List
                index = list(_2018ICM_['variable']).index(variable)
                ICM_threshold = _2018ICM_['threshold'][index]
                delta = float(X_test_[variable])

                ICM = {"ICM": ICM_threshold, "Q": float(Q), "delta": delta}
                ICM_Sorting = sorted(ICM, key=ICM.get)
                concate_ICM_Sorting = '_'.join(ICM_Sorting)

                if explainer.predict(X_test) == 0:   # Meta(I) = 0
                    result = ICM_0(concate_ICM_Sorting)
                    if len(result) == 3:
                        operator, _lower_bound, _upper_bound = result
                        lower_bound, upper_bound = ICM[_lower_bound], ICM[_upper_bound]

                    else:  # 不考慮
                        NotUsed_flag = 1

                elif explainer.predict(X_test) == 1:   # Meta(I) = 1
                    result = ICM_1(concate_ICM_Sorting)
                    if len(result) == 3:
                        operator, _lower_bound, _upper_bound = result
                        lower_bound, upper_bound = ICM[_lower_bound], ICM[_upper_bound]

                    else:  # 不考慮
                        NotUsed_flag = 1

            if NotUsed_flag == 1:
                decision_list[val] = result
            else:
                decision_list[val] = [
                    variable, operator, lower_bound, upper_bound]

            print(val, decision_list[val])
        except:
            print("You got an Exception.")

    return decision_list


# In[18]: Transfer the decision_path to POS Form
def transPOSForm(Candidate_DecisionPath):

    # In[19]: 讀取 rules_list, data
    # data = [
    # # '(Serum_ESR > 48.5 and Segment > 64.5) | (Serum_ESR > 49.0 and Serum_CRP > 13.5 and APTT > 29.5 and Synovial_PMN > 52.0) | (Serum_ESR > 39.5 and Synovial_PMN > 69.0 and Age > 32.5 and Synovial_WBC > 367.5) | (Serum_ESR > 55.0 and P_T > 5.0 and Age > 33.0 and Hb > 8.5) | (Serum_ESR > 41.0 and Synovial_PMN > 41.5 and P_T > 5.0 and APTT > 23.5 and Age > 31.5 and Segment > 64.5)',
    # '(Positive_Histology == True and Serum_CRP > 3.0 and Hb <= 13.5) | (Positive_Histology == True and Serum_CRP > 2.5 and APTT > 23.5 and Hb <= 13.5) | (Serum_ESR > 39.5 and APTT > 23.5 and Serum_CRP > 4.0 and Positive_Histology == True) | (Positive_Histology == True and Serum_CRP > 3.0) | (Positive_Histology == True and P_T > 5.0 and Segment > 65.5) | (Serum_ESR > 28.5 and Synovial_WBC > 2483.5 and APTT > 26.5)'
    # ]
    data = Candidate_DecisionPath
    res_combined = []
    # In[20] 拆解所有的 decision path 元素
    ## input: all_path
    ## output: all_singleton
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


# In[26]: Call the transPOSForm func.
POS_Form, final_singleton = transPOSForm(rules_list)

## 20210531 ##
# 套用演算法分類 ICM & non_ICM 分類
# 原始推導資料請參閱文件
## singleton_opt == parser.py
# In[27]: Call the singleton_opt function
opt_decision_list = singleton_opt(X_test)


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
