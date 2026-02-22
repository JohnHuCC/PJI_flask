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
sys.modules['sklearn.externals.six'] = six


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
        'delta_mu(N)_mu(I)': ['<=', 'delta', 'mu(I)'],  # sheet 9-1
        'delta_mu(I)_mu(N)': ['NotUsed'],  # sheet 9-2 11-1 11-2
        'mu(N)_mu(I)_delta': ['NotUsed'],  # sheet 9-3
        'mu(N)_delta_mu(I)': ['<=', 'delta', 'mu(I)'],  # sheet 10-1 11-3 11-4
        'mu(I)_delta_mu(N)': ['>=', 'mu(I)', 'delta'],  # sheet 10-2
        'mu(I)_mu(N)_delta': ['>=', 'mu(I)', 'delta'],  # sheet 10-3
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


X_res_test = pd.read_csv('PJI_Dataset/New_data_x_test.csv', encoding='utf-8')
internal_X = pd.read_csv('PJI_Dataset/internal_x.csv', encoding='utf-8')
internal_y = pd.read_csv('PJI_Dataset/internal_y.csv', encoding='utf-8')
internal_X_all = pd.read_csv(
    'PJI_Dataset/internal_x_all.csv', encoding='utf-8')
internal_y_all = pd.read_csv(
    'PJI_Dataset/internal_y_all.csv', encoding='utf-8')
Explainer_depth = 12
X_res = pd.read_csv('PJI_Dataset/New_data_x.csv', encoding='utf-8')
y_res = pd.read_csv('PJI_Dataset/New_data_y.csv', encoding='utf-8')
X_res_test = pd.read_csv('PJI_Dataset/New_data_x_test.csv', encoding='utf-8')
X_tr, X_val, y_tr, y_val = train_test_split(
    X_res, y_res, test_size=0.3, random_state=666, shuffle=True)
# 10.1 Prepare the val_df (size = 10) for calculate fidelity scores
VAL_SIZE = 1
VAL_DATASET = []
Y_VAL_DATASET = []
for i in range(VAL_SIZE):
    VAL_DATASET.append(resample(X_val, n_samples=15,
                                replace=False, random_state=i))
    Y_VAL_DATASET.append(resample(y_val, n_samples=15,
                                  replace=False, random_state=i))

PID = 1721
no_group = list(X_res_test['No.Group'])
PID_index = 2 + no_group.index(PID)
stacking_model = joblib.load('Stacking_model/Stacking_model_new_data')
personal_result = stacking_model.predict(X_res)[PID_index]

if personal_result == 0:
    rule_1, rule_2 = 0, 1  # ground_truth: N, Meta: N
else:
    rule_1, rule_2 = 1, 0  # ground_truth: I, Meta: I


def grid_search(final_rule_list):
    for i in range(0, len(final_rule_list), 2):
        print(final_rule_list[i], final_rule_list[i+1])

        NotUsed_flag = 0
        if ("==") in final_rule_list[i]:
            continue
        elif ("<=") in final_rule_list[i]:
            variable = final_rule_list[i][:final_rule_list[i].find("<=")-1]
            operator = final_rule_list[i][final_rule_list[i].find(
                "<="):final_rule_list[i].find("<=")+2]
            upper_bound = final_rule_list[i][final_rule_list[i].find("<=")+3:]
        elif ("< ") in final_rule_list[i]:
            variable = final_rule_list[i][:final_rule_list[i].find("<")-1]
            operator = final_rule_list[i][final_rule_list[i].find(
                "<"):final_rule_list[i].find("<")+1]
            upper_bound = final_rule_list[i][final_rule_list[i].find("<")+2:]
        elif (">=") in final_rule_list[i]:
            variable = final_rule_list[i][:final_rule_list[i].find(">=")-1]
            operator = final_rule_list[i][final_rule_list[i].find(
                ">="):final_rule_list[i].find(">=")+2]
            lower_bound = final_rule_list[i][final_rule_list[i].find(">=")+3:]
        elif ("> ") in final_rule_list[i]:
            variable = final_rule_list[i][:final_rule_list[i].find(">")-1]
            operator = final_rule_list[i][final_rule_list[i].find(
                ">"):final_rule_list[i].find(">")+1]
            lower_bound = final_rule_list[i][final_rule_list[i].find(">")+2:]

        if ("==") in final_rule_list[i+1]:
            continue
        elif ("<=") in final_rule_list[i+1]:
            variable = final_rule_list[i+1][:final_rule_list[i+1].find("<=")-1]
            operator = final_rule_list[i+1][final_rule_list[i +
                                                            1].find("<="):final_rule_list[i+1].find("<=")+2]
            upper_bound = final_rule_list[i +
                                          1][final_rule_list[i+1].find("<=")+3:]
        elif ("< ") in final_rule_list[i+1]:
            variable = final_rule_list[i+1][:final_rule_list[i+1].find("<")-1]
            operator = final_rule_list[i+1][final_rule_list[i +
                                                            1].find("<"):final_rule_list[i+1].find("<")+1]
            upper_bound = final_rule_list[i +
                                          1][final_rule_list[i+1].find("<")+2:]

        lower_bound_ = float(lower_bound)
        upper_bound_ = float(upper_bound)
        final_val = 0
        if lower_bound_ == upper_bound_:
            final_val = upper_bound_
        else:
            for i in np.arange(float(lower_bound), float(upper_bound), 0.5):
                fidelity = []
                rules_ = variable + " <= " + str(i)
                print(rules_)
                for val_df in VAL_DATASET:
                    stack_pred = stacking_model.predict(val_df)
                    val_df = val_df.rename(columns=d_path)
                    merge_pred = np.where(
                        val_df.eval(rules_), rule_1, rule_2)
                    fidelity.append(accuracy_score(stack_pred, merge_pred))
                print(fidelity)


def simplified_singleton_opt(PID, final_singleton):
    X_res_test = pd.read_csv(
        'PJI_Dataset/New_data_x_test.csv', encoding='utf-8')
    no_group = list(X_res_test['No.Group'])
    PID_index = no_group.index(PID)

    New_data_X = pd.read_csv('PJI_Dataset/New_data_x.csv', encoding='utf-8')
    New_data_y = pd.read_csv('PJI_Dataset/New_data_y.csv', encoding='utf-8')

    X_train, y_train = internal_X, internal_y
    X_test, y_test = New_data_X.iloc[PID_index:PID_index +
                                     1], New_data_y.iloc[PID_index:PID_index + 1]
    explainer = RandomForestClassifier(
        max_depth=Explainer_depth, n_estimators=100, random_state=123)
    explainer.fit(X_train.values, y_train.values)

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

    ICM_threshold = json.load(open("auto_boundary/ICM_threshold.json"))
    _2018ICM_ = pd.DataFrame(list(ICM_threshold.items()),
                             columns=['variable', 'threshold'])
    # non2018ICM_ = non2018ICM[['variable', 'mu(N)', 'mu(I)']]
    non2018ICM_I = json.load(open("auto_boundary/nonICM_bound_I.json"))
    non2018ICM_N = json.load(open("auto_boundary/nonICM_bound_N.json"))
    non2018ICM_N = pd.DataFrame(list(non2018ICM_N.items()),
                                columns=['variable', 'mu(N)'])
    non2018ICM_I = pd.DataFrame(list(non2018ICM_I.items()),
                                columns=['variable', 'mu(I)'])
    non2018ICM_ = non2018ICM_N.merge(non2018ICM_I, how='outer')

    res_combined = []
    rules_list_ = []
    decision_list = {}
    for element in final_singleton:
        element = element.replace(') | (', ' and ')
        element = element.replace('(', '')
        element = element.replace(')', '')
        element = element.replace('&', 'and')

        res_combined = res_combined + \
            [x.strip(' ') for x in element.split('and')]

    final_singleton_set = set()
    for atom in res_combined:
        final_singleton_set.add(atom)

    for val in list(final_singleton_set):
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

        if variable in list(non2018ICM_['variable']):  # non2018ICM List
            index = list(non2018ICM_['variable']).index(variable)
            mu_N = non2018ICM_['mu(N)'][index]
            mu_I = non2018ICM_['mu(I)'][index]
            delta = float(X_test_[variable])

            nonICM = {"mu(N)": mu_N, "mu(I)": mu_I, "delta": delta}
            # nonICM_Sorting = (sorted(nonICM.items(), key=lambda x:x[1]))
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

        if decision_list[val][0] == 'NotUsed':
            rules_list_.append(str(val))
        elif decision_list[val][1] == '>=':
            rules_list_.append(
                str(decision_list[val][0])+' '+str(decision_list[val][1])+' '+str(decision_list[val][2]))
            rules_list_.append(
                str(decision_list[val][0])+' <= '+str(decision_list[val][3]))
        elif decision_list[val][1] == '<=':
            rules_list_.append(
                str(decision_list[val][0])+' '+str(decision_list[val][1])+' '+str(decision_list[val][3]))
            rules_list_.append(
                str(decision_list[val][0])+' >= '+str(decision_list[val][2]))
        else:
            print('exception:')
            print(decision_list[val])
    return (rules_list_)


def transPOSForm_ini(Candidate_DecisionPath):
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
    return res_combined


def singleton_opt(PID, final_singleton_ini):
    X_res_test = pd.read_csv(
        'PJI_Dataset/New_data_x_test.csv', encoding='utf-8')
    no_group = list(X_res_test['No.Group'])
    PID_index = no_group.index(PID)

    New_data_X = pd.read_csv('PJI_Dataset/New_data_x.csv', encoding='utf-8')
    New_data_y = pd.read_csv('PJI_Dataset/New_data_y.csv', encoding='utf-8')

    X_train, y_train = internal_X, internal_y
    X_test, y_test = New_data_X.iloc[PID_index:PID_index +
                                     1], New_data_y.iloc[PID_index:PID_index + 1]
    explainer = RandomForestClassifier(
        max_depth=Explainer_depth, n_estimators=100, random_state=123)
    explainer.fit(X_train.values, y_train.values)
    rules_list_ = []
    rules_list_used = []
    ICM_threshold = json.load(open("auto_boundary/ICM_threshold.json"))
    _2018ICM_ = pd.DataFrame(list(ICM_threshold.items()),
                             columns=['variable', 'threshold'])
    # non2018ICM_ = non2018ICM[['variable', 'mu(N)', 'mu(I)']]
    non2018ICM_I = json.load(open("auto_boundary/nonICM_bound_I.json"))
    non2018ICM_N = json.load(open("auto_boundary/nonICM_bound_N.json"))
    non2018ICM_N = pd.DataFrame(list(non2018ICM_N.items()),
                                columns=['variable', 'mu(N)'])
    non2018ICM_I = pd.DataFrame(list(non2018ICM_I.items()),
                                columns=['variable', 'mu(I)'])
    non2018ICM_ = non2018ICM_N.merge(non2018ICM_I, how='outer')

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
    decision_list = {}
    for val in list(final_singleton_ini):
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

        try:
            if variable in list(non2018ICM_['variable']):  # non2018ICM List
                # print('non_ICM!')
                index = list(non2018ICM_['variable']).index(variable)
                mu_N = non2018ICM_['mu(N)'][index]
                mu_I = non2018ICM_['mu(I)'][index]
                delta = float(X_test_[variable])

                nonICM = {"mu(N)": mu_N, "mu(I)": mu_I, "delta": delta}
                # nonICM_Sorting = (sorted(nonICM.items(), key=lambda x:x[1]))
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

            if decision_list[val][0] == 'NotUsed':
                rules_list_.append(str(val))
            else:
                rules_list_used.append(str(val))

        except Exception as e:
            print("You got an Exception.", str(e))

    return rules_list_


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


def two_list_Diff(list_a, list_b):
    return (list(set(list_a).symmetric_difference(set(list_b))))


def rule_operator_by_stacking(stacking_result, feature):
    if stacking_result == 1:
        rule_operator = {
            'two_positive_culture': '==',
            'APTT': '>=',
            'ASA_2': '<=',
            'Age': '<=',
            'Hb': '<=',
            'P_T': '>=',
            'PLATELET': '>=',
            'Positive_Histology': '==',
            'Surgery': '<=',
            'Purulence': '==',
            'Segment': '>=',
            'Serum_CRP': '>=',
            'Serum_ESR': '>=',
            'Serum_WBC_': '>=',
            'Single_Positive_culture': '==',
            'Synovial_WBC': '>=',
            'Synovial_PMN': '>=',
            'Total_CCI': '>=',
            'Total_Elixhauser_Groups_per_record': '>='}
        return rule_operator[feature]
    else:
        rule_operator = {
            'two_positive_culture': '==',
            'APTT': '<=',
            'ASA_2': '>=',
            'Age': '>=',
            'Hb': '>=',
            'P_T': '<=',
            'PLATELET': '<=',
            'Positive_Histology': '==',
            'Surgery': '>=',
            'Purulence': '==',
            'Segment': '<=',
            'Serum_CRP': '<=',
            'Serum_ESR': '<=',
            'Serum_WBC_': '<=',
            'Single_Positive_culture': '==',
            'Synovial_WBC': '<=',
            'Synovial_PMN': '<=',
            'Total_CCI': '<=',
            'Total_Elixhauser_Groups_per_record': '<='}
        return rule_operator[feature]


def find_delta(rules_list, PID, singleton_num, top):
    # stacking_model = joblib.load('PJI_model/Stacking_model_'+str(PID))
    # personal_result = stacking_model.predict(internal_X_all)[PID]
    X_res_test = pd.read_csv(
        'PJI_Dataset/New_data_x_test.csv', encoding='utf-8')
    no_group = list(X_res_test['No.Group'])
    PID_index = no_group.index(PID)

    New_data_X = pd.read_csv('PJI_Dataset/New_data_x.csv', encoding='utf-8')
    New_data_y = pd.read_csv('PJI_Dataset/New_data_y.csv', encoding='utf-8')

    X_test, y_test = New_data_X.iloc[PID_index:PID_index +
                                     1], New_data_y.iloc[PID_index:PID_index + 1]
    stacking_model = joblib.load('Stacking_model/Stacking_model_new_data')
    personal_result = stacking_model.predict(X_test)
    if personal_result == 0:
        rule_1, rule_2 = 0, 1  # ground_truth: N, Meta: N
    else:
        rule_1, rule_2 = 1, 0  # ground_truth: I, Meta: I
    res_combined = []
    # In[20] 拆解所有的 decision path 元素
    # input: all_path
    # output: all_singleton
    _singleton_list = list()
    rule_dict = {}
    # print(rules_list)
    for i, (element) in enumerate(rules_list):
        res_combined = res_combined + \
            [x.strip(' ') for x in element.split('and')]

        _singleton_set = set()
        for atom in res_combined:
            _singleton_set.add(atom)

        for atom in _singleton_set:
            _singleton_list.append(atom)

    final_singleton_ini = transPOSForm_ini(rules_list)
    singleton_deleted = singleton_opt(PID, final_singleton_ini)
    print('singleton deleted:')
    print(singleton_deleted)
    print('')
    _singleton_list = two_list_Diff(_singleton_list, singleton_deleted)
    print('_singleton_list_used:')
    print(_singleton_list)
    print('')

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

    singleton_dict = {}
    max_fid_dict = dict.fromkeys(singleton_f, 0)
    max_fid_dict_val = dict.fromkeys(singleton_f, 0)
    max_fid_dict_fid = dict.fromkeys(singleton_f, 0)
    second_max_fid_dict = dict.fromkeys(singleton_f, None)
    second_max_fid_dict_fid = dict.fromkeys(singleton_f, 0)

    AVG_FIDELITYS = []
    for i, condition in enumerate(_singleton_list):
        fidelity = []
        for val_df in VAL_DATASET:
            stack_pred = stacking_model.predict(val_df)
            val_df = val_df.rename(columns=d_path)
            merge_pred = np.where(
                val_df.eval(condition), rule_1, rule_2)
            fidelity.append(accuracy_score(stack_pred, merge_pred))
        AVG_FIDELITYS.append(round(np.mean(fidelity), 3))
        AVG_FIDELITYS_ = round(np.mean(fidelity), 3)

        if condition.find("<=") > 0:
            key = condition[:condition.find("<=")-1]
            if key not in singleton_dict:
                singleton_dict.setdefault(key, set()).add(str(condition)+' : ' +
                                                          str(round(np.mean(fidelity), 3)))
            else:
                singleton_dict[key].add(str(condition)+' : ' +
                                        str(round(np.mean(fidelity), 3)))
            if max_fid_dict_fid.get(key) < AVG_FIDELITYS_:
                max_fid_dict[key] = i
                max_fid_dict_val[key] = condition
                max_fid_dict_fid[key] = AVG_FIDELITYS_
        elif condition.find("<") > 0:
            key = condition[:condition.find("<")-1]
            if key not in singleton_dict:
                singleton_dict.setdefault(key, set()).add(str(condition)+' : ' +
                                                          str(round(np.mean(fidelity), 3)))
            else:
                singleton_dict[key].add(str(condition)+' : ' +
                                        str(round(np.mean(fidelity), 3)))
            if max_fid_dict_fid.get(key) < AVG_FIDELITYS_:
                max_fid_dict[key] = i
                max_fid_dict_val[key] = condition
                max_fid_dict_fid[key] = AVG_FIDELITYS_
        elif condition.find("==") > 0:
            key = condition[:condition.find("==")-1]
            if key not in singleton_dict:
                singleton_dict.setdefault(key, set()).add(str(condition)+' : ' +
                                                          str(round(np.mean(fidelity), 3)))
            else:
                singleton_dict[key].add(str(condition)+' : ' +
                                        str(round(np.mean(fidelity), 3)))
            max_fid_dict[key] = i
            max_fid_dict_val[key] = condition
            max_fid_dict_fid[key] = AVG_FIDELITYS_
        elif condition.find(">=") > 0:
            key = condition[:condition.find(">=")-1]
            if key not in singleton_dict:
                singleton_dict.setdefault(key, set()).add(str(condition)+' : ' +
                                                          str(round(np.mean(fidelity), 3)))
            else:
                singleton_dict[key].add(str(condition)+' : ' +
                                        str(round(np.mean(fidelity), 3)))
            if max_fid_dict_fid.get(key) < AVG_FIDELITYS_:
                max_fid_dict[key] = i
                max_fid_dict_val[key] = condition
                max_fid_dict_fid[key] = AVG_FIDELITYS_
        elif condition.find(">") > 0:
            key = condition[:condition.find(">")-1]
            if key not in singleton_dict:
                singleton_dict.setdefault(key, set()).add(str(condition)+' : ' +
                                                          str(round(np.mean(fidelity), 3)))
            else:
                singleton_dict[key].add(str(condition)+' : ' +
                                        str(round(np.mean(fidelity), 3)))
            if max_fid_dict_fid.get(key) < AVG_FIDELITYS_:
                max_fid_dict[key] = i
                max_fid_dict_val[key] = condition
                max_fid_dict_fid[key] = AVG_FIDELITYS_
    print('All singletons fidelity:')
    print('')
    print(AVG_FIDELITYS)
    print('')
    print('max_fid_dict:')
    print(max_fid_dict)
    print('')
    print(max_fid_dict_fid)
    print('')
    # print(singleton_dict)
    # print('')
    for i, condition in enumerate(_singleton_list):
        if condition.find("<=") > 0:
            key = condition[:condition.find("<=")-1]
        elif condition.find("<") > 0:
            key = condition[:condition.find("<")-1]
        elif condition.find(">=") > 0:
            key = condition[:condition.find(">=")-1]
        elif condition.find(">") > 0:
            key = condition[:condition.find(">")-1]
        elif condition.find("==") > 0:
            key = condition[:condition.find("==")-1]

        second_max_gap = max_fid_dict_fid.get(key) - AVG_FIDELITYS[i]
        if second_max_gap >= 0 and second_max_fid_dict_fid.get(key) == 0 and max_fid_dict.get(key) != i:
            second_max_fid_dict[key] = i
            second_max_fid_dict_fid[key] = second_max_gap
        if second_max_gap >= 0 and second_max_gap < second_max_fid_dict_fid.get(key) and max_fid_dict.get(key) != i:
            second_max_fid_dict[key] = i
            second_max_fid_dict_fid[key] = second_max_gap
    _second_max_fid_dict = second_max_fid_dict.copy()
    for k, v in _second_max_fid_dict.items():
        if v == None:
            second_max_fid_dict.pop(k)
            second_max_fid_dict_fid.pop(k)

    print('second_max_fid_dict:')
    print(second_max_fid_dict)
    print('')
    print(second_max_fid_dict_fid)
    print('')
    singleton_num_gap = singleton_num - len(max_fid_dict)
    print('singleton_num_gap:')
    print(singleton_num_gap)
    second_max_choices = tuple()
    max_choices = tuple()
    if singleton_num_gap > 0 and len(second_max_fid_dict) > singleton_num_gap:
        second_max_choices = random.choices(
            list(second_max_fid_dict.items()), k=singleton_num_gap)
    elif singleton_num_gap <= 0:
        max_choices = random.sample(
            list(max_fid_dict.items()), singleton_num)
    second_max_choices_dict = dict((x, y) for x, y in second_max_choices)
    max_choices_dict = dict((x, y) for x, y in max_choices)
    max_choices = list()
    for key, val in max_choices_dict.items():
        max_choices.append(_singleton_list[val].strip())
    print('max_choices:')
    print(str(max_choices))
    print('second_max_choices:')
    print(second_max_choices_dict)

    if singleton_num_gap <= 0:
        for key, val in max_choices_dict.items():
            to_replace = _singleton_list[val].split(' ')[0]
            for i1, rules in enumerate(rules_list):
                each_rule_list = rules.split(' and ')
                for i2, each_rule in enumerate(each_rule_list):
                    if each_rule.startswith(to_replace):
                        each_rule_list[i2] = _singleton_list[val].strip()
                    elif each_rule_list[i2] not in max_choices:
                        each_rule_list.remove(each_rule_list[i2])
                rules_list[i1] = ' and '.join(each_rule_list)
    else:
        for key, val in max_fid_dict.items():
            to_replace = _singleton_list[val].split(' ')[0]
            for i1, rules in enumerate(rules_list):
                each_rule_list = rules.split(' and ')
                for i2, each_rule in enumerate(each_rule_list):
                    if each_rule.startswith(to_replace):
                        second_max_index = second_max_choices_dict.get(key)
                        if second_max_index != None:
                            if each_rule_list[i2] != _singleton_list[second_max_index].strip(
                            ):
                                each_rule_list[i2] = _singleton_list[val].strip(
                                )
                        else:
                            each_rule_list[i2] = _singleton_list[val].strip()
                rules_list[i1] = ' and '.join(each_rule_list)
    max_fid_item = list()
    second_max_fid_item = list()
    for key, val in max_fid_dict.items():
        max_fid_item.append(_singleton_list[val].strip())

    for key, val in second_max_choices_dict.items():
        second_max_fid_item.append(_singleton_list[val].strip())

    # 不考慮之singleton，在rule中刪除
    for i1, rules in enumerate(rules_list):
        each_rule_list = rules.split(' and ')
        each_rule_list_temp = rules.split(' and ')
        for i2, each_rule in enumerate(each_rule_list_temp):
            flag = 0
            for val in _singleton_list:
                if str(each_rule) == str(val):
                    flag = 1

            if flag == 0:
                each_rule_list.remove(each_rule)

        rules_list[i1] = ' and '.join(each_rule_list)
    rules_list = [x for x in rules_list if x != '']
    print('rule_deleted:', rules_list)
    res_combined_final = []
    # In[20] 拆解所有的 decision path 元素
    # input: all_path
    # output: all_singleton
    _singleton_list_final = list()
    rule_dict = {}
    # print(rules_list)
    for i, (element) in enumerate(rules_list):
        res_combined_final = res_combined_final + \
            [x.strip(' ') for x in element.split('and')]

        _singleton_set = set()
        for atom in res_combined:
            _singleton_set.add(atom)

        for atom in _singleton_set:
            _singleton_list_final.append(atom)
    # In[21]: 列舉 all_singleton 的 features by Set()
    singleton_f_final = set()
    for i, (val) in enumerate(_singleton_list):

        if val.find("<=") > 0:
            singleton_f_final.add(val[:val.find("<=")-1])

        elif val.find("<") > 0:
            singleton_f_final.add(val[:val.find("<")-1])

        elif val.find("==") > 0:
            singleton_f_final.add(val[:val.find("==")-1])

        elif val.find(">=") > 0:
            singleton_f_final.add(val[:val.find(">=")-1])

        elif val.find(">") > 0:
            singleton_f_final.add(val[:val.find(">")-1])

    eleMap = dict()
    for ele in _singleton_list_final:
        eleList = ele.split(' ')
        name = eleList[0]
        comp = eleList[1]
        if comp == '<=' or comp == '<':
            comp = '<='
        elif comp == '>=' or comp == '>':
            comp = '>='
        if comp == '==':
            val = eleList[2]
        else:
            val = float(eleList[2])
        if name not in eleMap:
            eleMap[name] = dict()
        if comp not in eleMap[name]:
            eleMap[name][comp] = val
        else:
            if comp == '<=' or comp == '<':
                eleMap[name][comp] = min(eleMap[name][comp], val)
            elif comp == '>=' or comp == '>':
                eleMap[name][comp] = max(eleMap[name][comp], val)
            else:
                eleMap[name][comp] = val

    print('singleton_dict_all_bound:')
    print(eleMap)
    print('True Class:', personal_result)
    print('')
    for i1, rules in enumerate(rules_list):
        each_rule_list = rules.split(' and ')
        for i2, each_rule in enumerate(each_rule_list):
            each_rule_item = each_rule.split(' ')
            name = each_rule_item[0]
            best_comp = rule_operator_by_stacking(personal_result, name)

            comp = each_rule_item[1]
            if comp == '<=' or comp == '<':
                key_error_comp = '<='
            elif comp == '>=' or comp == '>':
                key_error_comp = '>='
            if comp == '==':
                val = each_rule_item[2]
            else:
                try:
                    comp = best_comp
                    val = eleMap[name][best_comp]
                except KeyError as ke:
                    comp = key_error_comp
                    val = eleMap[name][comp]
            each_rule_list[i2] = name + ' ' + comp + \
                ' ' + str(val)
        rules_list[i1] = ' and '.join(each_rule_list)

    print('rule_wide_range:')
    print(rules_list)
    print('')
    rules_list = [rules.strip()
                  for rules in rules_list if rules.strip() != '']
    with open('final_result/final_result_v3_top'+str(top)+'_mix_v1.txt', 'a') as f:
        f.write('singleton_num_gap:')
        f.write(str(singleton_num_gap))
        f.write('\n')
        f.write('max_fid_item:')
        f.write(str(max_fid_item))
        f.write('\n')
        f.write('max_fid_item num:')
        f.write(str(len(max_fid_item)))
        f.write('\n')
        f.write('second_max_fid_item:')
        f.write(str(second_max_fid_item))
        f.write('\n')
        f.write('rule_top_fidelity:')
        f.write(str(rules_list))
        f.write('\n')
    print('rule_top_fidelity:')
    print(rules_list)
    print('')
    return rules_list
