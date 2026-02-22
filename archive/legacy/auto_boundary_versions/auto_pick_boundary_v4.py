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
sys.modules['sklearn.externals.six'] = six
rules_list = ['APTT > 28.5 and Serum_WBC_ <= 9.5 and Synovial_PMN > 42.5 and Serum_ESR > 49.0 and Serum_CRP > 5.5', 'Serum_ESR > 41.0 and Serum_CRP > 62.0', 'Synovial_PMN > 57.5 and Serum_ESR > 29.5 and Hb > 5.0 and APTT > 24.5 and Age > 31.5 and Serum_CRP > 44.5 and Synovial_PMN > 66.5', 'Serum_CRP > 11.5 and Serum_ESR > 41.0 and Age <= 71.5 and Segment > 58.5', 'Serum_ESR > 41.0 and Serum_WBC_ > 8.5 and Synovial_WBC > 470.0', 'Serum_CRP > 13.5 and PLATELET > 271.0 and Synovial_PMN > 41.5',
              'Serum_CRP > 25.0 and Synovial_PMN > 56.5 and Serum_ESR > 30.5 and Hb > 4.5', 'Serum_ESR > 28.0 and Positive_Histology == True and Hb <= 13.5', 'Synovial_PMN > 45.5 and Positive_Histology == True and Synovial_PMN > 63.0', 'Positive_Histology == True and Synovial_PMN > 63.0', 'Positive_Histology == True and Synovial_PMN > 63.0', 'Positive_Histology == True and Synovial_PMN > 63.0', 'Positive_Histology == True and APTT > 24.0 and Synovial_PMN > 59.0', 'Positive_Histology == True', 'Synovial_WBC > 7003.0 and P_T <= 10.5 and Positive_Histology == True and Age > 29.0', 'Segment <= 73.5 and Positive_Histology == True and PLATELET <= 304.5 and Serum_ESR <= 71.0',
              'Serum_CRP <= 38.5 and Positive_Histology == True and Serum_CRP > 2.5', 'Positive_Histology == True and APTT <= 32.5 and Synovial_PMN <= 88.5', 'Synovial_WBC > 7003.0 and Positive_Histology == True and APTT <= 33.5', 'Synovial_PMN > 46.5 and Synovial_WBC > 2374.5 and Positive_Histology == True and Segment > 54.0', 'Serum_ESR > 28.5 and Positive_Histology == True and Serum_CRP > 3.0', 'Synovial_WBC > 7003.0 and Serum_CRP > 7.5 and APTT > 26.5', 'Synovial_PMN > 51.5 and Positive_Histology == True and Synovial_PMN > 63.0', 'Positive_Histology == True and Serum_ESR > 51.0', 'Serum_CRP > 13.5 and Synovial_WBC > 5763.0 and Synovial_WBC > 14610.0', 'Positive_Histology == True and Synovial_WBC > 14750.0 and Age <= 88.0', 'Positive_Histology == True and P_T > 5.0 and Synovial_WBC > 14569.0', 'Positive_Histology == True and Age <= 88.0 and Serum_ESR > 21.5 and Hb <= 13.5',
              'Synovial_PMN > 52.0 and Serum_WBC_ > 9.5 and Serum_ESR > 20.5 and Serum_CRP > 32.0 and Synovial_PMN > 66.5', 'Serum_CRP > 14.0 and Serum_WBC_ > 8.5 and PLATELET > 270.5', 'Serum_CRP > 16.5 and Synovial_PMN > 41.0 and Age <= 88.0 and Serum_WBC_ > 4.5 and Serum_ESR > 10.5 and PLATELET > 268.5 and Synovial_WBC > 2425.0', 'APTT > 28.5 and Serum_WBC_ > 9.5 and Serum_ESR > 31.0 and Age <= 77.5', 'Segment > 70.5 and Synovial_WBC > 3410.0 and Serum_ESR > 11.5', 'Serum_ESR > 30.5 and Serum_ESR > 55.5 and Synovial_WBC > 745.0', 'Serum_ESR > 41.0 and Synovial_PMN > 56.5 and Synovial_WBC > 1150.0 and APTT > 24.5 and Serum_CRP > 3.5', 'Synovial_PMN > 45.5 and Positive_Histology == True and Synovial_PMN > 63.0', 'Positive_Histology == True and Synovial_PMN > 64.0', 'Synovial_WBC > 7003.0 and APTT > 26.5 and Synovial_PMN > 38.5 and Serum_ESR > 9.5 and Age <= 89.0 and Hb > 7.5', 'Positive_Histology == True and Synovial_PMN > 63.0', 'Positive_Histology == True and Synovial_PMN > 63.0', 'Positive_Histology == True and APTT > 24.0 and Synovial_PMN > 59.0', 'Positive_Histology == True',
              'Serum_ESR > 28.5 and Positive_Histology == True and Synovial_PMN > 63.0', 'Synovial_PMN > 45.5 and Positive_Histology == True and Synovial_PMN > 63.0', 'Positive_Histology == True and Synovial_PMN > 64.0', 'Positive_Histology == True and Synovial_PMN > 63.0', 'Positive_Histology == True and Synovial_PMN > 63.0', 'Positive_Histology == True and APTT > 24.0 and Synovial_PMN > 59.0', 'Positive_Histology == True', 'Synovial_WBC > 7286.0 and APTT > 25.5 and Serum_CRP > 4.0 and Serum_CRP > 7.5', 'Synovial_WBC > 7286.0 and Hb > 5.0 and Synovial_WBC > 14610.0 and Serum_CRP > 3.5', 'Positive_Histology == True and Synovial_WBC > 14750.0 and Age <= 88.0', 'Serum_ESR > 40.5 and Hb > 4.5 and Synovial_WBC > 14569.0', 'Serum_ESR > 41.0 and Synovial_PMN > 56.5 and Synovial_WBC > 1150.0 and APTT > 24.5 and Serum_CRP > 3.5', 'Positive_Histology == True and Age <= 88.0 and Hb <= 13.5', 'Serum_ESR > 28.5 and Synovial_WBC > 2680.0 and Positive_Histology == True and Age <= 89.0',
              'Serum_CRP > 24.0 and P_T <= 10.5 and APTT > 26.5 and Serum_ESR > 48.5 and Serum_CRP > 28.5 and Synovial_WBC > 1985.0', 'Segment > 70.5 and Serum_CRP > 10.5 and PLATELET > 270.5', 'Age > 51.5 and Serum_ESR > 51.5 and Segment > 65.5', 'Serum_ESR > 41.0 and Synovial_WBC > 2990.0 and Serum_ESR > 73.5', 'Serum_ESR > 28.5 and Synovial_PMN > 52.0 and PLATELET > 196.0 and Synovial_WBC > 2443.5 and Age <= 89.0 and Segment > 65.5 and Hb <= 13.5', 'Serum_ESR > 41.0 and Synovial_WBC > 2640.0 and Hb <= 13.5', 'Serum_ESR > 47.5 and Synovial_WBC > 1205.0 and PLATELET > 52.0 and Synovial_WBC > 2268.5']

# In[26]: Call the transPOSForm func.

# POS_Form, final_singleton = personal_DecisionPath2.transPOSForm(rules_list)
# # print('POS_Form:')
# # print(POS_Form)
# print('final_singleton:')
# print(final_singleton)
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
internal_X = pd.read_csv('PJI_Dataset/internal_x_all.csv', encoding='utf-8')
internal_y = pd.read_csv('PJI_Dataset/internal_y_all.csv', encoding='utf-8')
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
    # PID_index = PID
    X_res_test = pd.read_csv(
        'PJI_Dataset/New_data_x_test.csv', encoding='utf-8')
    no_group = list(X_res_test['No.Group'])
    PID_index = no_group.index(PID)
    # X_train, y_train = internal_X.drop(
    #     index=PID_index), internal_y.drop(index=PID_index)
    # X_test, y_test = internal_X.iloc[PID_index:PID_index +
    #                                  1], internal_y.iloc[PID_index:PID_index+1]

    New_data_X = pd.read_csv('PJI_Dataset/New_data_x.csv', encoding='utf-8')
    New_data_y = pd.read_csv('PJI_Dataset/New_data_y.csv', encoding='utf-8')
    X_train, y_train = internal_X, internal_y

    X_test, y_test = New_data_X.iloc[PID_index:PID_index +
                                     1], New_data_y.iloc[PID_index:PID_index + 1]
    explainer = RandomForestClassifier(
        max_depth=Explainer_depth, n_estimators=100, random_state=PID)
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
    # PID_index = PID
    X_res_test = pd.read_csv(
        'PJI_Dataset/New_data_x_test.csv', encoding='utf-8')
    no_group = list(X_res_test['No.Group'])
    PID_index = no_group.index(PID)

    New_data_X = pd.read_csv('PJI_Dataset/New_data_x.csv', encoding='utf-8')
    New_data_y = pd.read_csv('PJI_Dataset/New_data_y.csv', encoding='utf-8')
    X_train, y_train = internal_X, internal_y

    X_test, y_test = New_data_X.iloc[PID_index:PID_index +
                                     1], New_data_y.iloc[PID_index:PID_index + 1]
    # X_train, y_train = internal_X_all.drop(
    #     index=PID_index), internal_y_all.drop(index=PID_index)
    # X_test, y_test = internal_X_all.iloc[PID_index:PID_index +
    #                                      1], internal_y_all.iloc[PID_index:PID_index + 1]
    explainer = RandomForestClassifier(
        max_depth=Explainer_depth, n_estimators=100, random_state=PID)
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
            # if decision_list[val][1] == '>=':
            #     rules_list_.append(
            #         str(decision_list[val][0])+' '+str(decision_list[val][1])+' '+str(decision_list[val][2]))
            #     rules_list_.append(
            #         str(decision_list[val][0])+' <= '+str(decision_list[val][3]))
            # elif decision_list[val][1] == '<=':
            #     rules_list_.append(
            #         str(decision_list[val][0])+' '+str(decision_list[val][1])+' '+str(decision_list[val][3]))
            #     rules_list_.append(
            #         str(decision_list[val][0])+' >= '+str(decision_list[val][2]))
            # else:
                # print('exception or not used:')
                # print(decision_list[val])

        except Exception as e:
            print("You got an Exception.", str(e))

    return rules_list_


# run_id = [62, 121, 151, 171, 231, 271, 331, 491, 531]
# run_id = [62]
# for i in run_id:
#     PID = i
#     no_group = list(X_res_test['No.Group'])
#     PID_index = 2 + no_group.index(PID)

#     X_train, y_train = internal_X.drop(
#         index=PID_index), internal_y.drop(index=PID_index)
#     X_test, y_test = internal_X.iloc[PID_index:PID_index +
#                                      1], internal_y.iloc[PID_index:PID_index + 1]
#     explainer = RandomForestClassifier(
#         max_depth=Explainer_depth, n_estimators=100, random_state=123)
#     explainer.fit(X_train.values, y_train.values)
#     singleton_opt(X_test, PID, final_singleton)
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


def find_delta(rules_list, PID):
    # no_group = list(X_res_test['No.Group'])
    # PID_index = 2 + no_group.index(PID)
    X_res_test = pd.read_csv(
        'PJI_Dataset/New_data_x_test.csv', encoding='utf-8')
    no_group = list(X_res_test['No.Group'])
    PID_index = no_group.index(PID)

    New_data_X = pd.read_csv('PJI_Dataset/New_data_x.csv', encoding='utf-8')
    New_data_y = pd.read_csv('PJI_Dataset/New_data_y.csv', encoding='utf-8')
    # # Stacking model
    stacking_model = joblib.load('Stacking_model/Stacking_model_new_data')
    # stacking_model = joblib.load('PJI_model/Stacking_model_'+str(PID))

    X_test, y_test = New_data_X.iloc[PID_index:PID_index +
                                     1], New_data_y.iloc[PID_index:PID_index + 1]
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
        # element = element.replace(') | (', ' and ')
        # element = element.replace('(', '')
        # element = element.replace(')', '')
        # element = element.replace('&', 'and')

        res_combined = res_combined + \
            [x.strip(' ') for x in element.split('and')]

        # rule_dict.update(dict.fromkeys([x.strip(' ')
        #                  for x in element.split('and')], i))
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
    #  In[22]: 列舉 all_singleton 的 features by List()
    # _singleton_list_f = list()
    # for i, (val) in enumerate(_singleton_list):

    #     if val.find("<=") > 0:
    #         _singleton_list_f.append(val[:val.find("<=")-1])
    #     elif val.find("<") > 0:
    #         _singleton_list_f.append(val[:val.find("<")-1])
    #     elif val.find("==") > 0:
    #         _singleton_list_f.append(val[:val.find("==")-1])
    #     elif val.find(">=") > 0:
    #         _singleton_list_f.append(val[:val.find(">=")-1])
    #     elif val.find(">") > 0:
    #         _singleton_list_f.append(val[:val.find(">")-1])

    AVG_FIDELITYS = []
    for i, condition in enumerate(_singleton_list):
        fidelity = []
        for val_df in VAL_DATASET:
            stack_pred = stacking_model.predict(val_df)
            val_df = val_df.rename(columns=d_path)
            merge_pred = np.where(
                val_df.eval(condition), rule_1, rule_2)
            fidelity.append(accuracy_score(stack_pred, merge_pred))
        AVG_FIDELITYS.append(str(condition)+' : ' +
                             str(round(np.mean(fidelity), 3)))
        AVG_FIDELITYS_ = round(np.mean(fidelity), 3)
        condition_List = condition.split(' ')
        name = condition_List[0]
        comp = condition_List[1]
        val = condition_List[2]
        if comp == '<=' or comp == '<':
            key = name
            if key not in singleton_dict:
                singleton_dict.setdefault(key, set()).add(str(condition)+' : ' +
                                                          str(round(np.mean(fidelity), 3)))
            else:
                singleton_dict[key].add(str(condition)+' : ' +
                                        str(round(np.mean(fidelity), 3)))
            if max_fid_dict_fid.get(key) <= AVG_FIDELITYS_:
                max_fid_dict[key] = i
                if max_fid_dict_val[key] == 0:
                    max_fid_dict_val[key] = condition
                else:
                    max_fid_List = max_fid_dict_val[key].split(' ')
                    max_fid_val = max_fid_List[2]

                    if float(val) <= float(max_fid_val):
                        max_fid_dict_val[key] = condition
                        max_fid_dict[key] = i
                    else:
                        max_fid_dict_val[key] = condition
                max_fid_dict_fid[key] = AVG_FIDELITYS_
        # elif condition.find("<") > 0:
        #     key = condition[:condition.find("<")-1]
        #     if key not in singleton_dict:
        #         singleton_dict.setdefault(key, set()).add(str(condition)+' : ' +
        #                                                   str(round(np.mean(fidelity), 3)))
        #     else:
        #         singleton_dict[key].add(str(condition)+' : ' +
        #                                 str(round(np.mean(fidelity), 3)))
        #     if max_fid_dict_fid.get(key) < AVG_FIDELITYS_:
        #         max_fid_dict[key] = i
        #         max_fid_dict_val[key] = condition
        #         max_fid_dict_fid[key] = AVG_FIDELITYS_
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
        elif comp == '>=' or comp == '>':
            key = name
            if key not in singleton_dict:
                singleton_dict.setdefault(key, set()).add(str(condition)+' : ' +
                                                          str(round(np.mean(fidelity), 3)))
            else:
                singleton_dict[key].add(str(condition)+' : ' +
                                        str(round(np.mean(fidelity), 3)))
            if max_fid_dict_fid.get(key) <= AVG_FIDELITYS_:
                max_fid_dict[key] = i
                if max_fid_dict_val[key] == 0:
                    max_fid_dict_val[key] = condition
                else:
                    max_fid_List = max_fid_dict_val[key].split(' ')
                    max_fid_val = max_fid_List[2]
                    if float(val) >= float(max_fid_val):
                        max_fid_dict_val[key] = condition
                        max_fid_dict[key] = i
                    else:
                        max_fid_dict_val[key] = condition
                max_fid_dict_fid[key] = AVG_FIDELITYS_
        # elif condition.find(">") > 0:
        #     key = condition[:condition.find(">")-1]
        #     if key not in singleton_dict:
        #         singleton_dict.setdefault(key, set()).add(str(condition)+' : ' +
        #                                                   str(round(np.mean(fidelity), 3)))
        #     else:
        #         singleton_dict[key].add(str(condition)+' : ' +
        #                                 str(round(np.mean(fidelity), 3)))
        #     if max_fid_dict_fid.get(key) < AVG_FIDELITYS_:
        #         max_fid_dict[key] = i
        #         max_fid_dict_val[key] = condition
        #         max_fid_dict_fid[key] = AVG_FIDELITYS_
    print('All singletons fidelity:')
    print('')
    print(AVG_FIDELITYS)
    print('')
    print('max_fid_dict:')
    print(max_fid_dict_val)
    print('')
    print(max_fid_dict_fid)
    print('')
    # print(singleton_dict)
    # print('')

    for key, val in max_fid_dict.items():
        to_replace = _singleton_list[val].split(' ')[0]
        for i1, rules in enumerate(rules_list):
            each_rule_list = rules.split(' and ')
            for i2, each_rule in enumerate(each_rule_list):
                if each_rule.startswith(to_replace):
                    each_rule_list[i2] = _singleton_list[val].strip()
            rules_list[i1] = ' and '.join(each_rule_list)
    print('rule_top_fidelity:')
    print(rules_list)
    print('')
    # 不考慮之singleton，在rule中刪除
    for i1, rules in enumerate(rules_list):
        each_rule_list = rules.split(' and ')
        each_rule_list_temp = rules.split(' and ')
        for i2, each_rule in enumerate(each_rule_list_temp):
            flag = 0
            for val in _singleton_list:
                if str(each_rule) == str(val):
                    flag = 1
                # print('flag_1:')
                # print(flag)
            if flag == 0:
                each_rule_list.remove(each_rule)
                # try:
                #     each_rule_list.remove(each_rule)
                # except:
                #     continue
            #
        rules_list[i1] = ' and '.join(each_rule_list)
    print('rule_deleted:', rules_list)
    print('')
    rules_list = [x for x in rules_list if x != '']
    # res_combined = []
    # _singleton_list = list()
    # for i, (element) in enumerate(rules_list):
    #     res_combined = res_combined + \
    #         [x.strip(' ') for x in element.split('and')]

    # _singleton_set = set()
    # for atom in res_combined:
    #     _singleton_set.add(atom)

    # for atom in _singleton_set:
    #     _singleton_list.append(atom)
    # eleMap = dict()

    # for ele in _singleton_list:
    #     eleList = ele.split(' ')
    #     name = eleList[0]
    #     comp = eleList[1]
    #     if comp == '==':
    #         val = eleList[2]
    #     else:
    #         val = float(eleList[2])
    #     if name not in eleMap:
    #         eleMap[name] = dict()
    #     if comp not in eleMap[name]:
    #         eleMap[name][comp] = val
    #     else:
    #         if comp == '<=' or comp == '<':
    #             eleMap[name][comp] = min(eleMap[name][comp], val)
    #         elif comp == '>=' or comp == '>':
    #             eleMap[name][comp] = max(eleMap[name][comp], val)
    #         else:
    #             eleMap[name][comp] = val

    # print('singleton_dict_all_bound:')
    # print(eleMap)

    # for i1, rules in enumerate(rules_list):
    #     each_rule_list = rules.split(' and ')
    #     for i2, each_rule in enumerate(each_rule_list):
    #         each_rule_item = each_rule.split(' ')
    #         name = each_rule_item[0]
    #         comp = each_rule_item[1]
    #         if comp == '==':
    #             val = each_rule_item[2]
    #         else:
    #             val = eleMap[name][comp]
    #         each_rule_list[i2] = name + ' ' + comp + \
    #             ' ' + str(val)
    #     rules_list[i1] = ' and '.join(each_rule_list)

    # print('rule_wide_range:')
    # print(rules_list)
    # print('')

    return rules_list
