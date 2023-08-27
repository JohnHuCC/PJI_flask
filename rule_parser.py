from pyeda.boolalg.expr import expr, OrOp, AndOp, exprvar
from pyparsing import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import json
import warnings
warnings.filterwarnings("ignore")


def Non_ICM_1(x):
    return {
        # p6~12
        'mu(I)_delta_mu(N)': ['<=', 'delta', 'mu(N)'],  # 情境八
        'mu(I)_mu(N)_delta': ['NotUsed'],  # 情境七
        'delta_mu(N)_mu(I)': ['NotUsed'],  # 情境四 情境五 情境六
        'delta_mu(I)_mu(N)': ['<=', 'delta', 'mu(N)'],  # 情境九
        'mu(N)_delta_mu(I)': ['>=', 'mu(N)', 'delta'],  # 情境一 情境二 情境三
        'mu(N)_mu(I)_delta': ['>=', 'mu(N)', 'delta'],  # 情境十 情境十一 情境十二 情境十三

    }.get(x, 'Error')


def Non_ICM_0(x):
    return {
        # p13～21
        'delta_mu(N)_mu(I)': ['<=', 'delta', 'mu(I)'],  # 情境九 情境十二 情境十三 情境十四
        'delta_mu(I)_mu(N)': ['NotUsed'],  # 情境十五
        'mu(N)_mu(I)_delta': ['NotUsed'],  # 情境一 情境二 情境六 情境七 情境十一
        'mu(N)_delta_mu(I)': ['<=', 'delta', 'mu(I)'],  # 情境三 情境四 情境五 情境八 情境十
        'mu(I)_delta_mu(N)': ['>=', 'mu(I)', 'delta'],  # 情境十七
        'mu(I)_mu(N)_delta': ['>=', 'mu(I)', 'delta'],  # 情境十六
    }.get(x, 'Error')


def ICM_1(x):
    return {
        # p1~3
        'ICM_Q_delta': ['>=', 'ICM', 'delta'],  # 情境一
        'ICM_delta_Q': ['>=', 'ICM', 'delta'],  # 情境二
        'Q_ICM_delta': ['>=', 'ICM', 'delta'],  # 情境三
        'Q_delta_ICM': ['>=', 'Q', 'delta'],  # 情境四
        'delta_ICM_Q': ['NotUsed'],  # 情境五
        'delta_Q_ICM': ['NotUsed'],  # 情境六
        # 情境七
    }.get(x, 'Error')


def ICM_0(x):
    return {
        # p4~5
        'delta_Q_ICM': ['<=', 'delta', 'ICM'],  # 情境一
        'delta_ICM_Q': ['<=', 'delta', 'ICM'],  # 情境二
        'ICM_delta_Q': ['<=', 'delta', 'Q'],  # 情境三
        'ICM_Q_delta': ['NotUsed'],  # 情境四
        'Q_ICM_delta': ['NotUsed'],  # 情境五
        'Q_delta_ICM': ['<=', 'delta', 'ICM'],  # 情境六

    }.get(x, 'Error')


def parse_ast(e):
    if isinstance(e, OrOp):
        return " | ".join([parse_ast(i) for i in e.xs])
    elif isinstance(e, AndOp):
        return " & ".join([parse_ast(i) for i in e.xs])
    else:
        return "{}".format(e)


def parse_string(input_expr):
    # 定義符號
    # LPAR, RPAR = map(Suppress, '()')
    AND, OR = map(CaselessLiteral, ['And', 'Or'])
    LPAR, RPAR = map(CaselessLiteral, ['(', ')'])
    var = Word(alphas, exact=1)
    expression = Forward()

    # 定義語法規則
    Search_term = LPAR + ZeroOrMore(var + Optional(',')) + RPAR
    var_term = var + Optional(',')
    And_operand = AND + Search_term + Optional(',')
    Or_operand = OR + LPAR + ZeroOrMore(And_operand) + RPAR  \
        | OR + LPAR + var_term + var_term + ZeroOrMore(And_operand) + RPAR \
        | OR + LPAR + var_term + ZeroOrMore(And_operand) + RPAR \
        | OR + LPAR + var_term + var_term + var_term + ZeroOrMore(And_operand) + RPAR \
        | And_operand
    expression << LPAR + Or_operand + \
        Optional(',') + RPAR

    # 解析表達式
    # input_expr = "(Or(And(J, O, P), And(B, J), And(D, J), And(A, B, F, H, K, O), And(E, J, K), And(G, H)),)"
    # input_expr = "(Or(And(J, O, P), Or(B, Or(Z, X, And(Y, y))), Or(A, B, F, H, K, O), And(Or(a,b), H)),)"
    # '(Or(E, P, And(A, G), And(A, B), And(A, J, M, O)),)'
    parsed_string = expression.parse_string(input_expr)
    return parsed_string


def gen_readable_expr(src_expr):
    new_expr = ""
    cur_operator = ''
    # if src_expr[0] != '(': operator_stack = []
    operator_stack = ['']

    for i, literal in enumerate(src_expr):

        if literal == ',':
            continue
        elif literal == '(':
            new_expr += '('
        elif literal == ')':

            if len(operator_stack) != 0:
                operator = operator_stack.pop()

                if operator != "":
                    new_expr = new_expr[:-1]
                    new_expr += ')'
                    new_expr += operator_stack[-1]
        else:
            if literal == 'Or':
                operator_stack.append('|')
            elif literal == 'And':
                operator_stack.append('&')
            else:
                cur_operator = literal
                new_expr += (cur_operator+operator_stack[-1])

        # print(f"step {i:3d} \'{literal}\':\t", new_expr, operator_stack)

    return new_expr[1:]


# output: (J&O&P) | ()...
# "H & I & M | M & L | N & M & K | M & F & L & O | G & M & D & L & O & C | M & L & K | O & J | E & L & K & P & J & C | A & M | M & B & P"
singleton_map = dict()


def map_to_var(final_singleton, rule_str, singleton_map):
    # j = 65
    final_singleton = list(set(final_singleton))
    print('final_singleton:', final_singleton)
    for i in range(len(final_singleton)):
        if i < 26:
            j = 65
            a = j + i
            singleton_map[str(chr(a))] = final_singleton[i]
            rule_str = rule_str.replace(final_singleton[i], str(chr(a)))
        else:
            j = 71
            a = j + i
            singleton_map[str(chr(a))] = final_singleton[i]
            rule_str = rule_str.replace(final_singleton[i], str(chr(a)))
    return rule_str, singleton_map


def alphabet_to_singleton(parsed_string, singleton_map):
    _parsed_string = list(parsed_string)
    for i, item in enumerate(parsed_string):
        if item in singleton_map.keys():
            _parsed_string[i] = singleton_map[item]
    parsed_string = ''.join(_parsed_string)
    _parsed_string = list()
    _parsed_string.append(parsed_string)
    return _parsed_string


def spilt_rule(Candidate_DecisionPath):
    rule_splited = []
    for element in Candidate_DecisionPath:
        element = element.replace(')|(', ' | ')
        element = element.replace('(', '')
        element = element.replace(')', '')
        element = element.replace('&', ' and ')
        element = element.replace('|', ' | ')
        rule_splited = rule_splited + \
            [x.strip(' ') for x in element.split(' | ')]
    return rule_splited


def split_singleton(rule_splited):
    rule_dict = dict()
    for i, ele in enumerate(rule_splited):
        rule_dict[i] = ele.split(' and ')
    return rule_dict


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


Explainer_depth = 7


def explainer_pred(X_test):
    internal_X = pd.read_csv(
        'PJI_Dataset/internal_x_for_new_data.csv', encoding='utf-8')
    internal_y = pd.read_csv(
        'PJI_Dataset/internal_y_for_new_data.csv', encoding='utf-8')

    explainer = RandomForestClassifier(
        max_depth=Explainer_depth, n_estimators=100, random_state=123)
    explainer.fit(internal_X.values, internal_y.values)
    result = explainer.predict(X_test)
    return result


def check_singleton(PID, singleton_list):
    # X_res_test = pd.read_csv(
    #     'PJI_Dataset/New_data_x_test.csv', encoding='utf-8')
    # no_group = list(X_res_test['No.Group'])

    test_data_for_index = pd.read_csv(
        'PJI_Dataset/internal_x_test.csv', encoding='utf-8')
    no_group = list(test_data_for_index['No.Group'])
    PID_index = no_group.index(PID)

    # New_data_X = pd.read_csv('PJI_Dataset/New_data_x.csv', encoding='utf-8')
    # New_data_y = pd.read_csv('PJI_Dataset/New_data_y.csv', encoding='utf-8')
    X_test = test_data_for_index.iloc[PID_index:PID_index + 1, 1:]
    X_test_ = X_test_rename(X_test)
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

    explainer_result = explainer_pred(X_test)
    NotUsed_flag = 0
    delete_sign = False
    for ele in singleton_list:
        eleList = ele.split(' ')
        variable = eleList[0]
        operator = eleList[1]
        Q = eleList[2]
        if operator == "==":
            delete_sign = False
            continue
        else:
            if variable in list(non2018ICM_['variable']):  # non2018ICM List
                index = list(non2018ICM_['variable']).index(variable)
                mu_N = non2018ICM_['mu(N)'][index]
                mu_I = non2018ICM_['mu(I)'][index]
                delta = float(X_test_[variable])

                nonICM = {"mu(N)": mu_N, "mu(I)": mu_I, "delta": delta}
                # nonICM_Sorting = (sorted(nonICM.items(), key=lambda x:x[1]))
                nonICM_Sorting = sorted(nonICM, key=nonICM.get)
                concate_nonICM_Sorting = '_'.join(nonICM_Sorting)

                if explainer_result == 0:   # Meta(I) = 0
                    result = Non_ICM_0(concate_nonICM_Sorting)
                    if len(result) == 3:
                        operator, _lower_bound, _upper_bound = result
                        lower_bound, upper_bound = nonICM[_lower_bound], nonICM[_upper_bound]

                    else:  # 不考慮
                        NotUsed_flag = 1

                elif explainer_result == 1:  # Meta(I) = 1
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

                if explainer_result == 0:   # Meta(I) = 0
                    result = ICM_0(concate_ICM_Sorting)
                    if len(result) == 3:
                        operator, _lower_bound, _upper_bound = result
                        lower_bound, upper_bound = ICM[_lower_bound], ICM[_upper_bound]

                    else:  # 不考慮
                        NotUsed_flag = 1

                elif explainer_result == 1:   # Meta(I) = 1
                    result = ICM_1(concate_ICM_Sorting)
                    if len(result) == 3:
                        operator, _lower_bound, _upper_bound = result
                        lower_bound, upper_bound = ICM[_lower_bound], ICM[_upper_bound]

                    else:  # 不考慮
                        NotUsed_flag = 1

        if NotUsed_flag == 1:
            delete_sign = True

    return delete_sign


def filt_rule(rule_dict, PID):
    _rule_dict = rule_dict.copy()
    for key in rule_dict:
        delete_sign = check_singleton(PID, rule_dict[key])
        if delete_sign == True:
            _rule_dict.pop(key)
    return _rule_dict


def concat_rule(rule_dict):
    rule_str = list()
    for i, rule_item in rule_dict.items():
        rule_item = ' and '.join(rule_item)
        rule_item = '('+rule_item+')'
        rule_str.append(rule_item)
    rule_str = ' | '.join(rule_str)
    return rule_str


def concat_rule_tolist(rule_dict):
    rule_str = list()
    for i, rule_item in rule_dict.items():
        rule_item = ' and '.join(rule_item)
        rule_item = '('+rule_item+')'
        rule_str.append(rule_item)
    return rule_str
