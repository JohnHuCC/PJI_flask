import os
import ast
# from pyeda.boolalg.expr import exprvar
# from pyeda.boolalg import boolfunc
# from pyeda.boolalg.minimization import espresso_exprs
from sympy.logic.boolalg import to_dnf
# from rule_no_tune import transPOSForm
import re
import pandas as pd
from sympy import symbols
from sympy import Symbol
from pyeda.boolalg.expr import OrOp, AndOp, exprvar
import rule_parser
from sklearn.model_selection import StratifiedKFold
from rule_parser import *


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
            for i in index:

                if (_singleton_list[i].find('<=') > 0 or _singleton_list[i].find('<') > 0):
                    for val2 in index:  # index):
                        singleton_temp = _singleton_list[val2].split(' ')
                        if (singleton_temp[1] == '<=' or singleton_temp[1] == '<'):
                            val_list.append(float(singleton_temp[2]))
                        else:
                            val_list.append(float('-inf'))
                        # val_list.append(
                        #     float(_singleton_list[val2][_singleton_list[val2].find("<=")+3:]))
                    # min_value = min(val_list)
                    # min_index = val_list.index(min_value)
                    max_value = max(val_list)
                    max_index = val_list.index(max_value)
                    val_list = []
                    # final_singleton.add(_singleton_list[index[min_index]])
                    final_singleton.add(_singleton_list[index[max_index]])
                    for i in index:
                        _singleton_list_[_singleton_list[i]
                                         ] = _singleton_list[index[max_index]]

                elif (_singleton_list[i].find('>=') > 0 or _singleton_list[i].find('>') > 0):
                    for val2 in index:
                        singleton_temp = _singleton_list[val2].split(' ')
                        if (singleton_temp[1] == '>=' or singleton_temp[1] == '>'):
                            val_list.append(float(singleton_temp[2]))
                        else:
                            val_list.append(float('inf'))
                        # val_list.append(
                        #     float(_singleton_list[val2][_singleton_list[val2].find(">=")+3:]))
                    # max_value = max(val_list)
                    # max_index = val_list.index(max_value)
                    min_value = min(val_list)
                    min_index = val_list.index(min_value)
                    val_list = []
                    final_singleton.add(_singleton_list[index[min_index]])
                    for i in index:
                        _singleton_list_[_singleton_list[i]
                                         ] = _singleton_list[index[min_index]]

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
    transPOSForm_ini = []
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
        # transPOSForm_ini.append(eval(string))
    # print('transPOSForm_ini:', transPOSForm_ini)
    # return list(set(transPOSForm)), list(final_singleton)
    return list(set(transPOSForm)), list(set(final_singleton))


singleton_map = dict()


def map_to_var(final_singleton, rule_str):
    # j = 65
    for i in range(len(final_singleton)):
        if i < 26:
            j = 65
            a = j + i
            singleton_map[str(chr(a))] = final_singleton[i]
            rule_str = rule_str.replace(final_singleton[i], str(chr(a)))
        else:
            j = 90
            a = j + i
            singleton_map[str(chr(a))] = final_singleton[i]
            rule_str = rule_str.replace(final_singleton[i], str(chr(a)))
    return (rule_str)


def decode_expression(expr):
    if isinstance(expr, expr.Or):
        decoded_terms = [decode_expression(term) for term in expr.args]
        return ' | '.join(decoded_terms)
    elif isinstance(expr, expr.And):
        decoded_literals = []
        for literal in expr.args:
            if isinstance(literal, exprvar):
                decoded_literals.append(str(literal))
            else:
                decoded_literals.append(decode_expression(literal))
        return ' & '.join(decoded_literals)
    else:
        return str(expr)


A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z = map(
    exprvar, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = map(
    exprvar, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
# test_expr = (a & b) | (b & c)
rules_list = ['Serum_CRP > 13.5 and Positive_Histology == True', 'Synovial_WBC <= 13850.0 and Positive_Histology == True and Serum_WBC_ <= 7.5',
              'Serum_CRP > 14.0 and Synovial_WBC <= 25532.0 and two_positive_culture == False and Synovial_PMN > 78.5 and Total_Elixhauser_Groups_per_record <= 1.5 and Segment > 64.5']
POS_Form_fix, final_singleton_fix = transPOSForm(rules_list)
# print('POS_Form_fix:', POS_Form_fix)
# _POS_Form_fix = POS_Form_fix.copy()
# for i, rules in enumerate(POS_Form_fix):
#     # _POS_Form_fix[i] = '(' + str(rules) + ')'
#     _POS_Form_fix[i] = str(rules)
# _POS_Form_fix = ' | '.join(_POS_Form_fix)


def pp(e):
    if isinstance(e, OrOp):
        return "{} | {}".format(pp(e.xs[0]), pp(e.xs[1]))
    elif isinstance(e, AndOp):
        return "{} & {}".format(pp(e.xs[0]), pp(e.xs[1]))
    else:
        return "{}".format(e)


# f1 = C & D | C & I & G | C & H & I | C & H & B & A | C & E & F & J | C & H & G & E
# f2 = A & B & C | A & B | A & C & D | A & E | B & C & F & G | A & E & H | C & E & G
# f3 = a & b | a & c
# f1m, f2m, f3m = espresso_exprs(f1.to_dnf(), f2.to_dnf(), f3.to_dnf())
# # print(f1m)
# print(f2m)
# # f2m = f2.to_dnf()
# # # print('f1m:', f1m)
# # print('f2m:', f2m)
# # print('f2m expresso:', espresso_exprs(f2m))
# # # print('f3m:', f3m)
# parsed_string = parse_ast(f2m)
# print(parsed_string)

df = pd.read_csv('PJI_Dataset/PJI_all.csv')

all_ids = df['No.Group'].values.tolist()


# 資料夾路徑，其中應該包含所有的 JSON 檔
folder_path = 'Decision_rule/'

# 存儲缺失的 ID
missing_ids = []

# 檢查每個 ID 以查看是否有一個對應的 JSON 檔
for id in all_ids:
    expected_filename = f"decision_rule_map_{id}.json"
    if expected_filename not in os.listdir(folder_path):
        missing_ids.append(id)

print("缺失的 ID:", missing_ids)
