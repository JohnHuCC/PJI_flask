# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:52:28 2020

@author: Indi
"""
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, matthews_corrcoef, auc
import csv
import datetime
from itertools import permutations 
import random


def get_SingletonList(X):
    # In[18] 拆解所有的 decision path 元素
    ## input: all_path
    ## output: all_singleton
    data = X.split(" | ")
    res_combined = []
    
    for element in data:
        element = element.replace(') | (', ' and ')
        element = element.replace('(', '')
        element = element.replace(')', '')
        element = element.replace('&', 'and')
        
        res_combined = res_combined + [x.strip(' ') for x in element.split('and')]
        
    _singleton_set = set()
    for atom in res_combined:
        _singleton_set.add(atom)
        
    _singleton_list = list()
    for atom in _singleton_set:
        _singleton_list.append(atom)       
        
    singleton_f = set()
    for i, (val) in enumerate(_singleton_list):
        
        if val.find("<=") > 0:
            singleton_f.add(val[:val.find("<=")-1])
        
        elif val.find("<") > 0:
            singleton_f.add(val[:val.find("<")-1])
            
        # 不處理 " == True" or " == False"     
        # elif val.find("==") > 0:
            #singleton_f.add(val[:val.find("==")-1])
        
        elif val.find(">=") > 0:    
            singleton_f.add(val[:val.find(">=")-1])
    
        elif val.find(">") > 0:
            singleton_f.add(val[:val.find(">")-1])    

    return _singleton_list, sorted(list(singleton_f))

def get_random_point(bounds):
    r = random.random()
    rps = []
    for i in range(len(bounds)):
        bound = bounds[i]
        if type(bound[0]) == str:
            str_bounds = np.linspace(0, 1, len(bound))[1:]
            for i in range(str_bounds.shape[0]):
                if r < str_bounds[i]:
                    rp = bound[i]
                    break
        else:
            rp = bound[0] + int(r * (bound[1] - bound[0]))
        rps.append(rp)
    return rps


# 不回傳 " == True" or " == False"        
SingletonList, Singletons = get_SingletonList(Simplify_DecisionRules)

SingletonList_ = {}
for key in SingletonList:
    if key in opt_decision_list:
        SingletonList_[key] = opt_decision_list[key]
        
    
today=datetime.date.today().strftime('%Y%m%d')
output_path  = './output/' + str(today) + '_RS2_Report_PID' + str(PID) + '.csv'
# write CSV File ###
output_file = open(output_path, 'w',newline='')

csvHeader  = ['POS_Form',
              'Fidelity',
              'merge_mean_acc', 'stack_mean_acc',
              'merge_mean_auc', 'stack_mean_auc', 
              'merge_mean_mcc', 'stack_mean_mcc',
              'merge_mean_f1', 'stack_mean_f1',
              'merge_mean_recall', 'stack_mean_recall', 
              'merge_mean_precision', 'stack_mean_precision']

i = 1
for element in (SingletonList_):
    elem = SingletonList_[element][0] + " " + SingletonList_[element][1]
    csvHeader.insert(i, elem)
    i = i + 1
 

csvCursor  = csv.DictWriter(output_file, fieldnames=csvHeader)
csvCursor  = csv.writer(output_file)
csvCursor.writerow(csvHeader)

bounds_ = []
i = 0
for element in (SingletonList_):
    elem = [SingletonList_[element][2], SingletonList_[element][3]]
    bounds_.insert(i, elem)
    i = i + 1

# 預估計畫跑算法次數
# 從計算複雜度回推需要跑幾次，再取適當的num, 
num = 11
l = list(permutations(range(1, num+1), len(bounds_))) 
random.shuffle(l)

candidate_list = []
# string_Origin = Simplify_DecisionRules
for i in l:
    
    params = get_random_point(bounds_)
       
    # # 按照 SingletonList_.keys() 取得排序
   
    condition = Simplify_DecisionRules
    singleton_ = {}
    k = 0 
    for e1, e2 in zip(SingletonList_, params):
        # print (e1,',', SingletonList_[e1][0], SingletonList_[e1][1])
        str_ = SingletonList_[e1][0] + ' ' + SingletonList_[e1][1]
        condition = condition.replace(e1, str_ + str(e2))
        singleton_[k] = e2
        k += 1
    
   
    print(condition)
    
    fidelity = []
    acc_1 = []
    auc_1 = []
    f1_1 = []
    recall_1 = []
    precision_1 = []
    mcc_1 = []                
    acc_2 = []
    auc_2 = []
    f1_2 = []
    recall_2 = []
    precision_2 = []
    mcc_2 = []
    
    for val_df, y_val_df in zip(VAL_DATASET, Y_VAL_DATASET):
        stack_pred = stacking_model.predict(val_df)
        merge_pred = np.where(val_df.rename(columns=d_path).eval(condition), rule_1, rule_2)
        fidelity.append(accuracy_score(stack_pred, merge_pred))
        acc_1.append(accuracy_score(y_val_df, merge_pred))
        fpr, tpr, thresholds = metrics.roc_curve(y_val_df, merge_pred, pos_label=1)
        auc_1.append(auc(fpr, tpr))
        f1_1.append(f1_score(y_val_df, merge_pred, average='weighted'))
        recall_1.append(recall_score(y_val_df, merge_pred, average='weighted'))
        precision_1.append(precision_score(y_val_df, merge_pred, average='weighted'))
        mcc_1.append(matthews_corrcoef(y_val_df.values, merge_pred))
        
        acc_2.append(accuracy_score(y_val_df, stack_pred))
        fpr, tpr, thresholds = metrics.roc_curve(y_val_df, stack_pred, pos_label=1)
        auc_2.append(metrics.auc(fpr, tpr))
        f1_2.append(f1_score(y_val_df, stack_pred, average='weighted'))
        recall_2.append(recall_score(y_val_df, stack_pred, average='weighted'))
        precision_2.append(precision_score(y_val_df, stack_pred, average='weighted'))
        mcc_2.append(matthews_corrcoef(y_val_df.values, stack_pred))
    
    candidate_list.sort(reverse=True)
    mean_fidelity = np.array(fidelity).mean()
    
    if len(candidate_list) == 0:
        candidate_list.append(mean_fidelity)
        
    if (mean_fidelity >= np.median(candidate_list)):
        candidate_list.append(mean_fidelity)
        
        mean_acc_1 = np.array(acc_1).mean()
        mean_auc_1 = np.array(auc_1).mean()
        mean_mcc_1 = np.array(mcc_1).mean()
        mean_f1_1  = np.array(f1_1).mean()
        mean_recall_1 = np.array(recall_1).mean()
        mean_precision_1 = np.array(precision_1).mean() 
        
        mean_acc_2 = np.array(acc_2).mean()
        mean_auc_2 = np.array(auc_2).mean()
        mean_mcc_2 = np.array(mcc_2).mean()
        mean_f1_2  = np.array(f1_2).mean()
        mean_recall_2 = np.array(recall_2).mean()
        mean_precision_2 = np.array(precision_2).mean() 
        
        
        
        
        
        csvCursor.writerow((condition,
                            
                           
                            singleton_[0], # Synovial_WBC
                            singleton_[1], # Age
                            singleton_[2], # Serum_CRP
                            singleton_[3], # Serum_ESR
                            singleton_[4], # APTT
                            singleton_[5], # Segment
                            singleton_[6], # Serum_WBC_
                            
                            mean_fidelity, 
                            mean_acc_1, 
                            mean_acc_2, 
                            mean_auc_1, 
                            mean_auc_2,
                            mean_mcc_1,
                            mean_mcc_2,
                            mean_f1_1, 
                            mean_f1_2,
                            mean_recall_1, 
                            mean_recall_2,
                            mean_precision_1, 
                            mean_precision_2,
                            
                            ))

output_file.close() 
