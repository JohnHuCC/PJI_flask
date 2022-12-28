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

# [Singleton]
# APTT >= [28, 31]
# P_T >= [9.8, 11]
# Positive_Histology == True
# Segment >= [62, 68]
# Serum_CRP >= [10, 50]
# Serum_ESR >= [30, 81]
# Synovial_WBC >= [3000, 7966]

today=datetime.date.today().strftime('%Y%m%d')
output_path  = './output/' + str(today) + '_RS2_Report_Case1.csv'
# write CSV File ###
output_file = open(output_path, 'w',newline='')
csvHeader  = ['POS_Form',
              
              'APTT >= ',
              'P_T >= ',
              'Segment >= ',
              'Serum_CRP >= ',
              'Serum_ESR >= ',
              'Synovial_WBC >= ',
              
              'Fidelity',
              'merge_mean_acc', 'stack_mean_acc',
              'merge_mean_auc', 'stack_mean_auc', 
              'merge_mean_mcc', 'stack_mean_mcc',
              'merge_mean_f1', 'stack_mean_f1',
              'merge_mean_recall', 'stack_mean_recall', 
              'merge_mean_precision', 'stack_mean_precision']

csvCursor  = csv.DictWriter(output_file, fieldnames=csvHeader)
csvCursor  = csv.writer(output_file)
csvCursor.writerow(csvHeader)

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

CONDITIOS2 = [
# case1
'(Positive_Histology == True & Serum_CRP > 4.0) | (APTT > 26.5 & Serum_ESR > 39.5 & Synovial_WBC > 2483.5) | (P_T > 5.0 & Positive_Histology == True & Segment > 65.5)'

]

bounds = [
    [28, 31], # APTT
    [9.8, 11], #PT
    [62, 68], #Segment
    [10, 50], #SerumCRP
    [30, 81], #SerumESR
    [3000, 7966], #SynovialWBC    
]



num = 11
l = list(permutations(range(1, num+1), 6)) 
random.shuffle(l)

candidate_list = []
string_Origin = CONDITIOS2[0]
for i in l:
    
    params = get_random_point(bounds)
    
    aptt = params[0]
    pt = params[1]
    segment = params[2]
    serumCRP = params[3]
    serumESR = params[4]
    synovialWBC = params[5]
    
    condition = string_Origin
    condition = condition.replace('APTT > 26.5', 'APTT >= ' + str(aptt))
    condition = condition.replace('P_T > 5.0', 'P_T >= ' + str(pt))
    condition = condition.replace('Segment > 65.5', 'Segment >= ' + str(segment))
    condition = condition.replace('Serum_CRP > 4.0', 'Serum_CRP >= ' + str(serumCRP))
    condition = condition.replace('Serum_ESR > 39.5', 'Serum_ESR >= ' + str(serumESR))
    condition = condition.replace('Synovial_WBC > 2483.5', 'Synovial_WBC >= ' + str(synovialWBC))
                            
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
                            
                            aptt,
                            pt,
                            segment,
                            serumCRP,
                            serumESR,
                            synovialWBC,
                            
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
