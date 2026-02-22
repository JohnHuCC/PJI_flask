# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:44:30 2021
本檔案內容已納入 POS_form.py
@author: Indi
"""

def Non_ICM_1(x):
    return {
        'mu(I)_delta_mu(N)': ['<=', 'delta', 'mu(N)'],
        'mu(I)_mu(N)_delta': ['NotUsed'],
        'delta_mu(N)_mu(I)': ['NotUsed'],
        'delta_mu(I)_mu(N)': ['<=', 'delta', 'mu(N)'],
        'mu(N)_delta_mu(I)': ['>=', 'mu(N)', 'delta'],
        'mu(N)_mu(I)_delta': ['>=', 'mu(N)', 'delta'],
        
    }.get(x,'Error')

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
        
        'ICM_Q_delta':['>=', 'ICM', 'delta'],
        'ICM_delta_Q':['>=', 'ICM', 'delta'],
        'Q_ICM_delta':['>=', 'ICM', 'delta'],
        'Q_delta_ICM':['>=', 'Q', 'delta'],
        'delta_ICM_Q':['NotUsed'],
        'delta_Q_ICM':['NotUsed'],
            
    }.get(x, 'Error')

    
def ICM_0(x):
    return {
        'delta_Q_ICM':['<=', 'delta', 'ICM'],
        'delta_ICM_Q':['<=', 'delta', 'ICM'],
        'ICM_delta_Q':['<=', 'delta', 'Q'],
        'ICM_Q_delta':['NotUsed'],
        'Q_ICM_delta':['NotUsed'],
        'Q_delta_ICM':['<=', 'delta', 'ICM'],
        
    }.get(x, 'Error')



non2018ICM = pd.read_excel("C:/Users/Indi_/Desktop/Non2018ICM.xlsx")
_2018ICM = pd.read_excel("C:/Users/Indi_/Desktop/2018ICM.xlsx")

_2018ICM_ = _2018ICM[['variable','threshold']]
non2018ICM_ = non2018ICM[['variable','mu(N)', 'mu(I)']]

non2018ICM_.iloc[0, 1]

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

 
# for val in list(_singleton_list_.keys()):
     
#     variable = val[:val.find("<=")-1]
#     operator = val[val.find("<="):val.find("<=")+2]
#     Q = val[val.find("<=")+3:]
#     print (variable, operator, Q)



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
    
    print (variable, operator, Q)
    
    
    
    
    
    try: 
        if variable in list(non2018ICM_['variable']): # non2018ICM List
            index = list(non2018ICM_['variable']).index(variable)
            mu_N = non2018ICM_['mu(N)'][index]
            mu_I = non2018ICM_['mu(I)'][index]
            delta = float(X_test_[variable])
            
            nonICM = {"mu(N)":mu_N, "mu(I)":mu_I, "delta":delta}
            #nonICM_Sorting = (sorted(nonICM.items(), key=lambda x:x[1]))
            nonICM_Sorting = sorted(nonICM, key=nonICM.get)
            concate_nonICM_Sorting = '_'.join(nonICM_Sorting)
            
            
            if explainer.predict(X_test) == 0:   # Meta(I) = 0
                result = Non_ICM_0(concate_nonICM_Sorting)
                if len(result) == 3:
                    operator, _lower_bound, _upper_bound = result
                    lower_bound, upper_bound = nonICM[_lower_bound], nonICM[_upper_bound]                                                             
                
                else: # 不考慮
                    NotUsed_flag = 1
            
            elif explainer.predict(X_test) == 1: # Meta(I) = 1
                result = Non_ICM_1(concate_nonICM_Sorting)
                if len(result) == 3:
                    operator, _lower_bound, _upper_bound = result
                    lower_bound, upper_bound = nonICM[_lower_bound], nonICM[_upper_bound] 
                    
                else: # 不考慮
                    NotUsed_flag = 1
                    
        elif variable in list(_2018ICM_['variable']):  # 2018ICM List
            index = list(_2018ICM_['variable']).index(variable)
            ICM_threshold = _2018ICM_['threshold'][index]
            delta = float(X_test_[variable])
            
            ICM = {"ICM":ICM_threshold, "Q":float(Q), "delta":delta}
            ICM_Sorting = sorted(ICM, key=ICM.get)
            concate_ICM_Sorting = '_'.join(ICM_Sorting)
        
            if explainer.predict(X_test) == 0:   # Meta(I) = 0
                result = ICM_0(concate_ICM_Sorting)
                if len(result) == 3:
                    operator, _lower_bound, _upper_bound = result
                    lower_bound, upper_bound = ICM[_lower_bound], ICM[_upper_bound]                                                             
                
                else: # 不考慮
                    NotUsed_flag = 1
            
            elif explainer.predict(X_test) == 1:   # Meta(I) = 1
                result = ICM_1(concate_ICM_Sorting)
                if len(result) == 3:
                    operator, _lower_bound, _upper_bound = result
                    lower_bound, upper_bound = ICM[_lower_bound], ICM[_upper_bound]                                                             
                
                else: # 不考慮
                    NotUsed_flag = 1
        
        
        if NotUsed_flag == 1:
            decision_list[val] = result
        else:
            decision_list[val] = [variable, operator, lower_bound, upper_bound]
        
    except:
        print("You got an Exception.")
 
    