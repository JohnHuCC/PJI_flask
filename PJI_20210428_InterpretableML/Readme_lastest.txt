Implement the Explanation Generator

1. fidelity_DecisionPath.py

	In[1], 引用適當的 Library
	In[2], 蒐集跟 stacking model 相同答案的 Decision Tree By "def getCandidate()"
	In[3], 列舉特定樣本的決策路徑 By "def interpret()"
	In[4], 共病症分析 By "def Comorbidity()"
	In[5]: getTopN_Fidelity
	In[6], 主體參數設定
	In[7], File reading and pre-processing
	In[8], feature_selection2 by feature importance from PJI-PI-01-02-2021.docx (Table 4)
	In[9], Stacking Modeling #
	In[10], PID_Trace
	In[11], Randomly generate random forest and candidate tree
	In[12], d_path by feature importance from PJI-PI-01-02-2021.docx (Table 4)
	In[13], Enumerate the mean_fidelitys of decision path and decision tree
	In[14]: Concatenate multi lists for CONDITIOS_AvgFidelity
	fidelity 精確度
	Concatenate 串連 連接

2. POS_Form.py
	# Call the transPOSForm func.
	# Description: Transfer "decision path" with fidelity_DecisionPath.py to 'POS Form'
	# input: candidate decision path
	# output: POS Form rules

	# In[15]: Import Library, 引用適當的 Library
	# In[16]: Declare the function for clasification with ICM & nonICM
	# In[17]: Get the upperbound & lowerbound of sintletons
	# 	      注意：此處須匯入兩個檔案: Non2018ICM.xlsx, 2018ICM.xlsx

	# In[18]: Transfer the decision_path to POS Form
	  POS = product of sum
		# In[19]: 讀取 rules_list, data
		# In[20]: 拆解所有的 decision path 元素
		# In[21]: 列舉 all_singleton 的 features by Set()
		# In[22]: 列舉 all_singleton 的 features by List()
		# In[23]: 列舉所有 decision path的 singleton constraints
		# In[24]: 宣告 Symbols by ns[] for symbols
		# In[25]: decision path 同質性簡化

	# In[26]: Call the transPOSForm func (In[16]).
	# In[27]: Call the singleton_opt function

	# In[28]: return the indices of the each rule with in truth table
	indices = index複數
	# In[29]: POS_minterm process
	# In[30]: call the simplify_logic and get the Simplify_decisionRules

3. dnf_caseX_Trace.RS2.py
    # In[31]: Disassemble all decision path elements
		Disassemble 拆開、拆解
	# In[32]: Randomly specify the value in the variable interval
	# In[33]: Generate the optimized interval range to be tested
	# In[34]: Insert the Singleton header to be analyzed into Excel
	singleton 單張
	# In[35]: Estimate the complexity of the algorithm

	# Notes:
	#這段必須hardcode才知道應該放幾個變數

		singleton_[0], # APTT
		singleton_[1], # P_T
		singleton_[2], # Synovial_WBC
		singleton_[3], # Segment
		singleton_[4], # Serum_CRP
		singleton_[5], # Serum_ESR
