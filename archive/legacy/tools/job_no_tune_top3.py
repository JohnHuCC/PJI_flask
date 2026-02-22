import os
import subprocess
import time
New_data = [1311, 151, 8201, 23012, 23752, 42, 331, 531, 541, 581, 631, 671, 1101, 1121, 1291, 1311,
            1331, 1721, 1861, 2351, 2441, 2951, 3211, 3621, 3671, 4111, 4621, 5221, 5291, 5391, 5541,
            1312, 1882, 1912, 2102, 2502, 2892, 5082, 6012, 6142, 6182, 6512,
            6582, 6672, 6852, 6912, 7312, 7332]

top_num_high = [3, 5, 7]
expl_depth = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
for depth in expl_depth:
    for i, top in enumerate(top_num_high):
        for pid in New_data:
            proc = subprocess.Popen(
                ['python3', 'rule_no_tune.py', str(pid), str(top), str(depth)])
            start = time.time()
            while True:
                curr = time.time()
                if curr - start > 75:
                    if not os.path.exists("Decision_rule/simplified_decision_rule_top_"+str(top)+'_depth_'+str(depth)+"_"+str(pid)+".json"):
                        with open('final_result/final_result_no_tune_top_'+str(top)+'_depth_'+str(depth)+'.txt', 'a') as f:
                            f.write('overtime:'+str(75)+':\n')
                        proc.kill()
                        break
                    else:
                        print('ok')
                        try:
                            proc.kill()
                        except:
                            continue
                        break
