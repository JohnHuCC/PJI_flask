import os
import subprocess
import time

run_list = [212, 232, 242, 252, 262, 272, 282, 292, 302, 322, 332, 342]
for run in run_list:
    print('run: ', run)
    proc = subprocess.Popen(
        ['python3', 'personal_DecisionPath2.py', str(run)])
    start = time.time()
    while True:
        curr = time.time()
        if curr - start > 6 * 60:  # 5 minutes
            if not os.path.exists('Decision_rule/decision_rule_'+str(run)+'.json'):
                proc.kill()
                break
            else:
                print('ok')
                try:
                    proc.kill()
                except:
                    continue
                break
