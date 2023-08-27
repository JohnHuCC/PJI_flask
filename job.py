import os
import subprocess
import time

# run_list = [121, 271, 331, 531, 541, 581, 631, 671, 1101, 1121, 1291, 1311, 1331, 1721, 1861, 2351, 2441, 2951, 3211, 3621, 3661, 3671, 3971,
#             4111, 4621, 5221, 5291, 5391, 5541, 5631, 5761, 5841, 5991, 6091, 6101, 6121, 6181, 6411, 6751, 6971, 7011, 7031, 7331, 7361, 7451,
#             7551, 7611, 7831, 8021, 8061, 8131, 8201, 8381, 8491, 8511, 8601, 8631, 8641, 8691, 8841, 8901, 9011, 9031, 9131, 9201, 9301, 9341,
#             9391, 9481, 9991, 10161, 10191, 10221, 11121, 11301, 12401, 12601, 12691, 12701, 42, 62, 1302, 1312, 1882, 1912, 2102, 2502, 2892,
#             4562, 5082, 6012, 6142, 6182, 6512, 6582, 6672, 6852, 6912, 7312, 7332, 7482, 8232, 8782, 8792, 8872, 8942, 11732, 12242, 12632, 12712,
#             12752, 12942, 13402, 13702, 13862, 14042, 14512, 14722, 16172, 16532, 16962, 17512, 17702, 17722, 17762, 18112, 18502, 18722, 18782,
#             18832, 18912, 18932, 19122, 19162, 19242, 19282, 19262, 19322, 19332, 19342, 19352, 19392, 20022, 20142, 20152, 20172, 20182, 20532,
#             20652, 20692, 20732, 20992, 21242, 21312, 21612, 21882, 22472, 22642, 22702, 23012, 23132, 23142, 23152, 23172, 23182, 23202, 23212,
#             23222, 23242, 23292, 23362, 23372, 23452, 23512, 23522, 23602, 23632, 23642, 23672, 23392, 23682, 23742, 23812, 23852, 23892, 23952,
#             23982, 24002, 24032, 24042, 24082, 24092, 24102, 24112, 24162, 24182, 23802, 23862, 24302, 24352, 24362, 24392, 24402, 24422, 24062,
#             24462, 24472, 24152, 21802, 24372, 23752, 24122, 151, 171, 231, 491, 591, 681, 1041, 1081, 1091, 1181, 1191, 1201, 1301, 1411, 1851,
#             2001, 2151, 2321, 2841, 2851, 2971, 3141, 3391, 3431, 3501, 3521, 3871, 3881, 4481, 4631, 4701, 5071, 5251, 5461, 5521, 5911, 6031,
#             6141, 6171, 6381, 6641, 6661, 6711, 6721, 6731, 6771, 6781, 6811, 6871, 6951, 7171, 7261, 9421, 9451, 10381, 12271, 12501, 12751, 12781,
#             2591, 3601, 4451, 6741, 6941, 7911, 9041, 9051, 12331, 12721, 551, 1071, 1261, 1471, 1741, 1941, 2731, 5281, 5331, 5811, 6071, 6291, 6391,
#             7751, 8701, 8861, 9771, 11281, 1371, 1551, 1931, 2081, 2211, 2511, 2601, 2931, 3001, 3081, 3551, 4441, 4681, 4941, 4981, 5021, 5481, 5921,
#             6001, 6021, 6131, 6261, 11291, 2401]
from random import sample
import pandas as pd
X_res_test = pd.read_csv('PJI_Dataset/New_data_x_test.csv', encoding='utf-8')
no_group = list(X_res_test['No.Group'])
print('Get 20 random samples:')
run_id = sample(no_group, 20)
print(run_id)
print('\n-------------------------\n')
with open('test_out.txt', 'a') as f:
    f.write('Get 20 random samples:\n')
    f.write(str(run_id))
    f.write('\n-------------------------\n')
tree_nums = [7, 10, 13]
for tree in tree_nums:
    with open('test_out.txt', 'a') as f:
        f.write('top '+str(tree)+':\n')
    for run in run_id:
        print('run: ', run)
        with open('test_out.txt', 'a') as f:
            f.write('run: '+str(run)+':\n')
        proc = subprocess.Popen(
            ['python3', 'smote.py', str(run), str(tree)])
        start = time.time()
        while True:
            curr = time.time()
            if curr - start > tree * tree:
                if not os.path.exists('decision_rule_'+str(run)+'.json'):
                    with open('test_out.txt', 'a') as f:
                        f.write('overtime:'+str(tree*tree)+':\n')
                    proc.kill()
                    break
                else:
                    print('ok')
                    try:
                        proc.kill()
                    except:
                        continue
                    break
