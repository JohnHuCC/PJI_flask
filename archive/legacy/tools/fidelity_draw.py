import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# v1_x = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# v1_y = [0.8398, 0.8475, 0.8503, 0.8514, 0.8487, 0.8494, 0.8570,
#         0.8579, 0.8542, 0.8490, 0.8439, 0.8462, 0.8503, 0.8496]
# plt.plot(v1_x, v1_y, color='red', marker='.', markersize='16', label='v1')


# v2_x = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# v2_y = [0.9412, 0.9416, 0.9418, 0.9408, 0.9395, 0.9381, 0.9357,
#         0.934, 0.9331, 0.9323, 0.9320, 0.9313, 0.9278, 0.9304]
# plt.plot(v2_x, v2_y, color='orange', marker='.', markersize='16', label='v2')

# v3_x = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# v3_y = [0.9401, 0.9408, 0.9414, 0.9416, 0.9411, 0.9388, 0.9365,
#         0.9346, 0.9337, 0.9327, 0.9323, 0.9315, 0.9311, 0.9305]
# plt.plot(v3_x, v3_y, color='yellow', marker='.', markersize='16', label='v3')

# v4_x = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# v4_y = [0.9431, 0.9429, 0.941, 0.9404, 0.9393, 0.9387, 0.9387,
#         0.9387, 0.9388, 0.9387, 0.9387, 0.9384, 0.9381, 0.9378]
# plt.plot(v4_x, v4_y, color='green', marker='.', markersize='16', label='v4')

# v5_x = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# v5_y = [0.8974, 0.8953, 0.8949, 0.8923, 0.8898, 0.8881, 0.8792, 0.8750,
#         0.8752, 0.8769, 0.8766, 0.8740, 0.8733, 0.8728]

# plt.plot(v5_x, v5_y, color='blue', marker='.', markersize='16', label="v5")
# plt.xlabel('Top number of decision path')
# plt.ylabel('Average fidelity of test data')
# plt.xticks(np.linspace(7, 20, 14))
# plt.yticks(np.linspace(0.80, 0.95, 16))
# plt.legend(['v1', 'v2', 'v3', 'v4', 'v5'])
# plt.title('No restrict singleton numbers')
# plt.show()
# --------------------------------------------------------------------------------------------
f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16 = [0.757, 0.745, 0.770, 0.754], [0.710, 0.739, 0.725, 0.715], [0.698, 0.728, 0.716, 0.701], [0.691, 0.726, 0.715], [0.690, 0.725, 0.713], [
    0.691, 0.726, 0.713], [0.692, 0.726, 0.712], [0.692, 0.726, 0.712, ], [0.692, 0.726, 0.712], [0.692, 0.726, 0.712, 0.705], [0.692, 0.726, 0.712, 0.705], [0.692, 0.726, 0.712, 0.705], [0.692, 0.726, 0.712, 0.705]
for i in range(3, 17):
    print(np.mean('f'+str(i)))
v1_sin_x = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
v1_sin_y = [0.789, 0.725, 0.730, 0.724, 0.732, 0.740, 0.735,
            0.732, 0.732, 0.733, 0.733, 0.733, 0.733, 0.733]
singleton_num = ['7', '10', '11', '12', '16', '18',
                 '19', '17', '17', '17', '16', '16', '16', '16']
# singleton_num = ['5~7/0', '4~10/0', '5~11/0', '4~12/0', '4~12/1', '5~13/3',
#                  '4~13/1', '3~13/1', '3~13/5', '3~12/8', '3~12/8', '3~13/6', '5~13/4', '3~13/4']
for i, singleton_num_item in enumerate(singleton_num):
    plt.text(v1_sin_x[i], v1_sin_y[i], singleton_num_item, fontsize=10,
             verticalalignment="bottom", horizontalalignment="center")
plt.plot(v1_sin_x, v1_sin_y, color='blue',
         marker='.', markersize='16', label='top3')

# plt.xlabel('Depth of decision trees')
# plt.ylabel('Average ACC of rules')
# plt.xticks(np.linspace(3, 16, 14))
# plt.yticks(np.linspace(0.7, 1, 31))
# plt.legend(['all', 'single'])
# plt.title('Average ACC of 5 folds top 3')
# plt.show()

v3_sin_x = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
v3_sin_y = [0.7494, 0.7118, 0.6904, 0.6874, 0.6853, 0.6841, 0.6844,
            0.6844, 0.6844, 0.6845, 0.6844, 0.6844, 0.6844, 0.6844]
singleton_num = ['87', '86', '86', '12', '16', '18',
                 '19', '17', '17', '17', '16', '16', '16', '16']
# singleton_num = ['6~9/0', '7~11/7', '6~13/9', '7~12/19', '5~11/30', '6~12/27',
#                  '8~12/25', '7~12/26', '7~11/33', '7~11/35', '8~11/40', '7~12/33', '7~12/31', '7~12/32']
# for i, singleton_num_item in enumerate(singleton_num):
#     plt.text(v3_sin_x[i], v3_sin_y[i], singleton_num_item, fontsize=10,
#              verticalalignment="baseline", horizontalalignment="center")
plt.plot(v3_sin_x, v3_sin_y, color='red',
         marker='.', markersize='16', label='all')

plt.xlabel('Depth of decision trees')
plt.ylabel('Average ACC of rules')
plt.xticks(np.linspace(3, 16, 14))
plt.yticks(np.linspace(0.65, 1, 36))
plt.legend(['top3', 'all'])
plt.title('Average ACC of 5 folds')
plt.show()

# v5_sin_x = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# v5_sin_y = [15.62, 16.02, 19.75, 16.55, 19.38, 19.85, 19.38,
#             19.74, 2.84, 4.89, 2.61, 21.01, 18.87, 21.25]
# singleton_num = ['10/4', '11/24', '12/34', '12/36', '10/46', '12/42',
#                  '12/43', '11/43', '10/46', '10/46', '9/47', '11/45', '11/45', '11/45']
# # singleton_num = ['8~10/4', '8~11/24', '9~12/34', '9~12/36', '10~10/46', '10~12/42',
# #                  '10~12/43', '10~11/43', '9~10/46', '9~10/46', '9~9/47', '9~11/45', '9~11/45', '9~11/45']
# for i, singleton_num_item in enumerate(singleton_num):
#     plt.text(v5_sin_x[i], v5_sin_y[i], singleton_num_item, fontsize=10,
#              verticalalignment="baseline", horizontalalignment="center")
# plt.plot(v5_sin_x, v5_sin_y, color='orange',
#          marker='.', markersize='16', label="v5")
# plt.xlabel('Depth of decision trees')
# plt.ylabel('Average time of kmap')
# plt.xticks(np.linspace(3, 16, 14))
# plt.yticks(np.linspace(0, 30, 31))
# # plt.legend(['all', 'single'])
# plt.title('Average time of different tree top 7')
# plt.show()

# ---------------------------------------------------------------------------------------------
# v1_x = [8, 9, 10, 11, 12]
# v1_y = [0.9674, 0.9677, 0.9679, 0.9680, 0.9680]
# # for i in range(5):
# #     plt.text(v1_x[i], v1_y[i], singleton_num_item, fontsize=10,
# #              verticalalignment="baseline", horizontalalignment="center")
# plt.plot(v1_x, v1_y, color='blue',
#          marker='.', markersize='16', label='all')

# v3_x = [8, 9, 10, 11, 12]
# v3_y = [0.87, 0.8702, 0.8703, 0.8704, 0.8705]
# # for i in range(5):
# #     plt.text(v3_x[i], v3_y[i], singleton_num_item, fontsize=10,
# #              verticalalignment="baseline", horizontalalignment="center")
# plt.plot(v3_x, v3_y, color='yellow',
#          marker='.', markersize='16', label='single')


# plt.xlabel('Depth of decision trees')
# plt.ylabel('Average fidelity of decision paths')
# plt.xticks(np.linspace(8, 12, 5))
# plt.yticks(np.linspace(0.85, 1.0, 16))
# plt.legend(['all', 'single'])
# plt.title('Average fidelity of different tree depth')
# plt.show()
