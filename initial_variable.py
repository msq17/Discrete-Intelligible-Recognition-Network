# *** Writen by Shiquan Mei ***


import numpy as np

group_dict = {}
start_group_dict = {}
mini_group_dict = {}
train_num = 0
son_mini_max_correct_num = 1
son_mini_mid_correct_num = 1
need_sort = 0
image_number = 0
recycle_start_group_list = []
recycle_group_list = []
recycle_mini_list = []
save_num = 0
activate_length = 0
number_input = 0
total_refine_dict = {}
total_refine_original_dict = {}
# ************* statistics **********
test_signal = 0
comparator_num = 0
adder_num = 0
multiplier_num = 0
total_comparator_num = 0
total_adder_num = 0
total_multiplier_num = 0
total_num = 0

copy_image = np.zeros((28, 28))