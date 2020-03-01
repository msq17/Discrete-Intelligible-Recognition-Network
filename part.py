# *** Writen by Shiquan Mei ***


import numpy as np
import matplotlib.pyplot as plt
import math
import initial_variable
from model_list import *
import copy


class generator_entity(object):
    def __init__(self, group_label, image_save_dict, feature_num, father_mini,
                 scanning_image_dict, scanning_image_or_dict):
        self.correct_num = 1
        self.group_label = group_label
        self.feature_num = feature_num
        self.life_num = 3
        self.scanning_size = 2
        self.image_save_dict = image_save_dict
        self.result_total = []
        self.result_dict = {}
        self.activate_dict = {}
        self.total_save_dict = {}
        self.total_boundary_dict = {}
        self.ratio = 0
        self.recognition_ratio = 0
        self.scanning_image_dict = scanning_image_dict
        self.scanning_image_or_dict = scanning_image_or_dict
        self.max_image_save_point = 1
        self.total_train_num = 0
        self.continue_num = 0
        self.correct = 0
        self.increase_ratio = 2
        self.can_refine = 1
        self.refine_allow = 0
        self.allow_trans_son_dict = {}
        self.allow_mini_dict = {}
        self.can_refine_boundary = 6
        self.refine_momentum = 6
        self.ti = 0
        self.tj = 0
        self.save_end_dict = {}
        self.success = 0
        self.replace_save = {}
        self.save_ij_list = {}
        self.execute = 0
        self.delete = 0
        self.delete_num = 0
        self.find_neighbor_execute = 0
        self.merge_common_execute = []
        self.correct_writen_execute = 0
        self.match_success_signal = 0
        self.replace_signal = 0
        self.activate_father_mini = {}
        self.trans_son_total = {}
        self.trans_replace_num = {}
        self.trans_replace_max = {}
        self.trans_son_total_num = {}
        self.trans_num = {}
        self.trans_start_num = {}
        self.trans_base_num = {}
        self.activate_son = {}
        self.trans_total_num = {}
        self.trans_activate_num = {}
        self.trans_son_max_group = {}
        self.trans_father_total = {}
        self.trans_son_total_number = 8
        self.mid_trans_point = 4
        self.father_mini = father_mini
        self.father_mini_dict = {}
        self.recognition_path = []
        self.trans_father_keep_dict = {}
        self.sort_trans_father = {}
        self.max_trans_num = 1
        self.mid_trans_num = 1
        self.max_self_son = 1
        self.point_max_serve = {}
        self.point_max_sort = []

    def compare_recognition(self, image_test_dict, trans_save_dict, ij_list, son_signal=0):
        """
        Matching function with convolution: This function uses convolution to measure the degree of matching between
        the input and the memory
        Considering that the numbers in the MNIST data set are all approximately the same position, the convolution scan
        is only a simple scan around the original position to reduce the amount of calculation
        :param image_test_dict: boundaries extracted from input
        :param trans_save_dict: saved recognized input boundaries
        :param ij_list: best match position for key part
        :param son_signal: used to mark key part and son part, 0 for key part and 0 for son part
        :return: if len(return) more than 0, it represent matching ratio between the input and the memory more than
        recognition_ratio
        """
        if self.execute == 0 or (len(ij_list) > 0 and ij_list not in self.save_ij_list and self.success == 0):
            self.recognition_ratio = 0.4
            if son_signal == 0:
                self.result_total = recognition_input(self.scanning_size, self.image_save_dict,
                                                      image_test_dict, trans_save_dict, self.life_num,
                                                      self.feature_num,
                                                      self.scanning_image_dict,
                                                      self.recognition_ratio)
                self.execute = 1
                if len(self.result_total) > 0:
                    self.ratio = self.result_total[0] / self.feature_num
                    self.result_dict = self.result_total[1]
                    self.activate_dict = self.result_total[2]
                    self.total_save_dict = self.result_total[3]
                    self.total_boundary_dict = self.result_total[4]
                    self.ti = self.result_total[5][0]
                    self.tj = self.result_total[5][1]
                return self.result_total
            else:
                self.execute = 1
                self.save_ij_list[ij_list] = 1
                self.result_total = son_recognition_input(ij_list, self.image_save_dict, image_test_dict,
                                                          trans_save_dict, self.life_num,
                                                          self.feature_num,
                                                          self.scanning_image_dict,
                                                          self.recognition_ratio)
                if len(self.result_total) > 0:
                    self.ratio = self.result_total[0] / self.feature_num
                    self.result_dict = self.result_total[1]
                    self.activate_dict = self.result_total[2]
                    self.total_save_dict = self.result_total[3]
                    self.total_boundary_dict = self.result_total[4]
                return self.result_total
        elif self.success == 1 and len(self.total_save_dict) > 0:
            save_activate_number = 0
            for (y_save, x_save) in self.total_save_dict:
                if (y_save, x_save) not in trans_save_dict:
                    save_activate_number += 1
            if self.ratio * (save_activate_number / len(self.total_save_dict)) >= self.recognition_ratio:
                return self.result_total
            else:
                return []
        else:
            return self.result_total

    def merge_common(self, image_name_dict):
        """This function writes self name in image_name_dict to represent which memory points match input points"""
        merge_common_f(self.group_label, self.total_save_dict, image_name_dict)

    def match_success(self):
        if self.match_success_signal == 0:
            self.match_success_signal = 1
        else:
            return

    def correct_num_writen(self, correct):
        """This function is used to update the number of activations and arrange key parts and son parts
        according to the number of activations. We will save the key part frequently activated and a certain
        number of son part
        replace_key is used to update the program in future, it is currently not useful
        """
        if self.correct_writen_execute == 0:
            self.correct_writen_execute = 1
        else:
            return
        self.correct = correct
        if self.success == 1:
            execute_update_point = 0
            if (self.refine_allow == 1 and self.correct_num >= self.can_refine_boundary) or self.correct_num % 3 == 0:
                execute_update_point = 1
            max_image_save_point = update_save_image_activate_num(self.image_save_dict, self.activate_dict,
                                                                  self.max_image_save_point, correct,
                                                                  self.scanning_image_dict,
                                                                  self.scanning_image_or_dict, execute_update_point,
                                                                  self.point_max_serve, self.point_max_sort,
                                                                  self.feature_num)
            if max_image_save_point > self.max_image_save_point:
                self.max_image_save_point = max_image_save_point
            if correct >= 1:
                self.correct_num += 1
                if initial_variable.train_num - self.correct_num <= self.total_train_num + 2:
                    self.continue_num += 1
                else:
                    self.total_train_num = initial_variable.train_num - self.correct_num
                    self.continue_num = 0
            else:
                self.correct_num -= 2
        if self.success == 1:
            if correct >= 1:
                if len(self.activate_son) > 0:
                    for son_group in self.activate_son:
                        if self.trans_son_total_num[son_group][0] != -1:
                            if self.trans_son_total_num[son_group][0] < self.mid_trans_num:
                                self.trans_son_total_num[son_group][0] += self.trans_son_total_num[son_group][1]
                                if self.trans_son_total_num[son_group][0] > self.mid_trans_num:
                                    self.trans_son_total_num[son_group][0] = self.mid_trans_num
                            elif self.trans_son_total_num[son_group][1] > 0:
                                self.trans_son_total_num[son_group][0] += 1
                            self.trans_son_total_num[son_group][1] += 1
                        else:
                            self.trans_son_total_num[son_group][0] = self.mid_trans_num + 2

                        for start_group_key in self.activate_son[son_group]:
                            [replace_key, group_key] = self.activate_son[son_group][start_group_key]

                            if self.trans_num[son_group][start_group_key][replace_key][group_key][0] != -1:
                                self.trans_num[son_group][start_group_key][replace_key][group_key][0] += 1
                            if self.trans_start_num[son_group][start_group_key] != -1:
                                self.trans_start_num[son_group][start_group_key] += 1
                        break
            else:
                for son_group in self.activate_son:
                    if self.trans_son_total_num[son_group][0] != -1:
                        self.trans_son_total_num[son_group][0] -= 2
                    self.trans_son_total_num[son_group][1] -= 2
                    self.trans_son_total_num[son_group][2] += 1
                    for start_group_key in self.activate_son[son_group]:
                        [replace_key, group_key] = self.activate_son[son_group][start_group_key]
                        if self.trans_num[son_group][start_group_key][replace_key][group_key][0] != -1:
                            self.trans_num[son_group][start_group_key][replace_key][group_key][0] -= 2
                        if self.trans_start_num[son_group][start_group_key] != -1:
                            self.trans_start_num[son_group][start_group_key] -= 2

            # ************** update total son_trans *************
            max_group_signal = 0
            list_m = []
            del_son_group = []
            initial_trans_list = []
            for trans_son_key in self.trans_son_total_num:
                if self.trans_son_total_num[trans_son_key][0] == -1:
                    initial_trans_list.append(trans_son_key)
                    continue
                list_m.append([self.trans_son_total_num[trans_son_key][0], trans_son_key])
            list_m.sort(key=lambda x: x[0], reverse=True)

            if len(list_m) > 0 and list_m[0][0] > 0:
                self.max_trans_num = list_m[0][0]
                if list_m[0][1] in self.activate_son:
                    if list_m[0][1] not in self.trans_son_max_group:
                        self.trans_son_max_group = {list_m[0][1]: {}}
                    max_group_signal += 1
            if len(list_m) > self.mid_trans_point - 1:
                self.mid_trans_num = list_m[self.mid_trans_point - 1][0]
                for initial_trans_key in initial_trans_list:
                    list_m.insert(self.mid_trans_point - 1, [self.mid_trans_num, initial_trans_key])
                    self.trans_son_total_num[initial_trans_key][0] = self.mid_trans_num
            else:
                mid_number = len(list_m)
                if mid_number < 1:
                    mid_number = 0
                else:
                    self.mid_trans_num = list_m[mid_number - 1][0] - 1
                for initial_trans_key in initial_trans_list:
                    list_m.insert(mid_number, [self.mid_trans_num, initial_trans_key])
                    self.trans_son_total_num[initial_trans_key][0] = self.mid_trans_num

            trans_son_copy = copy.deepcopy(self.trans_son_total)
            self.trans_son_total.clear()
            save_number = 0
            for list_m_i in list_m:
                if save_number <= self.trans_son_total_number:
                    save_number += 1
                    self.trans_son_total[list_m_i[1]] = trans_son_copy[list_m_i[1]]
                else:
                    del_son_group.append(list_m_i[1])
            if len(del_son_group) > 0:
                for son_group_temp in del_son_group:
                    self.trans_son_total_num.pop(son_group_temp)
                    self.trans_start_num.pop(son_group_temp)
                    self.trans_num.pop(son_group_temp)
                    self.trans_replace_num.pop(son_group_temp)
                    self.trans_replace_max.pop(son_group_temp)
                    self.trans_total_num.pop(son_group_temp)
                    self.trans_activate_num.pop(son_group_temp)
                    if son_group_temp in self.trans_base_num:
                        self.trans_base_num.pop(son_group_temp)

            # ****************** update trans_son ***************
            sort_list = []
            assignment_value = []
            del_list = []
            first_time = 0
            replace_trans_group = []
            for son_group in self.activate_son:
                if son_group not in self.trans_son_total:
                    continue
                for_trans_son_total = self.trans_son_total[son_group].copy()
                max_start_num = 1
                start_group_assignment_value = []
                del_start_list = []
                for start_group_key in for_trans_son_total:
                    if max_start_num < self.trans_start_num[son_group][start_group_key]:
                        max_start_num = self.trans_start_num[son_group][start_group_key]
                for start_group_key in for_trans_son_total:
                    if self.trans_start_num[son_group][start_group_key] == -1:
                        start_group_assignment_value.append(start_group_key)
                        continue
                    if start_group_key in self.activate_son[son_group]:
                        if max_group_signal == 1:
                            if max_start_num == self.trans_start_num[son_group][start_group_key]:
                                if start_group_key not in self.trans_son_max_group:
                                    self.trans_son_max_group[son_group] = {start_group_key: {}}
                                max_group_signal += 1
                        if max_start_num - self.trans_start_num[son_group][start_group_key] > 5:
                            self.trans_start_num[son_group][start_group_key] = max_start_num - 5
                    else:
                        if max_start_num - self.trans_start_num[son_group][start_group_key] > 10:
                            del_start_list.append(start_group_key)
                        continue
                    [replace_key, group_key] = self.activate_son[son_group][start_group_key]
                    sort_list.clear()
                    del_list.clear()
                    if len(self.trans_son_total[son_group][start_group_key][replace_key]) > 1:
                        small_group_max = 1
                        for group_key in self.trans_num[son_group][start_group_key][replace_key]:
                            son_correct_num = self.trans_num[son_group][start_group_key][replace_key][group_key][0]
                            if son_correct_num > small_group_max:
                                small_group_max = son_correct_num
                            if son_correct_num == -1:
                                assignment_value.append(group_key)
                                continue
                            sort_list.append([son_correct_num, group_key])

                        sort_list.sort(key=lambda x: x[0], reverse=True)
                        small_group_copy = copy.deepcopy(self.trans_son_total[son_group][start_group_key][replace_key])
                        trans_num_copy = copy.deepcopy(self.trans_num[son_group][start_group_key][replace_key])
                        trans_son_total_all = copy.deepcopy(self.trans_son_total[son_group])
                        self.trans_son_total[son_group][start_group_key][replace_key].clear()
                        self.trans_num[son_group][start_group_key][replace_key].clear()
                        for group_key_temp in assignment_value:
                            self.trans_son_total[son_group][start_group_key][replace_key][group_key_temp] = \
                                small_group_copy[group_key_temp]
                            self.trans_num[son_group][start_group_key][replace_key][group_key_temp] = [small_group_max, 2]

                        execute_signal = 0
                        if len(sort_list) >= 1:
                            if max_group_signal == 3:
                                if sort_list[0][1] == group_key and group_key not in \
                                        self.trans_son_max_group[son_group][start_group_key][replace_key]:
                                    self.trans_son_max_group[son_group][start_group_key][replace_key] = {group_key: 1}
                        if len(sort_list) > 1:
                            if small_group_max - sort_list[1][0] > 6:
                                execute_signal = 1
                            elif small_group_max - self.trans_base_num[son_group][start_group_key][replace_key] > 20:
                                execute_signal = 2

                        if execute_signal == 0:
                            for list_temp in sort_list:
                                self.trans_son_total[son_group][start_group_key][replace_key][list_temp[1]] = \
                                    small_group_copy[list_temp[1]]
                                self.trans_num[son_group][start_group_key][replace_key][list_temp[1]] = \
                                    trans_num_copy[list_temp[1]]
                        else:
                            del_list.append(sort_list[1][1])
                            if len(small_group_copy[sort_list[0][1]]) > 1:
                                for replace_key_temp in self.trans_son_total[son_group][start_group_key]:
                                    if replace_key_temp != replace_key:
                                        for group_key_temp in self.trans_son_total[son_group][start_group_key][replace_key_temp]:
                                            for group_label_temp in \
                                                    self.trans_son_total[son_group][start_group_key][replace_key_temp][
                                                        group_key_temp]:
                                                replace_trans_group.append([group_label_temp, start_group_key])
                                self.trans_son_total[son_group].pop(start_group_key)
                                self.trans_num[son_group].pop(start_group_key)
                                self.trans_replace_num[son_group].pop(start_group_key)
                                if start_group_key in self.trans_replace_max[son_group]:
                                    self.trans_replace_max[son_group].pop(start_group_key)
                                self.trans_start_num[son_group].pop(start_group_key)
                                self.trans_total_num[son_group] -= self.trans_activate_num[son_group][start_group_key]
                                self.trans_activate_num[son_group].pop(start_group_key)
                                self.trans_base_num[son_group].pop(start_group_key)
                                for group_other in small_group_copy[sort_list[0][1]]:
                                    start_group_name = group_other
                                    while start_group_name in self.trans_son_total[son_group]:
                                        start_group_num = int(start_group_name)
                                        start_group_num += 1
                                        start_group_name = str(start_group_num)
                                    self.trans_son_total[son_group][start_group_name] = {"0": {"0": [group_other]}}
                                    self.trans_num[son_group][start_group_name] = {"0": {"0": [1, 2]}}
                                    self.trans_replace_num[son_group][start_group_name] = {"0": [1, 2]}
                                    self.trans_replace_max[son_group][start_group_name] = 1
                                    self.trans_start_num[son_group][start_group_name] = -1
                                    self.trans_total_num[son_group] += start_group_dict[group_other].feature_num
                                    self.trans_activate_num[son_group][start_group_name] = \
                                        start_group_dict[group_other].feature_num
                                    group_ob_other = start_group_dict[group_other]
                                    group_ob_other.trans_father_total[self.group_label][son_group] = \
                                        {start_group_name: {"0": "0"}}
                                del_list.append(sort_list[0][1])
                            else:
                                self.trans_son_total[son_group][start_group_key][replace_key] = \
                                    {sort_list[0][1]: small_group_copy[sort_list[0][1]]}
                                self.trans_num[son_group][start_group_key][replace_key] = \
                                    {sort_list[0][1]: trans_num_copy[sort_list[0][1]]}

                            if execute_signal > 1:
                                other_son_group = str(len(self.trans_son_total))
                                while other_son_group in self.trans_son_total:
                                    other_son_group_num = int(other_son_group)
                                    other_son_group_num += 1
                                    other_son_group = str(other_son_group_num)
                                self.trans_son_total_num[other_son_group] = \
                                    copy.deepcopy(self.trans_son_total_num[son_group])
                                self.trans_son_total[other_son_group] = {}
                                self.trans_replace_num[other_son_group] = {}
                                self.trans_replace_max[other_son_group] = {}
                                self.trans_num[other_son_group] = {}
                                self.trans_start_num[other_son_group] = {}
                                self.trans_total_num[other_son_group] = 0
                                self.trans_activate_num[other_son_group] = {}
                                self.trans_base_num[other_son_group] = {}
                                for start_group_other in trans_son_total_all:
                                    if start_group_other == start_group_key:
                                        continue
                                    self.trans_son_total[other_son_group][start_group_other] = copy.deepcopy(
                                        self.trans_son_total[son_group][start_group_other])
                                    self.trans_num[other_son_group][start_group_other] = copy.deepcopy(
                                        self.trans_num[son_group][start_group_other])
                                    self.trans_replace_num[other_son_group][start_group_other] = copy.deepcopy(
                                        self.trans_replace_num[son_group][start_group_other])
                                    if start_group_other in self.trans_replace_max[son_group]:
                                        self.trans_replace_max[other_son_group][start_group_other] = copy.deepcopy(
                                            self.trans_replace_max[son_group][start_group_other])
                                    self.trans_start_num[other_son_group][start_group_other] = \
                                        self.trans_start_num[son_group][start_group_other]
                                    self.trans_total_num[other_son_group] += \
                                        self.trans_activate_num[son_group][start_group_other]
                                    self.trans_activate_num[other_son_group][start_group_other] = \
                                        self.trans_activate_num[son_group][start_group_other]
                                    if son_group in self.trans_base_num and \
                                            start_group_other in self.trans_base_num[son_group]:
                                        self.trans_base_num[other_son_group][start_group_other] = \
                                            self.trans_base_num[son_group][start_group_other]
                                    for replace_key_temp in self.trans_son_total[son_group][start_group_other]:
                                        for group_key_other in self.trans_son_total[son_group][start_group_other][replace_key_temp]:
                                            for group_other in self.trans_son_total[son_group][start_group_other][replace_key_temp][group_key_other]:
                                                group_ob_other = start_group_dict[group_other]
                                                group_ob_other.trans_father_total[self.group_label][other_son_group] = \
                                                    {start_group_other: {replace_key_temp: group_key_other}}
                                for group_other in small_group_copy[sort_list[1][1]]:
                                    start_group_name = group_other
                                    group_ob_other = start_group_dict[group_other]
                                    while start_group_name in self.trans_son_total[other_son_group]:
                                        start_group_num = int(start_group_name)
                                        start_group_num += 1
                                        start_group_name = str(start_group_num)
                                    self.trans_son_total[other_son_group][start_group_name] = {"0": {"0": [group_other]}}
                                    self.trans_num[other_son_group][start_group_name] = {"0": {"0": [1, 2]}}
                                    self.trans_replace_num[other_son_group][start_group_name] = {"0": [1, 2]}
                                    self.trans_replace_max[other_son_group][start_group_name] = 1
                                    self.trans_start_num[other_son_group][start_group_name] = -1
                                    self.trans_total_num[other_son_group] += group_ob_other.feature_num
                                    self.trans_activate_num[other_son_group][start_group_name] = \
                                        group_ob_other.feature_num
                                    group_ob_other.trans_father_total[self.group_label][other_son_group] = \
                                        {start_group_name: {"0": "0"}}
                            if len(del_list) > 0:
                                for group_key in del_list:
                                    for start_group in small_group_copy[group_key]:
                                        replace_trans_group.append([start_group, start_group_key])
                    else:
                        if self.trans_num[son_group][start_group_key][replace_key][group_key][0] < 1:
                            self.trans_num[son_group][start_group_key][replace_key][group_key][0] = 1
                        if max_group_signal == 3:
                            if group_key not in \
                                    self.trans_son_max_group[son_group][start_group_key][replace_key]:
                                self.trans_son_max_group[son_group][start_group_key][replace_key] = {group_key: 1}
                        if self.trans_replace_max[son_group][start_group_key] == \
                                self.trans_replace_num[son_group][start_group_key][replace_key][0]:
                            trans_base_num = self.trans_num[son_group][start_group_key][replace_key][group_key][0]
                            if son_group not in self.trans_base_num:
                                self.trans_base_num[son_group] = {start_group_key: {replace_key: trans_base_num}}
                            elif start_group_key not in self.trans_base_num[son_group]:
                                self.trans_base_num[son_group][start_group_key] = {replace_key: trans_base_num}
                            else:
                                self.trans_base_num[son_group][start_group_key][replace_key] = trans_base_num
                            for son_group_label in self.trans_son_total[son_group][start_group_key][replace_key][group_key]:
                                if first_time == 0:
                                    son_group_ob = start_group_dict[son_group_label]
                                    son_group_ob.refine_allow = 1
                                    if (self.group_label, son_group) not in son_group_ob.allow_trans_son_dict:
                                        son_group_ob.allow_trans_son_dict[(self.group_label, son_group)] = \
                                            [1, start_group_key, replace_key]
                                    else:
                                        son_group_ob.allow_trans_son_dict[(self.group_label, son_group)][0] += 1
                    first_time = 1
                if len(start_group_assignment_value) > 0:
                    for start_group_key in start_group_assignment_value:
                        self.trans_start_num[son_group][start_group_key] = max_start_num
                if len(del_start_list) > 0:
                    for start_group_key in del_start_list:
                        for replace_key in self.trans_son_total[son_group][start_group_key]:
                            for group_key in self.trans_son_total[son_group][start_group_key][replace_key]:
                                for trans_key in self.trans_son_total[son_group][start_group_key][replace_key][group_key]:
                                    replace_trans_group.append([trans_key, start_group_key])
                        self.trans_start_num[son_group].pop(start_group_key)
                        self.trans_son_total[son_group].pop(start_group_key)
                        self.trans_num[son_group].pop(start_group_key)
                        self.trans_replace_num.pop(start_group_key)
                        if start_group_key in self.trans_replace_max:
                            self.trans_replace_max.pop(start_group_key)
                        self.trans_total_num[son_group] -= self.trans_activate_num[son_group][start_group_key]
                        if son_group in self.trans_base_num and start_group_key in self.trans_base_num[son_group]:
                            self.trans_base_num[son_group].pop(start_group_key)
                        self.trans_activate_num[son_group].pop(start_group_key)

                if len(replace_trans_group) > 0:
                    for [start_group, start_group_key] in replace_trans_group:
                        start_group_ob = start_group_dict[start_group]
                        if self.group_label in start_group_ob.trans_father_total:
                            if son_group in start_group_ob.trans_father_total[self.group_label]:
                                if start_group_key in \
                                        start_group_ob.trans_father_total[self.group_label][son_group]:
                                    start_group_ob.trans_father_total[self.group_label][son_group].pop(start_group_key)
                                if len(start_group_ob.trans_father_total[self.group_label][son_group]) == 0:
                                    start_group_ob.trans_father_total[self.group_label].pop(son_group)
                            if len(start_group_ob.trans_father_total[self.group_label]) == 0:
                                start_group_ob.trans_father_total.pop(self.group_label)
                                start_group_ob.sort_trans_father.pop(self.group_label)
                        if len(start_group_ob.trans_father_total) == 0 and start_group_ob.delete == 0 and \
                                len(start_group_ob.father_mini_dict) == 0:
                            del_group(start_group)
            for son_trans_key in del_son_group:
                if son_trans_key in trans_son_copy:
                    for son_start_group_key in trans_son_copy[son_trans_key]:
                        for replace_key in trans_son_copy[son_trans_key][son_start_group_key]:
                            for son_group_key in trans_son_copy[son_trans_key][son_start_group_key][replace_key]:
                                for trans_key in trans_son_copy[son_trans_key][son_start_group_key][replace_key][son_group_key]:
                                    trans_group_ob = start_group_dict[trans_key]
                                    if self.group_label in trans_group_ob.trans_father_total:
                                        if son_trans_key in trans_group_ob.trans_father_total[self.group_label]:
                                            trans_group_ob.trans_father_total[self.group_label].pop(son_trans_key)
                                        if len(trans_group_ob.trans_father_total[self.group_label]) == 0:
                                            trans_group_ob.trans_father_total.pop(self.group_label)
                                            if self.group_label in trans_group_ob.sort_trans_father:
                                                trans_group_ob.sort_trans_father.pop(self.group_label)
                                    if len(trans_group_ob.trans_father_total) == 0 and trans_group_ob.delete == 0 \
                                            and len(trans_group_ob.father_mini_dict) == 0:
                                        del_group(trans_key)

    def refine_self_mini_group(self):
        """This function refine some small parts from original part and generate new group_key to save these small parts.
        self.can_refine variable indicates whether this memory block can continue to be refined.Some memory blocks
        are too small to continue to be refined.
        """
        if self.success != 1 or self.can_refine == 0 or self.ratio > 0.95 or self.continue_num > 8\
                or self.refine_allow == 0 or self.correct_num < self.can_refine_boundary:
            self.refine_allow = 0
            return

        self.refine_allow = 0  # initialize variable refine_allow
        self.can_refine_boundary = self.correct_num + self.refine_momentum
        start_ratio = 0.45
        end_ratio = 0.6

        refine_total_list = []
        extend_total_list = []
        if refine_group_update(self.image_save_dict, self.life_num, start_ratio, end_ratio,
                               self.feature_num, refine_total_list, extend_total_list,
                               self.point_max_serve, self.point_max_sort):
            refine_save_image_list = []
            create_refine_group(refine_total_list, refine_save_image_list, self.image_save_dict,
                                self.scanning_image_dict)
            father_group_temp = []
            small_group = {}
            aim_dict = {}
            father_group_list = []
            if len(self.allow_mini_dict) > 0:
                execute_signal = 0
                sort_list = []
                for father_mini_label in self.allow_mini_dict:
                    sort_list.append([self.allow_mini_dict[father_mini_label][0], father_mini_label])
                sort_list.sort(key=lambda x: x[0], reverse=True)
                father_mini_label = sort_list[0][1]
                father_mini_ob = initial_variable.mini_group_dict[father_mini_label]
                father_mini = [father_mini_label]
                small_group[father_mini_label] = father_mini_ob.small_group
                start_group_label = self.allow_mini_dict[father_mini_label][1]
                replace_key = self.allow_mini_dict[father_mini_label][2]
                aim_dict[father_mini_label] = {start_group_label: {replace_key: []}}
            else:
                execute_signal = 1
                sort_list = []
                for temp_key in self.allow_trans_son_dict:
                    sort_list.append([self.allow_trans_son_dict[temp_key][0], temp_key])
                sort_list.sort(key=lambda x: x[0], reverse=True)
                (father_trans_label, son_group) = sort_list[0][1]
                father_trans_group_ob = start_group_dict[father_trans_label]
                father_mini = copy.deepcopy(father_trans_group_ob.father_mini)
                small_group[son_group] = father_trans_group_ob.trans_son_total[son_group]
                start_group_key = self.allow_trans_son_dict[(father_trans_label, son_group)][1]
                replace_key = self.allow_trans_son_dict[(father_trans_label, son_group)][2]
                father_group_list = [father_trans_label]
                aim_dict[son_group] = {start_group_key: {replace_key: []}}
            for list_temp in refine_save_image_list:
                image_save_dict = list_temp[0]
                feature_num = list_temp[1]
                scanning_image_dict = list_temp[2]
                scanning_image_or_dict = list_temp[3]
                group_label, length_dict = create_group_label(generator_entity,
                                                              image_save_dict,
                                                              feature_num, father_mini,
                                                              father_group_list, {}, scanning_image_dict,
                                                              scanning_image_or_dict,
                                                              small_group, aim_dict, 0, execute_signal)
                start_generator_ob = start_group_dict[group_label]
                self.replace_save[group_label] = 0
                start_generator_ob.replace_save[self.group_label] = 0
                father_group_temp.append(group_label)
            update_trans_son_list = father_group_temp
            for small_group in update_trans_son_list:
                small_group_ob = start_group_dict[small_group]
                small_group_ob.trans_son_total_num = copy.deepcopy(self.trans_son_total_num)
                small_group_ob.trans_son_total = copy.deepcopy(self.trans_son_total)
                small_group_ob.trans_num = copy.deepcopy(self.trans_num)
                small_group_ob.trans_replace_num = copy.deepcopy(self.trans_replace_num)
                small_group_ob.trans_replace_max = copy.deepcopy(self.trans_replace_max)
                small_group_ob.trans_start_num = copy.deepcopy(self.trans_start_num)
                small_group_ob.trans_total_num = copy.deepcopy(self.trans_total_num)
                small_group_ob.trans_activate_num = copy.deepcopy(self.trans_activate_num)
                small_group_ob.trans_base_num = copy.deepcopy(self.trans_base_num)
                for son_group in self.trans_son_total:
                    for start_group_other in small_group_ob.trans_son_total[son_group]:
                        for replace_other in small_group_ob.trans_son_total[son_group][start_group_other]:
                            for group_key_other in small_group_ob.trans_son_total[son_group][start_group_other][replace_other]:
                                for group_other in small_group_ob.trans_son_total[son_group][start_group_other][replace_other][group_key_other]:
                                    group_ob_other = start_group_dict[group_other]
                                    if small_group not in group_ob_other.sort_trans_father:
                                        group_ob_other.sort_trans_father[small_group] = -1
                                    if small_group not in group_ob_other.trans_father_total:
                                        group_ob_other.trans_father_total[small_group] = \
                                            {son_group: {start_group_other: {replace_other: group_key_other}}}
                                    else:
                                        group_ob_other.trans_father_total[small_group][son_group] = \
                                            {start_group_other: {replace_other: group_key_other}}
        else:
            self.replace_signal += 1
        self.allow_trans_son_dict.clear()
        self.allow_mini_dict.clear()

    def update_memory(self, image_test_dict):
        """This function is used to update memory dict
        """
        if self.success != 1 or self.find_neighbor_execute == 1:
            return

        self.find_neighbor_execute = 1

        refine_activate_num = self.max_image_save_point - 20
        if refine_activate_num < 0:
            refine_activate_num = 0
        if self.ratio < 0.93:
            merge_image(self.result_dict, image_test_dict, self.image_save_dict, refine_activate_num,
                        self.total_save_dict, self.scanning_image_dict, self.scanning_image_or_dict,
                        self.total_boundary_dict)

    def parameter_initial(self, execute_initial=0):
        """This function is used to initialize related parameters to prepare for the next run.
        If this class is deleted,the parameters will no longer be updated."""
        if self.delete == 1:
            return
        self.result_total.clear()
        self.execute = 0
        self.success = 0
        self.recognition_path.clear()
        self.result_dict.clear()
        self.activate_dict.clear()
        self.ratio = 0
        self.recognition_ratio = 0
        self.activate_dict.clear()
        self.total_save_dict.clear()
        self.total_boundary_dict.clear()
        self.save_ij_list.clear()
        self.activate_father_mini.clear()

        self.ti = 0
        self.tj = 0
        self.save_end_dict.clear()
        self.result_dict.clear()
        self.activate_son.clear()

        self.find_neighbor_execute = 0
        self.merge_common_execute.clear()
        self.correct_writen_execute = 0
        self.match_success_signal = 0

        self.point_max_serve.clear()
        self.point_max_sort.clear()
        if execute_initial == 0:
            replace_save_copy = copy.deepcopy(self.replace_save)
            for other_group_label in replace_save_copy:
                other_group_ob = start_group_dict[other_group_label]
                if other_group_ob.delete == 1:
                    self.replace_save.pop(other_group_label)

            if len(self.allow_mini_dict) > 0:
                allow_mini_dict_copy = self.allow_mini_dict.copy()
                for father_mini in allow_mini_dict_copy:
                    father_mini_ob = mini_group_dict[father_mini]
                    if father_mini_ob.delete == 1:
                        self.allow_mini_dict.pop(father_mini)
            elif len(self.allow_trans_son_dict) > 0:
                allow_trans_son_dict_copy = self.allow_trans_son_dict.copy()
                for father_group_key in allow_trans_son_dict_copy:
                    (father_group, son_group) = father_group_key
                    father_group_ob = start_group_dict[father_group]
                    if father_group_ob.delete == 1 or son_group not in father_group_ob.trans_son_total:
                        self.allow_trans_son_dict.pop(father_group_key)

            if len(self.trans_father_keep_dict) > 0:
                if self.correct == 1:
                    correct = 1
                else:
                    correct = -1
                for father_group in self.trans_father_keep_dict:
                    if father_group in self.sort_trans_father:
                        if self.sort_trans_father[father_group] > 0:
                            self.sort_trans_father[father_group] += correct
                    else:
                        self.sort_trans_father[father_group] = -1
                sort_list = []
                assign_list = []
                for father_group in self.sort_trans_father:
                    sort_list.append([self.sort_trans_father[father_group], father_group])
                sort_list.sort(key=lambda x: x[0], reverse=True)
                max_number = sort_list[0][0]
                if max_number < 1:
                    max_number = 1
                sort_list.pop(0)
                for other_list in sort_list:
                    if other_list[0] == -1:
                        assign_list.append(other_list[1])
                    elif max_number - other_list[0] > 6:
                        del_trans_father(self.group_label, other_list[1])
                if len(assign_list) > 0:
                    for father_group in assign_list:
                        self.sort_trans_father[father_group] = max_number
                self.trans_father_keep_dict.clear()


class generator_mini_group(object):
    def __init__(self, mini_group_label, father_mini):
        self.mini_group_label = mini_group_label
        self.correct_num = 0
        self.success = 0
        self.error = 0
        # self.error_allow = 0
        self.scanning_size = 4
        self.feature_num = 0
        self.small_group = {}
        self.small_group_num = {}
        self.small_replace_num = {}
        self.small_replace_max = {}
        self.small_start_num = {}
        self.small_total_num = 0
        self.small_activate_num = {}
        self.activate_small_group = {}
        self.small_group_base_num = {}
        self.father_mini = father_mini
        self.son_mini = []
        self.activate_group = []
        self.son_mini_correct = []
        self.save_update_group = []
        self.execute = 0
        self.delete = 0
        self.increase_ratio = 0

    def activate_report(self, correct, first_one):
        """This function is used to update the number of activations and arrange key parts according to the
        number of activations. We will save the key part frequently activated
        replace_key is used to update the program in future, it is currently not useful
        """
        if self.execute == 1:
            return
        self.execute = 1
        if correct >= 1:
            self.success = 1
            for mini_group in self.father_mini:
                mini_ob = initial_variable.mini_group_dict[mini_group]
                mini_ob.son_mini_correct.append([self.correct_num, self.mini_group_label])

        self.correct_num_writen(correct, first_one)
        delete_replace_list = []
        for start_group_key in self.activate_small_group:
            if self.small_start_num[start_group_key] == -1:
                continue
            [replace_key, group_key] = self.activate_small_group[start_group_key]
            if correct >= 1:
                if self.small_group_num[start_group_key][replace_key][group_key][0] != -1:
                    if self.small_group_num[start_group_key][replace_key][group_key][1] > 0:
                        self.small_group_num[start_group_key][replace_key][group_key][0] += 1
                self.small_start_num[start_group_key] += 1
                self.small_group_num[start_group_key][replace_key][group_key][1] += 1
                self.small_replace_num[start_group_key][replace_key][1] += 1
            else:
                if self.small_group_num[start_group_key][replace_key][group_key][0] != -1:
                    self.small_group_num[start_group_key][replace_key][group_key][0] -= 2
                self.small_start_num[start_group_key] -= 2
                self.small_group_num[start_group_key][replace_key][group_key][1] -= 2

            sort_list = []
            del_list = []
            assignment_value = []

            if len(self.small_group_num[start_group_key][replace_key]) > 1:
                sort_list.clear()
                small_group_max = 1
                for group_key in self.small_group_num[start_group_key][replace_key]:
                    son_correct_num = self.small_group_num[start_group_key][replace_key][group_key][0]
                    if son_correct_num > small_group_max:
                        small_group_max = son_correct_num
                    if son_correct_num == -1:
                        assignment_value.append(group_key)
                        continue
                    sort_list.append([son_correct_num, group_key])

                sort_list.sort(key=lambda x: x[0], reverse=True)
                small_group_num_copy = copy.deepcopy(self.small_group_num[start_group_key][replace_key])
                small_group_other = copy.deepcopy(self.small_group)
                self.small_group[start_group_key][replace_key].clear()
                self.small_group_num[start_group_key][replace_key].clear()
                for group_key_temp in assignment_value:
                    self.small_group[start_group_key][replace_key][group_key_temp] = \
                        small_group_other[start_group_key][replace_key][group_key_temp]
                    self.small_group_num[start_group_key][replace_key][group_key_temp] = [small_group_max, 2]

                execute_signal = 0
                if len(sort_list) > 1:
                    if small_group_max - sort_list[1][0] > 6:
                        execute_signal = 1
                    elif small_group_max - self.small_group_base_num[start_group_key][replace_key] > 20:
                        execute_signal = 2

                if execute_signal == 0:
                    for list_temp in sort_list:
                        self.small_group[start_group_key][replace_key][list_temp[1]] = \
                            small_group_other[start_group_key][replace_key][list_temp[1]]
                        self.small_group_num[start_group_key][replace_key][list_temp[1]] = small_group_num_copy[list_temp[1]]
                else:
                    del_list.append(sort_list[1][1])
                    self.small_group_base_num.pop(start_group_key)
                    if len(small_group_other[start_group_key][replace_key][sort_list[0][1]]) > 1:
                        self.small_group.pop(start_group_key)
                        self.small_replace_num.pop(start_group_key)
                        if start_group_key in self.small_replace_max:
                            self.small_replace_max.pop(start_group_key)
                        self.small_group_num.pop(start_group_key)
                        self.small_start_num.pop(start_group_key)
                        self.small_total_num -= self.small_activate_num[start_group_key]
                        self.small_activate_num.pop(start_group_key)

                        for group_other in small_group_other[start_group_key][replace_key][sort_list[0][1]]:
                            start_group_name = group_other
                            group_other_ob = start_group_dict[group_other]
                            while start_group_name in self.small_group:
                                start_group_num = int(start_group_name)
                                start_group_num += 1
                                start_group_name = str(start_group_num)
                            self.small_group[start_group_name] = {"0": {"0": [group_other]}}
                            self.small_group_num[start_group_name] = {"0": {"0": [1, 2]}}
                            self.small_replace_num[start_group_name] = {"0": [1, 2]}
                            self.small_replace_max[start_group_name] = 1
                            self.small_start_num[start_group_name] = -1
                            self.small_total_num += group_other_ob.feature_num
                            self.small_activate_num[start_group_name] = group_other_ob.feature_num
                            group_other_ob.father_mini_dict[self.mini_group_label] = {start_group_name: {"0": "0"}}
                        del_list.append(sort_list[0][1])
                    else:
                        self.small_group[start_group_key][replace_key] = \
                            {sort_list[0][1]: small_group_other[start_group_key][replace_key][sort_list[0][1]]}
                        self.small_group_num[start_group_key][replace_key] = \
                            {sort_list[0][1]: small_group_num_copy[sort_list[0][1]]}

                    if execute_signal > 1:
                        mini_group_label = create_mini_group(generator_mini_group, self.father_mini)
                        mini_group_ob = initial_variable.mini_group_dict[mini_group_label]
                        for start_group_other in small_group_other:
                            if start_group_other == start_group_key:
                                continue
                            mini_group_ob.small_group[start_group_other] = copy.deepcopy(self.small_group[start_group_other])
                            mini_group_ob.small_group_num[start_group_other] = copy.deepcopy(
                                self.small_group_num[start_group_other])
                            mini_group_ob.small_replace_num[start_group_other] = copy.deepcopy(
                                self.small_replace_num[start_group_other])
                            mini_group_ob.small_replace_max[start_group_other] = copy.deepcopy(
                                self.small_replace_max[start_group_other])
                            mini_group_ob.small_start_num[start_group_other] = self.small_start_num[start_group_other]
                            mini_group_ob.small_total_num += self.small_activate_num[start_group_other]
                            mini_group_ob.small_activate_num[start_group_other] = \
                                self.small_activate_num[start_group_other]
                            if start_group_other in self.small_group_base_num:
                                mini_group_ob.small_group_base_num[start_group_other] = \
                                    self.small_group_base_num[start_group_other]
                            for replace_key_temp in self.small_group[start_group_other]:
                                for group_key_other in self.small_group[start_group_other][replace_key_temp]:
                                    for group_other in self.small_group[start_group_other][replace_key_temp][group_key_other]:
                                        group_ob_other = start_group_dict[group_other]
                                        group_ob_other.father_mini_dict[mini_group_label] = {start_group_other: {replace_key_temp: group_key_other}}
                        for group_other in small_group_other[start_group_key][replace_key][sort_list[1][1]]:
                            start_group_name = group_other
                            group_other_ob = start_group_dict[group_other]
                            while start_group_name in mini_group_ob.small_group:
                                start_group_num = int(start_group_name)
                                start_group_num += 1
                                start_group_name = str(start_group_num)
                            mini_group_ob.small_group[start_group_name] = {"0": {"0": [group_other]}}
                            mini_group_ob.small_group_num[start_group_name] = {"0": {"0": [1, 2]}}
                            mini_group_ob.small_replace_num[start_group_name] = {"0": [1, 2]}
                            self.small_replace_max[start_group_name] = 1
                            # -1 represent it is not marked by number of activating
                            mini_group_ob.small_start_num[start_group_name] = -1
                            mini_group_ob.small_total_num += group_other_ob.feature_num
                            mini_group_ob.small_activate_num[start_group_name] = group_other_ob.feature_num
                            group_ob_other = start_group_dict[group_other]
                            group_ob_other.father_mini_dict[mini_group_label] = {start_group_name: {"0": "0"}}

                    if len(del_list) > 0:
                        for group_key in del_list:
                            for start_group in small_group_other[start_group_key][replace_key][group_key]:
                                delete_replace_list.append([start_group, start_group_key])
            else:
                if self.small_group_num[start_group_key][replace_key][group_key][0] < 1:
                    self.small_group_num[start_group_key][replace_key][group_key][0] = 1
                if self.small_replace_max[start_group_key] == \
                        self.small_replace_num[start_group_key][replace_key][0]:
                    if start_group_key not in self.small_group_base_num:
                        self.small_group_base_num[start_group_key] = \
                            {replace_key: self.small_group_num[start_group_key][replace_key][group_key][0]}
                    else:
                        self.small_group_base_num[start_group_key][replace_key] = \
                            self.small_group_num[start_group_key][replace_key][group_key][0]
                    for son_group_label in self.small_group[start_group_key][replace_key][group_key]:
                        son_group_ob = start_group_dict[son_group_label]
                        son_group_ob.refine_allow = 1
                        if self.mini_group_label not in son_group_ob.allow_mini_dict:
                            son_group_ob.allow_mini_dict[self.mini_group_label] = \
                                [1, start_group_key, replace_key]
                        else:
                            son_group_ob.allow_mini_dict[self.mini_group_label][0] += 1
        small_start_assignment = []
        max_small_start = 1
        for start_group_key in self.small_start_num:
            if self.small_start_num[start_group_key] == -1:
                small_start_assignment.append(start_group_key)
                continue
            if max_small_start < self.small_start_num[start_group_key]:
                max_small_start = self.small_start_num[start_group_key]
        for small_start_key in small_start_assignment:
            self.small_start_num[small_start_key] = max_small_start
        small_start_num_copy = copy.deepcopy(self.small_start_num)
        for start_group_key in small_start_num_copy:
            if start_group_key in self.activate_small_group and correct >= 1:
                if max_small_start - self.small_start_num[start_group_key] > 5:
                    self.small_start_num[start_group_key] = max_small_start - 5
            elif max_small_start - self.small_start_num[start_group_key] > 10:
                for replace_key_temp in self.small_group[start_group_key]:
                    for group_key in self.small_group[start_group_key][replace_key_temp]:
                        for start_group in self.small_group[start_group_key][replace_key_temp][group_key]:
                            delete_replace_list.append([start_group, start_group_key])
                if start_group_key in self.small_start_num:
                    self.small_start_num.pop(start_group_key)
                if start_group_key in self.small_group_base_num:
                    self.small_group_base_num.pop(start_group_key)
                if start_group_key in self.small_group:
                    self.small_group.pop(start_group_key)
                    self.small_group_num.pop(start_group_key)
                if start_group_key in self.small_activate_num:
                    self.small_total_num -= self.small_activate_num[start_group_key]
                    self.small_activate_num.pop(start_group_key)
        if len(delete_replace_list) > 0:
            for [start_group, start_group_key] in delete_replace_list:
                start_group_ob = start_group_dict[start_group]
                if start_group_key in start_group_ob.father_mini_dict[self.mini_group_label]:
                    start_group_ob.father_mini_dict[self.mini_group_label].pop(start_group_key)
                if len(start_group_ob.father_mini_dict[self.mini_group_label]) == 0:
                    start_group_ob.father_mini_dict.pop(self.mini_group_label)
                    if self.mini_group_label in start_group_ob.father_mini:
                        start_group_ob.father_mini.remove(self.mini_group_label)
                if len(start_group_ob.trans_father_total) == 0 and start_group_ob.delete == 0 \
                        and len(start_group_ob.father_mini_dict) == 0:
                    del_group(start_group)

    def correct_num_writen(self, correct, first_one):
        """This function is used to update the number of activations(Here is self.correct_num)
        When the label is correct and the recognition is successful, the program only rewards the son mini with the
        highest number of activations. However, if the label is wrong and the identification is successful,
        the program will punish all the successful son mini accordingly.
        """
        if self.success == 1:
            if correct >= 1:
                if first_one == 1:
                    if self.correct_num < initial_variable.son_mini_mid_correct_num:
                        self.correct_num += self.increase_ratio
                        if self.correct_num > initial_variable.son_mini_mid_correct_num:
                            self.correct_num = initial_variable.son_mini_mid_correct_num
                    else:
                        if self.increase_ratio > 0:
                            self.correct_num += correct

                self.increase_ratio += 1
            else:
                self.increase_ratio -= 2
                self.correct_num -= 2

    def parameter_initial(self, execute_initial=0):
        """This function is used to initialize related parameters to prepare for the next run and to arrange son mini
         according to the number of activations
         We set 20 son mini in total."""
        if self.delete == 1:
            return
        self.execute = 0
        self.success = 0
        self.delete = 0

        self.activate_group.clear()
        self.son_mini_correct.clear()
        self.activate_small_group.clear()

        if execute_initial == 0:
            if self.mini_group_label == "0" and (initial_variable.train_num % 5 == 0 or initial_variable.need_sort == 1):
                initial_variable.need_sort = 0
                sort_list = []
                for son_mini_group in self.son_mini:
                    son_mini_ob = initial_variable.mini_group_dict[son_mini_group]
                    sort_list.append([son_mini_ob.correct_num, son_mini_group])
                sort_list.sort(key=lambda x: x[0], reverse=True)
                if len(sort_list) > 0 and sort_list[0][0] > 0:
                    initial_variable.son_mini_max_correct_num = sort_list[0][0]

                self.son_mini.clear()
                save_length = 0
                del_list = []
                for son_mini_list in sort_list:
                    if save_length == 5:
                        initial_variable.son_mini_mid_correct_num = son_mini_list[0]
                    if save_length < 20:
                        # Only keep the top 20 son minis with activations
                        save_length += 1
                        self.son_mini.append(son_mini_list[1])
                    else:
                        del_list.append(son_mini_list[1])
                for son_mini in del_list:
                    del_mini(son_mini)


def create_group(father_mini, image_test_list):
    """Create new a son_mini and a new group
    """
    mini_group_label = create_mini_group(generator_mini_group, father_mini)
    for [image_test_copy, feature_num, save_end_dict, scanning_image_dict, scanning_image_or_dict] in image_test_list:
        [yi, xi] = save_end_dict[0]
        image_test_copy[yi][xi]["next_end"] = 1
        [yi2, xi2] = save_end_dict[1]
        image_test_copy[yi2][xi2]["previous_end"] = 1
        group_label, length_dict = create_group_label(generator_entity, image_test_copy,
                                                      feature_num, [mini_group_label],
                                                      [], {}, scanning_image_dict, scanning_image_or_dict)
