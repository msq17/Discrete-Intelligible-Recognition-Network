# *** Writen by Shiquan Mei ***


import numpy as np
import struct
import matplotlib.pyplot as plt
from part import *
from initial_variable import *
# from DcomNet_model_list_update import *
import copy


def recognition(image_test_list, image_test, image_name_dict, num, first_mini, image_label, label):
    """This function is used to train our program
    """
    for name in start_group_dict:
        start_group_dict[name].parameter_initial()
    for mini_name in initial_variable.mini_group_dict:
        initial_variable.mini_group_dict[mini_name].parameter_initial()

    first_mini_ob = initial_variable.mini_group_dict[first_mini[0]]
    son_mini_total = first_mini_ob.son_mini.copy()
    success_total = 0
    success_verify = 0
    first_one = 1
    activate_total_number = 0
    if image_label == label:
        precedent_total_number = 20
    else:
        precedent_total_number = 8
    for son_mini in son_mini_total:
        if activate_total_number >= precedent_total_number:
            break
        activate_total_number += 1
        image_name_dict_copy = copy.deepcopy(image_name_dict)
        image_test_dict = copy.deepcopy(image_test)
        total_save_dict = {}

        son_mini_ob = initial_variable.mini_group_dict[son_mini]
        success_num = 0
        small_total_num = 0
        for start_group_key in son_mini_ob.small_group:
            feature_total_list = []
            for replace_key in son_mini_ob.small_group[start_group_key]:
                for group_key in son_mini_ob.small_group[start_group_key][replace_key]:
                    feature_total = 0
                    feature_activate = 0
                    for start_group in son_mini_ob.small_group[start_group_key][replace_key][group_key]:
                        start_group_ob = start_group_dict[start_group]
                        if son_mini not in start_group_ob.activate_father_mini:
                            start_group_ob.activate_father_mini[son_mini] = 0
                        else:
                            break
                        result_list = start_group_ob.compare_recognition(image_test_dict, total_save_dict, ())
                        feature_total += start_group_ob.feature_num
                        if len(result_list) > 0:
                            feature_activate += result_list[0]
                    feature_total_list.append([feature_activate / feature_total, feature_activate, feature_total,
                                               group_key, replace_key])
            if len(feature_total_list) > 0:
                feature_total_list.sort(key=lambda x: x[0], reverse=True)
                success_num += feature_total_list[0][1]
                small_total_num += feature_total_list[0][2]
                group_key = feature_total_list[0][3]
                replace_key = feature_total_list[0][4]
                son_mini_ob.activate_small_group[start_group_key] = [replace_key, group_key]
        if small_total_num > 0 and success_num / small_total_num > 0.75:
            precedent_total_number = 8
            for start_group_key in son_mini_ob.activate_small_group:
                [replace_key, group_key] = son_mini_ob.activate_small_group[start_group_key]
                for start_group in son_mini_ob.small_group[start_group_key][replace_key][group_key]:
                    start_group_ob = start_group_dict[start_group]
                    start_group_ob.success = 1
                    total_save_dict.update(start_group_ob.total_save_dict)
                    start_group_ob.recognition_path = [start_group]
            for start_group_key in son_mini_ob.activate_small_group:
                [replace_key, group_key] = son_mini_ob.activate_small_group[start_group_key]
                for start_group in son_mini_ob.small_group[start_group_key][replace_key][group_key]:
                    start_group_ob = start_group_dict[start_group]
                    iteration_recognition(start_group_ob, start_group_ob.ti, start_group_ob.tj,
                                          start_group, image_test_dict, total_save_dict, son_mini)
                    start_group_ob.match_success()
                    for group_label in start_group_ob.recognition_path:
                        group_temp_ob = start_group_dict[group_label]
                        group_temp_ob.merge_common(image_name_dict_copy)
            other_save_list_total, total_sum = verify_recognition(image_test_dict, image_name_dict_copy)
            if image_label == label:
                if 0.45 > total_sum / num > 0.05:
                    for other_save_list in other_save_list_total:
                        length_dict = {}
                        create_group_f(generator_entity, [], other_save_list,
                                       image_test_dict,
                                       image_name_dict_copy, length_dict)
                if total_sum / num < 0.45:
                    success_total = 1
                    for start_group_key in son_mini_ob.activate_small_group:
                        [replace_key, group_key] = son_mini_ob.activate_small_group[start_group_key]
                        if start_group_key in son_mini_ob.small_group and \
                                replace_key in son_mini_ob.small_group[start_group_key] and \
                                group_key in son_mini_ob.small_group[start_group_key][replace_key]:
                            for start_group in son_mini_ob.small_group[start_group_key][replace_key][group_key]:
                                start_group_ob = start_group_dict[start_group]
                                for group_label1 in start_group_ob.recognition_path:
                                    group_ob1 = start_group_dict[group_label1]
                                    group_ob1.correct_num_writen(1)
                                    group_ob1.update_memory(image_test_dict)
                                    group_ob1.refine_self_mini_group()
                    if total_sum / num < 0.15:
                        success_verify = 1
                        son_mini_ob.activate_report(2, first_one)
                    else:
                        son_mini_ob.activate_report(1, first_one)
                    first_one = 0
            else:
                success_verify = 1
                son_mini_ob.activate_report(0, 1)
                for start_group_key in son_mini_ob.activate_small_group:
                    [replace_key, group_key] = son_mini_ob.activate_small_group[start_group_key]
                    if start_group_key in son_mini_ob.small_group and \
                            replace_key in son_mini_ob.small_group[start_group_key] and \
                            group_key in son_mini_ob.small_group[start_group_key][replace_key]:
                        for start_group in son_mini_ob.small_group[start_group_key][replace_key][group_key]:
                            start_group_ob = start_group_dict[start_group]
                            for group_label1 in start_group_ob.recognition_path:
                                group_label_ob = start_group_dict[group_label1]
                                group_label_ob.correct_num_writen(0)

    if success_total == 1 and image_label == label:
        initial_variable.train_num += 1
    elif success_total == 0 and image_label == label:
        create_group(first_mini, image_test_list)

    return success_verify


def iteration_recognition(group_temp_ob, ti, tj, son_group_label, image_test_dict, trans_save_dict, son_mini):
    length_list = [[son_group_label, trans_save_dict]]
    length_list_temp = []
    activate_son_temp = {}
    layer = 1
    while len(length_list) > 0:
        for [group_label_temp, trans_save_dict] in length_list:
            if group_label_temp not in group_temp_ob.recognition_path:
                group_temp_ob.recognition_path.append(group_label_temp)
            group_son_temp = start_group_dict[group_label_temp]
            activate_son_temp.clear()
            sort_list = []
            if len(group_son_temp.trans_son_total) > 0:
                for son_group_key in group_son_temp.trans_son_total:
                    save_dict_copy = copy.deepcopy(trans_save_dict)
                    success_num = 0
                    small_total_num = 0
                    for start_group_key in group_son_temp.trans_son_total[son_group_key]:
                        feature_total_list = []
                        for replace_key in group_son_temp.trans_son_total[son_group_key][start_group_key]:
                            feature_total = 0
                            feature_activate = 0
                            for group_key in group_son_temp.trans_son_total[son_group_key][start_group_key][replace_key]:
                                for start_group_label in group_son_temp.trans_son_total[son_group_key][start_group_key][replace_key][group_key]:
                                    start_group_ob = start_group_dict[start_group_label]
                                    if son_mini not in start_group_ob.activate_father_mini:
                                        start_group_ob.activate_father_mini[son_mini] = layer
                                    elif layer != start_group_ob.activate_father_mini[son_mini]:
                                        break
                                    feature_total += start_group_ob.feature_num
                                    result_list = start_group_ob.compare_recognition(image_test_dict, save_dict_copy,
                                                                                     (ti, tj), 1)
                                    if len(result_list) > 0:
                                        feature_activate += start_group_ob.feature_num
                                if feature_total > 0:
                                    feature_total_list.append([feature_activate / feature_total, feature_activate, feature_total,
                                                               group_key, replace_key])
                        if len(feature_total_list) > 0:
                            feature_total_list.sort(key=lambda x: x[0], reverse=True)
                            success_num += feature_total_list[0][1]
                            small_total_num += feature_total_list[0][2]
                            group_key = feature_total_list[0][3]
                            replace_key = feature_total_list[0][4]
                            if son_group_key not in activate_son_temp:
                                activate_son_temp[son_group_key] = {start_group_key: [replace_key, group_key]}
                            else:
                                activate_son_temp[son_group_key][start_group_key] = [replace_key, group_key]
                    if small_total_num > 0 and success_num / small_total_num > 0.75:
                        for start_group_key in activate_son_temp[son_group_key]:
                            [replace_key, group_key] = activate_son_temp[son_group_key][start_group_key]
                            for start_group_label in \
                                    group_son_temp.trans_son_total[son_group_key][start_group_key][replace_key][
                                        group_key]:
                                father_group_ob = start_group_dict[start_group_label]
                                father_group_ob.success = 1
                                save_dict_copy.update(father_group_ob.total_save_dict)
                        sort_list.append([len(save_dict_copy), son_group_key, save_dict_copy])
                    else:
                        if son_group_key in activate_son_temp:
                            activate_son_temp.pop(son_group_key)

                if len(activate_son_temp) > 0:
                    sort_list.sort(key=lambda x: x[0], reverse=True)
                    group_son_temp.activate_son[sort_list[0][1]] = copy.deepcopy(activate_son_temp[sort_list[0][1]])
                    for start_group_key in activate_son_temp[sort_list[0][1]]:
                        [replace_key, group_key] = activate_son_temp[sort_list[0][1]][start_group_key]
                        for start_group_label in \
                                group_son_temp.trans_son_total[sort_list[0][1]][start_group_key][replace_key][group_key]:
                            father_group_ob = start_group_dict[start_group_label]
                            if father_group_ob.match_success_signal == 1:
                                continue
                            if father_group_ob.success == 1:
                                length_list_temp.append([start_group_label, sort_list[0][2]])
                            father_group_ob.match_success()
        length_list.clear()
        if len(length_list_temp) > 0:
            length_list += length_list_temp
            layer += 1
            length_list_temp.clear()
