# *** Writen by Shiquan Mei ***

import numpy as np
import matplotlib.pyplot as plt
from initial_variable import *
import copy
import initial_variable


def scanning_input(image_test):
    """This function is used to extract boundaries from the input image"""
    image_test_dict = {}
    scanning_image_dict = {}
    scanning_image_or_dict = {}
    image_test_list = []
    image_test_total = {}
    image_name_dict = {}
    image_name_dict_temp = {}
    num = 0
    test_num = 0
    start_signal = 0
    scanning_dict = {}
    previous_next = [(-1, -1), (-1, -1)]
    y_length, x_length = image_test.shape
    for yi in range(1, y_length - 1):
        for xi in range(1, x_length - 1):
            if image_test[yi][xi] == 0:
                key = 0
                if image_test[yi - 1][xi] == 1:
                    key = 1
                if image_test[yi + 1][xi] == 1:
                    key = 2
                if image_test[yi][xi - 1] == 1:
                    key = 3
                if image_test[yi][xi + 1] == 1:
                    key = 4

                if (yi, xi, key) in scanning_dict:
                    continue

                if key > 0:
                    serve_list = [[yi, xi, key, 1]]
                    num_temp = 0
                    while len(serve_list) > 0:
                        list_temp = serve_list.pop(0)
                        yii = list_temp[0]
                        xii = list_temp[1]
                        key = list_temp[2]
                        direction = list_temp[3]
                        if (yii, xii, key) in scanning_dict:
                            continue
                        scanning_dict[(yii, xii, key)] = 1
                        son_list = []
                        if key == 1:
                            if direction == 1:
                                if xii + 1 < x_length:
                                    if image_test[yii][xii + 1] == 0 and image_test[yii - 1][xii + 1] == 1:
                                        serve_list.append([yii, xii + 1, key, direction])
                                        son_list = [(yii, xii, key), (yii, xii + 1, key)]
                                    elif image_test[yii][xii + 1] == 1:
                                        serve_list.append([yii, xii, 4, 1])
                                        son_list = [(yii, xii, 4)]
                                    elif image_test[yii][xii + 1] == 0 and image_test[yii - 1][xii + 1] == 0:
                                        serve_list.append([yii - 1, xii + 1, 3, 0])
                                        son_list = [(yii, xii, key), (yii - 1, xii + 1, 3)]
                            else:
                                if xii - 1 >= 0:
                                    if image_test[yii][xii - 1] == 0 and image_test[yii - 1][xii - 1] == 1:
                                        serve_list.append([yii, xii - 1, key, direction])
                                        son_list = [(yii, xii, key), (yii, xii - 1, key)]
                                    elif image_test[yii][xii - 1] == 1:
                                        serve_list.append([yii, xii, 3, 1])
                                        son_list = [(yii, xii, 3)]
                                    elif image_test[yii][xii - 1] == 0 and image_test[yii - 1][xii - 1] == 0:
                                        serve_list.append([yii - 1, xii - 1, 4, 0])
                                        son_list = [(yii, xii, key), (yii - 1, xii - 1, 4)]
                        elif key == 2:
                            if direction == 1:
                                if xii + 1 < x_length:
                                    if image_test[yii][xii + 1] == 0 and image_test[yii + 1][xii + 1] == 1:
                                        serve_list.append([yii, xii + 1, key, direction])
                                        son_list = [(yii, xii, key), (yii, xii + 1, key)]
                                    elif image_test[yii][xii + 1] == 1:
                                        serve_list.append([yii, xii, 4, 0])
                                        son_list = [(yii, xii, 4)]
                                    elif image_test[yii][xii + 1] == 0 and image_test[yii + 1][xii + 1] == 0:
                                        serve_list.append([yii + 1, xii + 1, 3, 1])
                                        son_list = [(yii, xii, key), (yii + 1, xii + 1, 3)]
                            else:
                                if xii - 1 >= 0:
                                    if image_test[yii][xii - 1] == 0 and image_test[yii + 1][xii - 1] == 1:
                                        serve_list.append([yii, xii - 1, key, direction])
                                        son_list = [(yii, xii, key), (yii, xii - 1, key)]
                                    elif image_test[yii][xii - 1] == 1:
                                        serve_list.append([yii, xii, 3, 0])
                                        son_list = [(yii, xii, 3)]
                                    elif image_test[yii][xii - 1] == 0 and image_test[yii + 1][xii - 1] == 0:
                                        serve_list.append([yii + 1, xii - 1, 4, 1])
                                        son_list = [(yii, xii, key), (yii + 1, xii - 1, 4)]
                        elif key == 3:
                            if direction == 1:
                                if yii + 1 < y_length:
                                    if image_test[yii + 1][xii] == 0 and image_test[yii + 1][xii - 1] == 1:
                                        serve_list.append([yii + 1, xii, key, direction])
                                        son_list = [(yii, xii, key), (yii + 1, xii, key)]
                                    elif image_test[yii + 1][xii] == 1:
                                        serve_list.append([yii, xii, 2, 1])
                                        son_list = [(yii, xii, 2)]
                                    elif image_test[yii + 1][xii] == 0 and image_test[yii + 1][xii - 1] == 0:
                                        serve_list.append([yii + 1, xii - 1, 1, 0])
                                        son_list = [(yii, xii, key), (yii + 1, xii - 1, 1)]
                            else:
                                if yii - 1 >= 0:
                                    if image_test[yii - 1][xii] == 0 and image_test[yii - 1][xii - 1] == 1:
                                        serve_list.append([yii - 1, xii, key, direction])
                                        son_list = [(yii, xii, key), (yii - 1, xii, key)]
                                    elif image_test[yii - 1][xii] == 1:
                                        serve_list.append([yii, xii, 1, 1])
                                        son_list = [(yii, xii, 1)]
                                    elif image_test[yii - 1][xii] == 0 and image_test[yii - 1][xii - 1] == 0:
                                        serve_list.append([yii - 1, xii - 1, 2, 0])
                                        son_list = [(yii, xii, key), (yii - 1, xii - 1, 2)]
                        elif key == 4:
                            if direction == 1:
                                if yii + 1 < y_length:
                                    if image_test[yii + 1][xii] == 0 and image_test[yii + 1][xii + 1] == 1:
                                        serve_list.append([yii + 1, xii, key, direction])
                                        son_list = [(yii, xii, key), (yii + 1, xii, key)]
                                    elif image_test[yii + 1][xii] == 1:
                                        serve_list.append([yii, xii, 2, 0])
                                        son_list = [(yii, xii, 2)]
                                    elif image_test[yii + 1][xii] == 0 and image_test[yii + 1][xii + 1] == 0:
                                        serve_list.append([yii + 1, xii + 1, 1, 1])
                                        son_list = [(yii, xii, key), (yii + 1, xii + 1, 1)]
                            else:
                                if yii - 1 >= 0:
                                    if image_test[yii - 1][xii] == 0 and image_test[yii - 1][xii + 1] == 1:
                                        serve_list.append([yii - 1, xii, key, direction])
                                        son_list = [(yii, xii, key), (yii - 1, xii, key)]
                                    elif image_test[yii - 1][xii] == 1:
                                        serve_list.append([yii, xii, 1, 0])
                                        son_list = [(yii, xii, 1)]
                                    elif image_test[yii - 1][xii] == 0 and image_test[yii - 1][xii + 1] == 0:
                                        serve_list.append([yii - 1, xii + 1, 2, 1])
                                        son_list = [(yii, xii, key), (yii - 1, xii + 1, 2)]
                        length_s = len(son_list)
                        for ii in range(length_s):
                            (ym, xm, key_m) = son_list[ii]
                            if ym in image_test_dict and xm in image_test_dict[ym]:
                                if key_m not in image_test_dict[ym][xm]["direction"]:
                                    image_test_dict[ym][xm]["direction"][key_m] = 1
                                if length_s == 2:
                                    if ii == 0:
                                        image_test_dict[ym][xm]["next"][(son_list[1][0], son_list[1][1])] = 1
                                    else:
                                        image_test_dict[ym][xm]["previous"][(son_list[0][0], son_list[0][1])] = 1
                                        (yl, xl, key_l) = son_list[0]
                                        previous_next[0] = (yl, xl)

                            elif ym in image_test_dict and xm not in image_test_dict[ym]:
                                num += 1  # feature_num
                                num_temp += 1

                                dict1 = {key_m: 1}
                                image_test_dict[ym][xm] = {"direction": dict1,
                                                           "previous": {},
                                                           "next": {},
                                                           "encode": [0, 1000],
                                                           "mini_encode": [num_temp],
                                                           "encode_recode": {0: (ym, xm),
                                                                             1: 0},
                                                           "mini_encode_temp": num_temp,
                                                           "previous_end": 0,
                                                           "next_end": 0,
                                                           "activate_num": 0,
                                                           "increase_ratio": 2}

                                image_name_dict_temp[ym][xm] = []
                                if num_temp not in scanning_image_dict:
                                    scanning_image_dict[num_temp] = {(ym, xm): 1}
                                    scanning_image_or_dict[num_temp] = (ym, xm)
                                elif (ym, xm) not in scanning_image_dict[num_temp]:
                                    scanning_image_dict[num_temp][(ym, xm)] = 1
                                    scanning_image_or_dict[num_temp] = (ym, xm)
                                # number += 1
                                if start_signal == 0:
                                    previous_next[1] = (ym, xm)
                                    start_signal = 1
                                elif length_s == 1:
                                    previous_next[0] = (ym, xm)
                                if length_s == 2:
                                    if ii == 0:
                                        image_test_dict[ym][xm]["next"][(son_list[1][0], son_list[1][1])] = 1
                                    else:
                                        image_test_dict[ym][xm]["previous"][(son_list[0][0], son_list[0][1])] = 1
                            else:
                                num += 1
                                num_temp += 1

                                dict1 = {key_m: 1}
                                image_test_dict[ym] = {xm: {"direction": dict1,
                                                            "previous": {},
                                                            "next": {},
                                                            "encode": [0, 1000],
                                                            "mini_encode": [num_temp],
                                                            "encode_recode": {0: (ym, xm),
                                                                              1: 0},
                                                            "mini_encode_temp": num_temp,
                                                            "previous_end": 0,
                                                            "next_end": 0,
                                                            "activate_num": 0,
                                                            "increase_ratio": 2}}

                                image_name_dict_temp[ym] = {xm: []}
                                if num_temp not in scanning_image_dict:
                                    scanning_image_dict[num_temp] = {(ym, xm): 1}
                                    scanning_image_or_dict[num_temp] = (ym, xm)
                                elif (ym, xm) not in scanning_image_dict[num_temp]:
                                    scanning_image_dict[num_temp][(ym, xm)] = 1
                                    scanning_image_or_dict[num_temp] = (ym, xm)
                                if start_signal == 0:
                                    previous_next[1] = (ym, xm)
                                    start_signal = 1
                                elif length_s == 1:
                                    previous_next[0] = (ym, xm)

                                if length_s == 2:
                                    if ii == 0:
                                        image_test_dict[ym][xm]["next"][(son_list[1][0], son_list[1][1])] = 1
                                    else:
                                        image_test_dict[ym][xm]["previous"][(son_list[0][0], son_list[0][1])] = 1
                    if num_temp > 7 and previous_next[0][0] != -1 and previous_next[1][0] != -1:
                        test_num += 1
                        start_signal = 0
                        image_test_copy = copy.deepcopy(image_test_dict)
                        save_end_dict = {}
                        scanning_image_dict_copy = copy.deepcopy(scanning_image_dict)
                        scanning_image_or_dict_copy = copy.deepcopy(scanning_image_or_dict)
                        for i in range(2):
                            (yn, xn) = previous_next[i]
                            if i == 0:
                                image_test_copy[yn][xn]["next_end"] = 1
                                save_end_dict[0] = [yn, xn]
                                print("next_end is: " + str(save_end_dict[0]))
                            else:
                                image_test_copy[yn][xn]["previous_end"] = 1
                                save_end_dict[1] = [yn, xn]
                                print("previous_end is: " + str(save_end_dict[1]))
                        image_test_list.append([image_test_copy, num_temp,
                                                save_end_dict, scanning_image_dict_copy,
                                                scanning_image_or_dict_copy])
                        if len(image_test_total) == 0:
                            image_test_total.update(image_test_dict)
                            image_name_dict.update(image_name_dict_temp)
                        else:
                            for yii in image_test_dict:
                                if yii in image_test_total:
                                    image_test_total[yii].update(image_test_dict[yii])
                                    image_name_dict[yii].update(image_name_dict_temp[yii])
                                else:
                                    image_test_total[yii] = copy.deepcopy(image_test_dict[yii])
                                    image_name_dict[yii] = copy.deepcopy(image_name_dict_temp[yii])

                    image_test_dict.clear()
                    image_name_dict_temp.clear()
                    previous_next = [(-1, -1), (-1, -1)]
                    scanning_image_dict.clear()
                    scanning_image_or_dict.clear()
    return image_test_list, image_test_total, image_name_dict, num, test_num


def verify_recognition(image_test_dict, image_name_dict):
    """"This function is used to calculate how many input boundary points have not been matched"""
    save_scanning = {}
    other_save_list = []
    other_save_list_total = []
    total_sum = 0
    for yi in image_name_dict:
        for xi in image_name_dict[yi]:
            if len(image_name_dict[yi][xi]) == 0:
                if (yi, xi) in save_scanning:
                    continue
                else:
                    other_save_list.clear()
                    sum_num = 0
                    for key in range(2):
                        serve_list = [[yi, xi]]
                        if (yi, xi) in save_scanning:
                            save_scanning.pop((yi, xi))
                            sum_num -= 1
                        while len(serve_list) > 0:
                            temp_list = serve_list.pop(0)
                            yii = temp_list[0]
                            xii = temp_list[1]
                            if yii in image_name_dict and xii in image_name_dict[yii] \
                                    and len(image_name_dict[yii][xii]) == 0:
                                other_save_list.append([yii, xii])
                                sum_num += 1
                                save_scanning[(yii, xii)] = 1
                                dict_or = image_test_dict[yii][xii]
                                if key == 1:
                                    for tuple_temp in dict_or["next"]:
                                        if tuple_temp in save_scanning:
                                            continue
                                        else:
                                            serve_list.append([tuple_temp[0], tuple_temp[1]])
                                else:
                                    for tuple_temp in dict_or["previous"]:
                                        if tuple_temp in save_scanning:
                                            continue
                                        else:
                                            serve_list.append([tuple_temp[0], tuple_temp[1]])
                    if sum_num >= 5:
                        total_sum += sum_num
                        other_save_list_total.append(copy.deepcopy(other_save_list))
    return other_save_list_total, total_sum


def create_group_f(generator_entity, father_mini, other_save_list, image_test_dict,
                   image_name_dict, length_dict):
    """This function is used to write the points of input boundaries without matching to memory"""
    save_scanning = {}
    image_save_dict = {}
    scanning_image_dict = {}
    scanning_image_or_dict = {}
    feature_num = 0
    save_end_dict = {}
    parent_list = []
    if len(other_save_list) > 0:
        for [yi, xi] in other_save_list:
            if (yi, xi) in save_scanning or len(image_name_dict[yi][xi]) > 0:
                continue
            else:
                for key in range(2):
                    serve_list = [[yi, xi]]
                    if (yi, xi) in save_scanning:
                        save_scanning.pop((yi, xi))
                        feature_num -= 1

                    while len(serve_list) > 0:
                        list1 = serve_list.pop(0)
                        yii = list1[0]
                        xii = list1[1]

                        if (yii, xii) in save_scanning:
                            if yii in image_name_dict and xii in image_name_dict[yii]:
                                if key == 0:
                                    save_end_dict[0] = (yii, xii)
                                elif key == 1:
                                    save_end_dict[1] = (yii, xii)
                            continue
                        elif yii in image_name_dict and xii in image_name_dict[yii]:
                            dict_temp = image_test_dict[yii][xii]
                            save_scanning[(yii, xii)] = 1
                            feature_num += 1

                            if len(image_name_dict[yii][xii]) == 0:
                                if key == 0:
                                    for (yi_m, xi_m) in dict_temp["next"]:
                                        if [yi_m, xi_m] not in serve_list:
                                            if yi_m in image_test_dict and xi_m in image_test_dict[yi_m]:
                                                if (yii, xii) not in image_test_dict[yi_m][xi_m]["previous"]:
                                                    image_test_dict[yi_m][xi_m]["previous"][(yii, xii)] = 1
                                                serve_list.append([yi_m, xi_m])
                                    if len(dict_temp["next"]) == 0:
                                        save_end_dict[0] = (yii, xii)
                                else:
                                    for (yi_m, xi_m) in dict_temp["previous"]:
                                        if [yi_m, xi_m] not in serve_list:
                                            if yi_m in image_test_dict and xi_m in image_test_dict[yi_m]:
                                                if (yii, xii) not in image_test_dict[yi_m][xi_m]["next"]:
                                                    image_test_dict[yi_m][xi_m]["next"][(yii, xii)] = 1
                                                serve_list.append([yi_m, xi_m])
                                            serve_list.append([yi_m, xi_m])
                                    if len(dict_temp["previous"]) == 0:
                                        save_end_dict[1] = (yii, xii)
                            else:
                                for parent_group in image_name_dict[yii][xii]:
                                    if parent_group not in parent_list:
                                        parent_list.append(parent_group)
                                if key == 0:
                                    save_end_dict[0] = (yii, xii)
                                elif key == 1:
                                    save_end_dict[1] = (yii, xii)
                if feature_num >= 5 and len(parent_list) > 0:
                    (yi1, xi1) = save_end_dict[1]

                    serve_list = [[yi1, xi1, 0]]
                    serve_list_temp = []
                    repeat_list = {}
                    total_encode_number = 0
                    while len(serve_list) > 0:
                        for list1 in serve_list:
                            yii = list1[0]
                            xii = list1[1]
                            encode_number = list1[2]
                            if yii in image_test_dict and xii in image_test_dict[yii]:
                                dict_temp_or = image_test_dict[yii][xii]
                                dict_temp = copy.deepcopy(dict_temp_or)
                                if yii not in image_save_dict:
                                    image_save_dict[yii] = {xii: dict_temp}
                                elif xii not in image_save_dict[yii]:
                                    image_save_dict[yii][xii] = dict_temp

                                dict_temp["mini_encode"][0] = encode_number
                                dict_temp["encode_recode"][0] = (yii, xii)
                                dict_temp["encode_recode"][1] = 0
                                if encode_number not in scanning_image_dict:
                                    scanning_image_dict[encode_number] = {(yii, xii): 1}
                                    scanning_image_or_dict[encode_number] = (yii, xii)
                                elif (yii, xii) not in scanning_image_dict[encode_number]:
                                    scanning_image_dict[encode_number][(yii, xii)] = 1
                                    scanning_image_or_dict[encode_number] = (yii, xii)
                                total_encode_number = encode_number
                                for (yi_m, xi_m) in dict_temp["next"]:
                                    if (yi_m, xi_m) not in repeat_list and (yi_m, xi_m) in save_scanning:
                                        repeat_list[(yi_m, xi_m)] = 1
                                        serve_list_temp.append([yi_m, xi_m, encode_number + 1])
                        serve_list.clear()
                        if len(serve_list_temp) > 0:
                            serve_list += serve_list_temp
                            serve_list_temp.clear()
                    scanning_image_dict_copy = copy.deepcopy(scanning_image_dict)
                    scanning_image_or_dict_copy = copy.deepcopy(scanning_image_or_dict)
                    group_label, _ = create_group_label(generator_entity,
                                                        copy.deepcopy(image_save_dict),
                                                        total_encode_number, father_mini,
                                                        parent_list.copy(), length_dict,
                                                        scanning_image_dict_copy,
                                                        scanning_image_or_dict_copy)

                    for yi_n in image_save_dict:
                        for xi_n in image_save_dict[yi_n]:
                            image_name_dict[yi_n][xi_n].append(group_label)
                image_save_dict.clear()
                parent_list.clear()
                feature_num = 0
                scanning_image_dict.clear()
                scanning_image_or_dict.clear()


def recognition_input(scanning_num, image_save_dict, image_test_dict, save_dict_trans, life_num,
                      length_num, scanning_image_dict,
                      recognition_ratio=0.75, repeat_recognize_ratio=0.45):
    """This function is used to scan in a small area to find the best position with the input boundary point."""
    list_save = []
    position_num = scanning_num // 2
    for ti in range(-position_num, position_num + 1):
        for tj in range(-position_num, position_num + 1):
            return_list = son_recognition_input([ti, tj], image_save_dict, image_test_dict, save_dict_trans, life_num,
                                                length_num, scanning_image_dict,
                                                recognition_ratio, repeat_recognize_ratio)
            if len(return_list) > 0:
                return_list.append([ti, tj])
                list_save.append(return_list)
    if len(list_save) > 0:
        list_save.sort(key=lambda x: x[0], reverse=True)
        result_list = list_save.pop(0)
        return result_list
    return []


def son_recognition_input(ij_list, image_save_dict, image_test_dict, save_dict_trans, life_num,
                          length_num, scanning_image_dict,
                          recognition_ratio=0.75, repeat_recognize_ratio=0.45):
    """This function is used to calculate the number of matches between the input boundary points and
    the points in memory in this position.
    """
    ti = ij_list[0]
    tj = ij_list[1]
    save_scanning = {}
    save_scanning_copy = {}
    image_save_scanning = {}
    image_save_scanning_temp = {}
    save_dict = {}
    activate_dict = {}
    activate_dict_temp = {}
    total_save_dict_simple = {}
    encode = [0, 0]
    record_slice = {}
    mark_num = 0
    repeat_recognise_num = 0
    repeat_recognise_num_temp = 0
    scanning_point = 0
    fail_num = 0

    boundary_dict = {}
    total_boundary_dict = {}
    while scanning_point < length_num:
        previous_scanning_point = scanning_point
        if scanning_point in scanning_image_dict:
            for (yi, xi) in scanning_image_dict[scanning_point]:
                if yi in image_save_dict and xi in image_save_dict[yi]:
                    serve_list_1 = []
                    match_num = 0
                    if (yi, xi) in image_save_scanning:
                        continue
                    else:
                        dict1 = image_save_dict[yi][xi]["direction"]
                        for key in dict1:
                            if key == 1 or key == 2:
                                yr = 1
                                xr = 0
                            else:
                                yr = 0
                                xr = 1

                            if yi + ti in image_test_dict and xi + tj in image_test_dict[yi + ti]:
                                serve_list_1 = [yi + ti, xi + tj, 0]
                                # this variable is used to count number of comparison, addition and multiplication
                                if initial_variable.test_signal == 1:
                                    initial_variable.comparator_num += 1
                            elif yi + ti - yr in image_test_dict and xi + tj - xr in image_test_dict[yi + ti - yr]:
                                serve_list_1 = [yi + ti - yr, xi + tj - xr, 0]
                                if initial_variable.test_signal == 1:
                                    initial_variable.comparator_num += 1
                            elif yi + ti + yr in image_test_dict and xi + tj + xr in image_test_dict[yi + ti + yr]:
                                serve_list_1 = [yi + ti + yr, xi + tj + xr, 0]
                                if initial_variable.test_signal == 1:
                                    initial_variable.comparator_num += 1

                            if len(serve_list_1) > 0:
                                break
                    if len(serve_list_1) > 0:
                        first_time = 0
                        save_scanning_copy = copy.deepcopy(save_scanning)
                        dict_temp_encode = image_save_dict[yi][xi]["mini_encode"][0]
                        if dict_temp_encode > 200:
                            continue
                        serve_list_temp = []
                        for mark in range(2):
                            serve_list = [[serve_list_1[0], serve_list_1[1], 0, mark, dict_temp_encode]]
                            if mark == 1 and (serve_list_1[0], serve_list_1[1]) in save_scanning_copy:
                                save_scanning_copy.pop((serve_list_1[0], serve_list_1[1]))
                                if (serve_list_1[0], serve_list_1[1]) in save_dict_trans:
                                    repeat_recognise_num_temp -= 1
                            if mark == 1 and (yi, xi) in image_save_scanning_temp:
                                image_save_scanning_temp.pop((yi, xi))
                            while len(serve_list) > 0:
                                for list1 in serve_list:
                                    yii = list1[0]
                                    xii = list1[1]
                                    life = list1[2]
                                    next_signal = list1[3]
                                    now_encode_0 = list1[4]

                                    if (yii, xii) in save_scanning_copy or life >= life_num:
                                        continue
                                    elif yii in image_test_dict and xii in image_test_dict[yii]:
                                        if (yii, xii) in save_dict_trans:
                                            repeat_recognise_num_temp += 1
                                        save_scanning_copy[(yii, xii)] = 1
                                        dict_or = image_test_dict[yii][xii]
                                        life += 1
                                        if initial_variable.test_signal == 1:
                                            initial_variable.adder_num += 1
                                        for key in dict_or["direction"]:
                                            if key == 1 or key == 2:
                                                yr = 1
                                                xr = 0
                                            else:
                                                yr = 0
                                                xr = 1
                                            compensation_list = []
                                            if yii - ti in image_save_dict and xii - tj in image_save_dict[yii - ti] \
                                                    and key in image_save_dict[yii - ti][xii - tj]["direction"]:
                                                compensation_list = [0, 0]
                                                if initial_variable.test_signal == 1:
                                                    initial_variable.comparator_num += 1
                                            elif yii - ti - yr in image_save_dict and xii - tj - xr in image_save_dict[
                                                yii - ti - yr] and \
                                                    key in image_save_dict[yii - ti - yr][xii - tj - xr]["direction"]:
                                                compensation_list = [-yr, -xr]
                                                if initial_variable.test_signal == 1:
                                                    initial_variable.comparator_num += 1
                                            elif yii - ti + yr in image_save_dict and \
                                                    xii - tj + xr in image_save_dict[yii - ti + yr] and \
                                                    key in image_save_dict[yii - ti + yr][xii - tj + xr]["direction"]:
                                                compensation_list = [yr, xr]
                                                if initial_variable.test_signal == 1:
                                                    initial_variable.comparator_num += 1

                                            if len(compensation_list) > 0:
                                                compensation_y1 = compensation_list[0]
                                                compensation_x1 = compensation_list[1]
                                                if (yii - ti + compensation_y1, xii - tj + compensation_x1) in \
                                                        image_save_scanning_temp:
                                                    if image_save_scanning_temp[(yii - ti + compensation_y1,
                                                                                 xii - tj + compensation_x1)] != mark_num:
                                                        save_dict[(yii, xii)] = []
                                                        continue
                                                dict_temp = image_save_dict[yii - ti + compensation_y1][xii - tj + compensation_x1]

                                                if (next_signal == 0 <= dict_temp["mini_encode"][0] - now_encode_0 <= life_num) or \
                                                        (next_signal == 1 and 0 <= now_encode_0 - dict_temp["mini_encode"][0] <= life_num):

                                                    activate_dict_temp[(yii - ti + compensation_y1,
                                                                        xii - tj + compensation_x1)] = 1

                                                    life = 0
                                                    match_num += 1
                                                    image_save_scanning_temp[
                                                        (yii - ti + compensation_y1, xii - tj + compensation_x1)] = mark_num
                                                    if dict_temp["mini_encode"][0] < 200:
                                                        now_encode_0 = dict_temp["mini_encode"][0]
                                                    if next_signal == 0 or first_time == 0:
                                                        if dict_temp["mini_encode"][0] < 200:
                                                            encode[0] = dict_temp["mini_encode"][0]
                                                            now_mini = dict_temp["mini_encode"][0]
                                                            boundary_dict[0] = now_mini
                                                            if first_time == 0:
                                                                encode[1] = dict_temp["mini_encode"][0]
                                                                boundary_dict[1] = dict_temp["mini_encode"][0]
                                                                first_time = 1
                                                    if next_signal == 1:
                                                        if dict_temp["mini_encode"][0] < 200:
                                                            encode[1] = dict_temp["mini_encode"][0]
                                                            now_mini = dict_temp["mini_encode"][0]
                                                            boundary_dict[1] = now_mini
                                                else:
                                                    save_dict[(yii, xii)] = []
                                                    continue
                                                save_dict[(yii, xii)] = [yii - ti + compensation_y1,
                                                                         xii - tj + compensation_x1]
                                                break
                                            else:
                                                save_dict[(yii, xii)] = []
                                        if next_signal == 0:
                                            for tuple_temp in dict_or["next"]:
                                                serve_list_temp.append([tuple_temp[0], tuple_temp[1], life, 0,
                                                                        now_encode_0])
                                        else:
                                            for tuple_temp in dict_or["previous"]:
                                                serve_list_temp.append([tuple_temp[0], tuple_temp[1], life, 1,
                                                                        now_encode_0])
                                serve_list.clear()
                                serve_list += serve_list_temp
                                serve_list_temp.clear()

                    if encode[0] - encode[1] >= 4 and ((length_num >= 8 and match_num >= 5) or
                                                       (length_num < 8 and match_num >= 3)):
                        scanning_point = encode[0]
                        save_scanning.update(save_scanning_copy)
                        activate_dict = copy.deepcopy(activate_dict_temp)
                        repeat_recognise_num += repeat_recognise_num_temp
                        if repeat_recognise_num / length_num > repeat_recognize_ratio:
                            return []

                        for direction_temp in boundary_dict:
                            total_boundary_dict[boundary_dict[direction_temp]] = 1
                        save_dict_2 = copy.deepcopy(save_dict)
                        total_save_dict_simple.update(save_dict_2)
                        record_slice[mark_num] = [encode[0], encode[1], save_dict_2]
                        image_save_scanning.update(image_save_scanning_temp)
                        mark_num += 1
                    else:
                        activate_dict_temp = copy.deepcopy(activate_dict)
                    save_dict.clear()
                    image_save_scanning_temp.clear()
                    boundary_dict.clear()
                    encode = [0, 0]
                    repeat_recognise_num_temp = 0
            if previous_scanning_point == scanning_point:
                fail_num += 1
                if fail_num / length_num > 0.4:
                    if initial_variable.test_signal == 1:
                        initial_variable.multiplier_num += 1
                    return []
        scanning_point += 1

    if len(record_slice) > 0:
        recognition_length = 0
        for mark_num in record_slice:
            recognition_length += record_slice[mark_num][0] - record_slice[mark_num][1] + 1
            if initial_variable.test_signal == 1:
                initial_variable.adder_num += 1
        ratio = recognition_length / length_num
        if initial_variable.test_signal == 1:
            initial_variable.multiplier_num += 1
        if ratio >= recognition_ratio:
            return [recognition_length, record_slice, activate_dict, total_save_dict_simple,
                    total_boundary_dict]

    return []


def merge_image(record_slice, image_test_dict, image_save_dict, refine_activate_num,
                total_save_dict, scanning_image_dict, scanning_image_or_dict, total_boundary_dict):
    """This function is used to incorporate input boundary points into memory"""
    save_scanning = {}
    y_temp = 0
    x_temp = 0
    serve_list_temp = []
    repeat_dict = {}
    for mark_num in record_slice:
        save_dict = record_slice[mark_num][2]
        for (yi, xi) in save_dict:
            if len(save_dict[(yi, xi)]) == 2:
                for direction in range(2):
                    serve_list = [[yi, xi, y_temp, x_temp, direction]]
                    if (yi, xi) in save_scanning:
                        save_scanning.pop((yi, xi))
                    while len(serve_list) > 0:
                        repeat_dict.clear()
                        for temp_list in serve_list:
                            yii = temp_list[0]
                            xii = temp_list[1]
                            y_temp = temp_list[2]
                            x_temp = temp_list[3]
                            direction = temp_list[4]

                            if (yii, xii) not in save_scanning and (yii, xii) in save_dict:
                                save_scanning[(yii, xii)] = 1
                                if len(save_dict[(yii, xii)]) == 2:
                                    [y_save, x_save] = save_dict[(yii, xii)]
                                    y_temp = y_save - yii
                                    x_temp = x_save - xii
                                new_merge_image(y_temp, x_temp, yii, xii,
                                                total_save_dict, image_save_dict,
                                                image_test_dict, refine_activate_num, scanning_image_dict,
                                                scanning_image_or_dict,
                                                total_boundary_dict)
                                dict_or = image_test_dict[yii][xii]
                                if direction == 0:
                                    for tuple_temp in dict_or["next"]:
                                        if tuple_temp not in repeat_dict:
                                            repeat_dict[tuple_temp] = 1
                                            serve_list_temp.append(
                                                [tuple_temp[0], tuple_temp[1], y_temp, x_temp, direction])
                                else:
                                    for tuple_temp in dict_or["previous"]:
                                        if tuple_temp not in repeat_dict:
                                            repeat_dict[tuple_temp] = 1
                                            serve_list_temp.append(
                                                [tuple_temp[0], tuple_temp[1], y_temp, x_temp, direction])
                        serve_list.clear()
                        if len(serve_list_temp) > 0:
                            serve_list += serve_list_temp
                            serve_list_temp.clear()


def new_merge_image(compensation_y, compensation_x, yi, xi, save_dict, image_save_dict,
                    image_test_dict, refine_activate_num, scanning_image_dict, scanning_image_or_dict,
                    total_boundary_dict):
    """This function is used to determine the label of each input boundary point"""
    yii = compensation_y + yi
    xii = compensation_x + xi
    execute_signal = 0

    if yii in image_save_dict and xii in image_save_dict[yii]:
        save_dict[(yi, xi)] = [0, 0, (yii, xii)]
        mini_encode = image_save_dict[yii][xii]["mini_encode"][0]
        execute_signal = 1
        image_save_dict[yii][xii]["mini_encode_temp"] = mini_encode
    elif yii < 28 and xii < 28:
        dict_temp = image_test_dict[yi][xi]
        execute_signal = 1
        new_dict = {"direction": copy.deepcopy(dict_temp["direction"]),
                    "previous": {},
                    "next": {},
                    "encode": [1000, 1000],
                    "mini_encode": [1000, 1000],
                    "mini_encode_temp": 1000,
                    "encode_recode": {},
                    "previous_end": 0,
                    "next_end": 0,
                    "activate_num": refine_activate_num,
                    "increase_ratio": 2}
        if yii in image_save_dict and xii not in image_save_dict[yii]:
            image_save_dict[yii][xii] = new_dict
        elif yii not in image_save_dict:
            image_save_dict[yii] = {xii: new_dict}
        save_dict[(yi, xi)] = [0, 0, (yii, xii)]

    if execute_signal == 1:
        dict_temp = image_test_dict[yi][xi]
        self_dict = image_save_dict[yii][xii]
        for (y_t, x_t) in dict_temp["next"]:
            if (y_t, x_t) in save_dict and len(save_dict[(y_t, x_t)]) == 3:
                (save_y, save_x) = save_dict[(y_t, x_t)][2]
                record_dict = image_save_dict[save_y][save_x]
                if (yii, xii) not in record_dict["previous"]:
                    record_dict["previous"][(yii, xii)] = 1
                    self_dict["next"][(save_y, save_x)] = 1
                    if self_dict["encode"][0] != 0 and len(record_dict["encode_recode"]) > 0 and \
                            7 >= record_dict["encode_recode"][1] and record_dict["mini_encode_temp"] < 1000:
                        self_dict["mini_encode_temp"] = record_dict["mini_encode_temp"]

        for (y_t, x_t) in dict_temp["previous"]:
            if (y_t, x_t) in save_dict and len(save_dict[(y_t, x_t)]) == 3:
                (save_y, save_x) = save_dict[(y_t, x_t)][2]
                record_dict = image_save_dict[save_y][save_x]
                if (yii, xii) not in record_dict["next"]:
                    record_dict["next"][(yii, xii)] = 1
                    self_dict["previous"][(save_y, save_x)] = 1
                    if self_dict["encode"][0] != 0 and len(record_dict["encode_recode"]) > 0 and \
                            7 >= record_dict["encode_recode"][1] and record_dict["mini_encode_temp"] < 1000:
                        self_dict["mini_encode_temp"] = record_dict["mini_encode_temp"]
        server_list = [[yii, xii]]
        server_list_temp = []
        total_scanning_dict = {}
        while len(server_list) > 0:
            for son_server_list in server_list:
                yii = son_server_list[0]
                xii = son_server_list[1]
                dict_or = image_save_dict[yii][xii]
                signal_trans = 0
                direction_temp_list = []
                if dict_or["encode"][0] != 0 and 1000 > dict_or["mini_encode_temp"]:
                    if (yii, xii) in total_scanning_dict:
                        continue
                    total_scanning_dict[(yii, xii)] = 1

                    mini_encode_temp = dict_or["mini_encode_temp"]
                    previous_mini = mini_encode_temp - 3
                    future_mini = mini_encode_temp + 3
                    for ii in range(previous_mini, future_mini + 1):
                        direction_temp_list.append(ii)
                    scanning_dict = {}
                    min_distance = 8
                    while len(direction_temp_list) > 0:
                        for replace_mini_encode in direction_temp_list:
                            scanning_dict[replace_mini_encode] = 1
                            if replace_mini_encode in scanning_image_or_dict:
                                (y_other, x_other) = scanning_image_or_dict[replace_mini_encode]
                                distance_other = abs(yii - y_other) + abs(xii - x_other)
                                if min_distance > distance_other:
                                    signal_trans = 1
                                    dict_or["mini_encode"][0] = replace_mini_encode
                                    dict_or["mini_encode_temp"] = replace_mini_encode
                                    dict_or["encode_recode"][0] = (y_other, x_other)
                                    dict_or["encode_recode"][1] = distance_other
                                    dict_or["encode"][0] = 0
                                    min_distance = distance_other
                        direction_temp_list.clear()

                    if signal_trans == 1:
                        if (yii, xii) not in scanning_image_dict[dict_or["mini_encode"][0]]:
                            scanning_image_dict[dict_or["mini_encode"][0]][(yii, xii)] = 1
                        mini_encode_temp = dict_or["mini_encode_temp"]
                        for tuple_temp in dict_or["next"]:
                            if tuple_temp[0] in image_save_dict and tuple_temp[1] in image_save_dict[tuple_temp[0]]:
                                dict_temp = image_save_dict[tuple_temp[0]][tuple_temp[1]]
                                if tuple_temp not in server_list_temp and dict_temp["encode"][0] != 0:
                                    dict_temp["mini_encode_temp"] = mini_encode_temp
                                    server_list_temp.append([tuple_temp[0], tuple_temp[1]])
                        dict_or["next"].clear()

                        for tuple_temp in dict_or["previous"]:
                            if tuple_temp[0] in image_save_dict and tuple_temp[1] in image_save_dict[tuple_temp[0]]:
                                dict_temp = image_save_dict[tuple_temp[0]][tuple_temp[1]]
                                if tuple_temp not in server_list_temp and dict_temp["encode"][0] != 0:
                                    dict_temp["mini_encode_temp"] = mini_encode_temp
                                    server_list_temp.append([tuple_temp[0], tuple_temp[1]])
                        dict_or["previous"].clear()
            server_list.clear()
            if len(server_list_temp) > 0:
                server_list += server_list_temp
                server_list_temp.clear()

    if yii in image_save_dict and xii in image_save_dict[yii]:
        dict_temp = image_save_dict[yii][xii]
        if len(dict_temp["encode_recode"]) > 0:
            if dict_temp["mini_encode"][0] in total_boundary_dict and dict_temp["encode_recode"][1] > 3:
                dict_temp["encode"][1] = 0


def merge_common_f(group_label, total_save_dict, image_name_dict):
    """image_name_dict主要是赋值image_test_dict，但是加入了name list
    在find_neighbor前面执行"""
    for (yi, xi) in total_save_dict:
        if group_label not in image_name_dict[yi][xi]:
            image_name_dict[yi][xi].append(group_label)


def refine_group_update(image_save_dict, life_num, start_ratio, end_ratio, feature_num,
                        refine_total_list, extend_total_list, point_max_serve, point_max_sort):
    """This function is used to refine the original part into smaller parts"""
    if len(point_max_sort) == 0:
        return False
    refine_dict = {}
    refine_list = []
    total_list = []

    extend_dict = {}
    refine_num_list = []
    max_number = 0
    refine_ratio = start_ratio
    for sort_list in point_max_sort:
        max_number += sort_list[1]
        if end_ratio >= max_number / feature_num >= refine_ratio:
            refine_num_list.append(sort_list[0])
            refine_ratio = max_number / feature_num + 0.05
            if end_ratio < refine_ratio:
                break
    if len(refine_num_list) > 0:
        refine_num_list.reverse()

        for max_refine_num in refine_num_list:
            refine_list.clear()
            total_list.clear()
            break_signal = 0

            for mini_encode in point_max_serve:
                if point_max_serve[mini_encode][0] >= max_refine_num:
                    refine_list.append(mini_encode)
                elif len(point_max_serve[mini_encode]) == 2 and point_max_serve[mini_encode][0] >= max_refine_num - 10:
                    total_list.append(mini_encode)

            if len(refine_list) > 0:
                refine_list.sort()
                i_list = []
                num_temp = 0
                for num_refine in range(len(refine_list)):
                    if refine_list[num_refine] - refine_list[num_temp] < life_num:
                        num_temp = num_refine
                        i_list.append(refine_list[num_refine])
                    else:
                        if len(i_list) / feature_num > 0.9:
                            break_signal = 1
                            break
                        elif len(i_list) >= 5:
                            refine_dict.clear()
                            previous_mini_encode = i_list[0]
                            for mini_encode_num in i_list:
                                if mini_encode_num - previous_mini_encode > 1:
                                    for j_temp in range(previous_mini_encode + 1, mini_encode_num):
                                        if j_temp in point_max_serve and len(point_max_serve[j_temp]) == 2:
                                            refine_dict[j_temp] = point_max_serve[j_temp][1]
                                            if j_temp in total_list:
                                                total_list.remove(j_temp)
                                previous_mini_encode = mini_encode_num
                                if mini_encode_num in total_list:
                                    total_list.remove(mini_encode_num)
                                refine_dict[mini_encode_num] = point_max_serve[mini_encode_num][1]
                            if len(refine_dict) / feature_num > 0.9:
                                break_signal = 1
                                break
                            refine_total_list.append(copy.deepcopy(refine_dict))
                        i_list.clear()
                        num_temp = num_refine
                        i_list.append(refine_list[num_refine])
                if break_signal == 0:
                    if len(i_list) / feature_num > 0.9:
                        continue
                    elif len(i_list) >= 5:
                        refine_dict.clear()
                        previous_mini_encode = i_list[0]
                        for mini_encode_num in i_list:
                            if mini_encode_num - previous_mini_encode > 1:
                                for j_temp in range(previous_mini_encode + 1, mini_encode_num):
                                    if j_temp in point_max_serve and len(point_max_serve[j_temp]) == 2:
                                        refine_dict[j_temp] = point_max_serve[j_temp][1]
                                        if j_temp in total_list:
                                            total_list.remove(j_temp)
                            previous_mini_encode = mini_encode_num
                            if mini_encode_num in total_list:
                                total_list.remove(mini_encode_num)
                            refine_dict[mini_encode_num] = point_max_serve[mini_encode_num][1]
                        if len(refine_dict) / feature_num > 0.9:
                            continue
                        refine_total_list.append(copy.deepcopy(refine_dict))

                    if len(refine_total_list) > 0:
                        num_temp = 0
                        i_list.clear()
                        if len(total_list) > 0:
                            total_list.sort()
                            for num_total in range(len(total_list)):
                                if total_list[num_total] - total_list[num_temp] < life_num:
                                    num_temp = num_total
                                    i_list.append(total_list[num_total])
                                else:
                                    if len(i_list) >= 5:
                                        extend_dict.clear()
                                        for mini_encode_num in i_list:
                                            extend_dict[mini_encode_num] = point_max_serve[mini_encode_num][1]
                                        refine_total_list.append(copy.deepcopy(extend_dict))
                                    i_list.clear()
                                    num_temp = num_total
                                    i_list.append(total_list[num_total])
                            if len(i_list) >= 5:
                                extend_dict.clear()
                                for mini_encode_num in i_list:
                                    extend_dict[mini_encode_num] = point_max_serve[mini_encode_num][1]
                                refine_total_list.append(copy.deepcopy(extend_dict))
                        return True
    return False


def create_refine_group(refine_total_list, refine_save_image_list, image_save_dict, scanning_image_dict):
    """This function is used to prepare for generating a part"""
    mini_encode_list = []
    refine_save_dict = {}
    son_scanning_image_dict = {}
    son_scanning_image_or_dict = {}
    activate_sort_dict = copy.deepcopy(scanning_image_dict)
    for refine_dict in refine_total_list:
        mini_encode_list.clear()
        for mini_encode in refine_dict:
            mini_encode_list.append(mini_encode)
        mini_encode_list.sort()

        encode_num = 0
        refine_save_dict.clear()
        son_scanning_image_dict.clear()
        son_scanning_image_or_dict.clear()
        for mini_encode in mini_encode_list:
            execute = 0
            (yi_first, xi_first) = refine_dict[mini_encode]
            if yi_first not in refine_save_dict:
                refine_save_dict[yi_first] = {xi_first: copy.deepcopy(image_save_dict[yi_first][xi_first])}
            elif xi_first not in refine_save_dict[yi_first]:
                refine_save_dict[yi_first][xi_first] = copy.deepcopy(image_save_dict[yi_first][xi_first])
            refine_save_dict[yi_first][xi_first]["mini_encode"][0] = encode_num
            son_scanning_image_dict[encode_num] = {(yi_first, xi_first): 1}
            son_scanning_image_or_dict[encode_num] = (yi_first, xi_first)
            if refine_save_dict[yi_first][xi_first]["encode_recode"][1] != 0:
                execute = 1
                refine_save_dict[yi_first][xi_first]["encode_recode"][0] = (yi_first, xi_first)
                refine_save_dict[yi_first][xi_first]["encode_recode"][1] = 0
                refine_save_dict[yi_first][xi_first]["next"].clear()
                refine_save_dict[yi_first][xi_first]["previous"].clear()
                if (yi_first, xi_first) in activate_sort_dict[mini_encode]:
                    activate_sort_dict[mini_encode].pop((yi_first, xi_first))

            if len(activate_sort_dict[mini_encode]) >= 1 and execute == 1:
                for (yi_refine, xi_refine) in activate_sort_dict[mini_encode]:
                    if yi_refine not in refine_save_dict:
                        refine_save_dict[yi_refine] = {xi_refine: copy.deepcopy(image_save_dict[yi_refine][xi_refine])}
                    elif xi_refine not in refine_save_dict[yi_refine]:
                        refine_save_dict[yi_refine][xi_refine] = copy.deepcopy(image_save_dict[yi_refine][xi_refine])
                    refine_save_dict[yi_refine][xi_refine]["mini_encode"][0] = encode_num
                    son_scanning_image_dict[encode_num][(yi_refine, xi_refine)] = 1
                    refine_save_dict[yi_refine][xi_refine]["encode"][0] = 1000
                    refine_save_dict[yi_refine][xi_refine]["encode_recode"][0] = (yi_first, xi_first)
                    refine_save_dict[yi_refine][xi_refine]["encode_recode"][1] = \
                        abs(yi_refine - yi_first) + abs(xi_refine - xi_first)
                    refine_save_dict[yi_refine][xi_refine]["next"].clear()
                    refine_save_dict[yi_refine][xi_refine]["previous"].clear()
            elif len(activate_sort_dict[mini_encode]) >= 1 and execute == 0:
                for (yi_m, xi_m) in activate_sort_dict[mini_encode]:
                    if yi_m not in refine_save_dict:
                        refine_save_dict[yi_m] = {xi_m: copy.deepcopy(image_save_dict[yi_m][xi_m])}
                    elif xi_m not in refine_save_dict[yi_m]:
                        refine_save_dict[yi_m][xi_m] = copy.deepcopy(image_save_dict[yi_m][xi_m])
                    refine_save_dict[yi_m][xi_m]["mini_encode"][0] = encode_num
                    son_scanning_image_dict[encode_num][(yi_m, xi_m)] = 1
            encode_num += 1
        refine_save_image_list.append([copy.deepcopy(refine_save_dict), encode_num,
                                       copy.deepcopy(son_scanning_image_dict),
                                       copy.deepcopy(son_scanning_image_or_dict)])


def update_save_image_activate_num(image_save_dict, activate_dict, max_image_save_point, correct,
                                   scanning_image_dict, scanning_image_or_dict, execute_update_point,
                                   point_max_serve, point_max_sort, feature_num):
    """This function is used to update the activations number of the boundary points in memory and to count the maximum
    distribution of the activations number of each label
    A label is a set of boundary points. These boundary points have the same meaning in matching."""
    refine_activate_num = max_image_save_point - 20
    if refine_activate_num < 1:
        refine_activate_num = 1

    self_max_num = 0
    if correct >= 1:
        number = 1
        increase_number = 1
    else:
        number = -2
        increase_number = -2

    for (yii, xii) in activate_dict:
        activate_number = image_save_dict[yii][xii]["activate_num"]
        if activate_number < refine_activate_num and correct >= 1:
            image_save_dict[yii][xii]["activate_num"] += image_save_dict[yii][xii]["increase_ratio"]
            if image_save_dict[yii][xii]["activate_num"] > refine_activate_num:
                image_save_dict[yii][xii]["activate_num"] = refine_activate_num
        else:
            if image_save_dict[yii][xii]["increase_ratio"] > 1:
                image_save_dict[yii][xii]["activate_num"] += number
        if image_save_dict[yii][xii]["activate_num"] > max_image_save_point:
            self_max_num = image_save_dict[yii][xii]["activate_num"]

        image_save_dict[yii][xii]["increase_ratio"] += increase_number
    if max_image_save_point < 15:
        return self_max_num
    elif execute_update_point == 1 and correct >= 1:
        del_dict = {}
        del_num = max_image_save_point - 35
        point_max_sort_dict = {}
        scanning_image_dict_copy = copy.deepcopy(scanning_image_dict)
        for mini_encode in scanning_image_dict_copy:
            if mini_encode not in point_max_serve:
                point_max_serve[mini_encode] = [-1]
            for (yii, xii) in scanning_image_dict_copy[mini_encode]:
                activate_num = image_save_dict[yii][xii]["activate_num"]
                if activate_num <= del_num or image_save_dict[yii][xii]["encode"][1] == 0 or \
                        image_save_dict[yii][xii]["mini_encode"][0] > 200:
                    if (yii, xii) != scanning_image_or_dict[mini_encode]:
                        del_dict[(yii, xii)] = 1
                    if (yii, xii) in scanning_image_dict[mini_encode]:
                        scanning_image_dict[mini_encode].pop((yii, xii))
                    continue
                if activate_num > point_max_serve[mini_encode][0]:
                    point_max_serve[mini_encode] = [activate_num, (yii, xii)]
            if point_max_serve[mini_encode][0] not in point_max_sort_dict:
                point_max_sort_dict[point_max_serve[mini_encode][0]] = 1
            else:
                point_max_sort_dict[point_max_serve[mini_encode][0]] += 1

        if len(del_dict) > 0:
            for (yi, xi) in del_dict:
                for (y_next, x_next) in image_save_dict[yi][xi]["next"]:
                    if y_next in image_save_dict and x_next in image_save_dict[y_next]:
                        if (yi, xi) in image_save_dict[y_next][x_next]["previous"]:
                            image_save_dict[y_next][x_next]["previous"].pop((yi, xi))
                for (y_previous, x_previous) in image_save_dict[yi][xi]["previous"]:
                    if y_previous in image_save_dict and x_previous in image_save_dict[y_previous]:
                        if (yi, xi) in image_save_dict[y_previous][x_previous]["next"]:
                            image_save_dict[y_previous][x_previous]["next"].pop((yi, xi))
                image_save_dict[yi].pop(xi)
                if len(image_save_dict[yi]) == 0:
                    image_save_dict.pop(yi)
        if len(point_max_sort_dict) >= 4:
            for max_number in point_max_sort_dict:
                point_max_sort.append([max_number, point_max_sort_dict[max_number]])
            point_max_sort.sort(key=lambda x: x[0], reverse=True)
            merge_number = 0
            for son_list in point_max_sort:
                merge_number += son_list[1]
                if merge_number / feature_num > 0.6:
                    self_max_num = son_list[0]
    return self_max_num


def create_group_label(generator_entity, image_save_dict, feature_num, father_mini,
                       father_group_list, length_str_dict, scanning_image_dict, scanning_image_or_dict,
                       small_group=None, aim_dict=None, signal=0, execute_signal=0):
    """This function is used to generate new part"""
    if len(initial_variable.recycle_start_group_list) == 0:
        start_group_label = str(len(start_group_dict))
        while start_group_label in start_group_dict:
            group_num = int(start_group_label)
            group_num += 1
            start_group_label = str(group_num)
        start_generator_ob = generator_entity(start_group_label, image_save_dict, feature_num, father_mini,
                                              scanning_image_dict, scanning_image_or_dict)
        if feature_num * 0.3 < 3:
            start_generator_ob.life_num = 2
        else:
            start_generator_ob.life_num = 3
        start_group_dict[start_group_label] = start_generator_ob
    else:
        start_group_label = initial_variable.recycle_start_group_list.pop(0)
        start_generator_ob = start_group_dict[start_group_label]
        start_generator_ob.delete = 0
        start_generator_ob.execute = 0
        start_generator_ob.success = 0
        start_generator_ob.replace_signal = 0
        start_generator_ob.result_dict.clear()
        start_generator_ob.total_save_dict.clear()
        start_generator_ob.activate_dict.clear()
        start_generator_ob.image_save_dict = image_save_dict
        start_generator_ob.scanning_image_dict = scanning_image_dict
        start_generator_ob.scanning_image_or_dict = scanning_image_or_dict
        start_generator_ob.feature_num = feature_num
        if feature_num * 0.3 < 3:
            start_generator_ob.life_num = 2
        else:
            start_generator_ob.life_num = 3
        start_generator_ob.correct_num = 0
        start_generator_ob.father_mini = father_mini
        start_generator_ob.father_mini_dict.clear()
        start_generator_ob.can_refine = 1
        start_generator_ob.total_train_num = 0
        start_generator_ob.continue_num = 0
        start_generator_ob.replace_save.clear()

        start_generator_ob.trans_son_total.clear()
        start_generator_ob.trans_son_total_num.clear()
        start_generator_ob.trans_num.clear()
        start_generator_ob.trans_replace_num.clear()
        start_generator_ob.trans_replace_max.clear()
        start_generator_ob.trans_start_num.clear()
        start_generator_ob.trans_total_num.clear()
        start_generator_ob.trans_activate_num.clear()
        start_generator_ob.trans_base_num.clear()

    if feature_num <= 5:
        start_generator_ob.can_refine = 0
    if signal == 0:
        if len(father_group_list) > 0 and execute_signal == 0:
            for father_group in father_group_list:
                father_group_ob = start_group_dict[father_group]
                start_generator_ob.sort_trans_father[father_group] = 0
                if father_group not in length_str_dict:
                    if len(father_group_ob.activate_son) == 0:
                        son_group = str(len(father_group_ob.trans_son_total))
                        while son_group in father_group_ob.trans_son_total:
                            son_group_num = int(son_group)
                            son_group_num += 1
                            son_group = str(son_group_num)
                        father_group_ob.trans_son_total_num[son_group] = [-1, 2, 0]
                        father_group_ob.trans_son_total[son_group] = {start_group_label: {"0": {"0": [start_group_label]}}}
                        father_group_ob.trans_num[son_group] = {start_group_label: {"0": {"0": [1, 2]}}}
                        father_group_ob.trans_replace_num[son_group] = {start_group_label: {"0": [1, 2]}}
                        father_group_ob.trans_replace_max[son_group] = {start_group_label: 1}
                        father_group_ob.trans_start_num[son_group] = {start_group_label: -1}
                        father_group_ob.trans_total_num[son_group] = feature_num
                        father_group_ob.trans_activate_num[son_group] = {start_group_label: feature_num}
                        start_generator_ob.trans_father_total[father_group] = {son_group: {start_group_label: {"0": "0"}}}
                        length_str_dict[father_group] = son_group
                    else:
                        if father_group not in length_str_dict:
                            for son_group in father_group_ob.activate_son:
                                other_son_group = str(len(father_group_ob.trans_son_total))
                                while other_son_group in father_group_ob.trans_son_total:
                                    other_son_group_num = int(other_son_group)
                                    other_son_group_num += 1
                                    other_son_group = str(other_son_group_num)
                                father_group_ob.trans_son_total_num[other_son_group] = [-1, 2, 0]  # [-1, 2, 0]
                                father_group_ob.trans_son_total[other_son_group] = {}
                                father_group_ob.trans_num[other_son_group] = {}
                                father_group_ob.trans_replace_num[other_son_group] = {}
                                father_group_ob.trans_replace_max[other_son_group] = {}
                                father_group_ob.trans_start_num[other_son_group] = {}
                                father_group_ob.trans_total_num[other_son_group] = 0
                                father_group_ob.trans_activate_num[other_son_group] = {}
                                father_group_ob.trans_base_num[other_son_group] = {}
                                for start_group_other in father_group_ob.activate_son[son_group]:
                                    father_group_ob.trans_son_total[other_son_group][start_group_other] = \
                                        copy.deepcopy(father_group_ob.trans_son_total[son_group][start_group_other])
                                    father_group_ob.trans_num[other_son_group][start_group_other] = \
                                        copy.deepcopy(father_group_ob.trans_num[son_group][start_group_other])
                                    father_group_ob.trans_replace_num[other_son_group][start_group_other] = \
                                        copy.deepcopy(father_group_ob.trans_replace_num[son_group][start_group_other])
                                    if start_group_other in father_group_ob.trans_replace_max[son_group]:
                                        father_group_ob.trans_replace_max[other_son_group][start_group_other] = \
                                            copy.deepcopy(father_group_ob.trans_replace_max[son_group][start_group_other])
                                    father_group_ob.trans_start_num[other_son_group][start_group_other] = \
                                        father_group_ob.trans_start_num[son_group][start_group_other]
                                    father_group_ob.trans_total_num[other_son_group] += \
                                        father_group_ob.trans_activate_num[son_group][start_group_other]
                                    father_group_ob.trans_activate_num[other_son_group][start_group_other] = \
                                        father_group_ob.trans_activate_num[son_group][start_group_other]
                                    if son_group in father_group_ob.trans_base_num and \
                                            start_group_other in father_group_ob.trans_base_num[son_group]:
                                        father_group_ob.trans_base_num[other_son_group][start_group_other] = \
                                            father_group_ob.trans_base_num[son_group][start_group_other]
                                    for replace_key_other in \
                                            father_group_ob.trans_son_total[son_group][start_group_other]:
                                        for group_key_other in \
                                                father_group_ob.trans_son_total[son_group][start_group_other][replace_key_other]:
                                            for group_other in \
                                                    father_group_ob.trans_son_total[son_group][start_group_other][replace_key_other][group_key_other]:
                                                group_ob_other = start_group_dict[group_other]
                                                group_ob_other.trans_father_total[father_group_ob.group_label][
                                                    other_son_group] = {start_group_other: {replace_key_other: group_key_other}}

                                start_group_name = start_group_label
                                while start_group_name in father_group_ob.trans_son_total[other_son_group]:
                                    start_group_num = int(start_group_name)
                                    start_group_num += 1
                                    start_group_name = str(start_group_num)
                                father_group_ob.trans_son_total[other_son_group][start_group_name] = {
                                    "0": {"0": [start_group_label]}}
                                father_group_ob.trans_num[other_son_group][start_group_name] = {"0": {"0": [1, 2]}}
                                father_group_ob.trans_replace_num[other_son_group][start_group_name] = {"0": [1, 2]}
                                father_group_ob.trans_replace_max[other_son_group][start_group_name] = 1
                                father_group_ob.trans_start_num[other_son_group][start_group_name] = -1
                                father_group_ob.trans_total_num[other_son_group] += feature_num
                                father_group_ob.trans_activate_num[other_son_group][start_group_name] = feature_num
                                start_generator_ob.trans_father_total[father_group] = \
                                    {other_son_group: {start_group_name: {"0": "0"}}}
                                start_generator_ob.sort_trans_father[father_group] = 1
                                length_str_dict[father_group] = other_son_group
                else:
                    son_group = length_str_dict[father_group]
                    start_group_name = start_group_label
                    while start_group_name in father_group_ob.trans_son_total[son_group]:
                        start_group_num = int(start_group_name)
                        start_group_num += 1
                        start_group_name = str(start_group_num)
                    father_group_ob.trans_son_total[son_group][start_group_name] = {"0": {"0": [start_group_label]}}
                    father_group_ob.trans_num[son_group][start_group_name] = {"0": {"0": [1, 2]}}
                    father_group_ob.trans_replace_num[son_group][start_group_name] = {"0": [1, 2]}
                    father_group_ob.trans_replace_max[son_group][start_group_name] = 1
                    father_group_ob.trans_start_num[son_group][start_group_name] = -1
                    father_group_ob.trans_total_num[son_group] += feature_num
                    father_group_ob.trans_activate_num[son_group][start_group_name] = feature_num
                    start_generator_ob.trans_father_total[father_group] = {son_group: {start_group_name: {"0": "0"}}}
                    start_generator_ob.sort_trans_father[father_group] = 1

        elif len(father_mini) > 0 or (execute_signal == 1 and len(father_group_list) > 0):
            if len(father_mini) > 0 and execute_signal == 0:
                father_mini_label = father_mini[0]
                father_mini_ob = initial_variable.mini_group_dict[father_mini_label]
                small_group_num = father_mini_ob.small_group_num
                if small_group is None:
                    father_mini_ob.small_group[start_group_label] = {"0": {"0": [start_group_label]}}
                    father_mini_ob.small_group_num[start_group_label] = {"0": {"0": [-1, 2]}}
                    father_mini_ob.small_replace_num[start_group_label] = {"0": [1, 2]}
                    father_mini_ob.small_replace_max[start_group_label] = 1
                    father_mini_ob.small_start_num[start_group_label] = 1
                    father_mini_ob.small_total_num += feature_num
                    father_mini_ob.small_activate_num[start_group_label] = feature_num
                    start_generator_ob.father_mini_dict[father_mini_label] = {start_group_label: {"0": "0"}}
            else:
                start_generator_ob.sort_trans_father[father_group_list[0]] = 1
                father_trans_label = father_group_list[0]
                father_trans_group_ob = start_group_dict[father_trans_label]
                key_list = list(aim_dict)
                small_group_num = father_trans_group_ob.trans_num[key_list[0]]
            if small_group is not None:
                for father_mini_label in aim_dict:
                    for start_group_key in aim_dict[father_mini_label]:
                        for replace_key in aim_dict[father_mini_label][start_group_key]:
                            if len(aim_dict[father_mini_label][start_group_key][replace_key]) == 0:
                                group_key = str(len(small_group[father_mini_label][start_group_key][replace_key]))
                                while group_key in small_group[father_mini_label][start_group_key][replace_key]:
                                    group_num = int(group_key)
                                    group_num += 1
                                    group_key = str(group_num)
                                small_group[father_mini_label][start_group_key][replace_key][group_key] = [start_group_label]
                                small_group_num[start_group_key][replace_key][group_key] = [-1, 2]
                                if len(father_mini) > 0 and execute_signal == 0:
                                    start_generator_ob.father_mini_dict[father_mini_label] = {start_group_key:
                                                                                              {replace_key: group_key}}
                                else:
                                    start_generator_ob.trans_father_total[father_group_list[0]] = \
                                        {father_mini_label: {start_group_key: {replace_key: group_key}}}
                                aim_dict[father_mini_label][start_group_key][replace_key] = [group_key]
                            else:
                                group_key = aim_dict[father_mini_label][start_group_key][replace_key][0]
                                small_group[father_mini_label][start_group_key][replace_key][group_key].append(start_group_label)
                                if len(father_mini) > 0 and execute_signal == 0:
                                    start_generator_ob.father_mini_dict[father_mini_label] = {start_group_key:
                                                                                              {replace_key: group_key}}
                                else:
                                    start_generator_ob.trans_father_total[father_group_list[0]] = \
                                        {father_mini_label: {start_group_key: {replace_key: group_key}}}

    return start_group_label, length_str_dict


def write_son_group(father_group_list, son_group_list, length_str_dict):
    for father_group in father_group_list:
        father_group_ob = start_group_dict[father_group]
        for son_group_label in son_group_list:
            son_group_ob = start_group_dict[son_group_label]
            if father_group not in length_str_dict:
                son_group = str(len(father_group_ob.trans_son_total))
                while son_group in father_group_ob.trans_son_total:
                    son_group_num = int(son_group)
                    son_group_num += 1
                    son_group = str(son_group_num)
                father_group_ob.trans_son_total_num[son_group] = [-1, 2, 0]
                father_group_ob.trans_son_total[son_group] = {son_group_label: {"0": {"0": [son_group_label]}}}
                father_group_ob.trans_num[son_group] = {son_group_label: {"0": {"0": [1, 2]}}}
                father_group_ob.trans_replace_num[son_group] = {son_group_label: {"0": [1, 2]}}
                father_group_ob.trans_replace_max[son_group] = {son_group_label: 1}
                father_group_ob.trans_start_num[son_group] = {son_group_label: -1}
                father_group_ob.trans_total_num[son_group] = son_group_ob.feature_num
                father_group_ob.trans_activate_num[son_group] = {son_group_label: son_group_ob.feature_num}
                son_group_ob.trans_father_total[father_group] = {son_group: {son_group_label: {"0": "0"}}}
                length_str_dict[father_group] = son_group
            else:
                son_group = length_str_dict[father_group]
                start_group_name = son_group_label
                son_group_ob = start_group_dict[son_group_label]
                while start_group_name in father_group_ob.trans_son_total[son_group]:
                    start_group_num = int(start_group_name)
                    start_group_num += 1
                    start_group_name = str(start_group_num)
                father_group_ob.trans_son_total[son_group][start_group_name] = {"0": {"0": [son_group_label]}}
                father_group_ob.trans_num[son_group][start_group_name] = {"0": {"0": [1, 2]}}
                father_group_ob.trans_replace_num[son_group][start_group_name] = {"0": [1, 2]}
                father_group_ob.trans_replace_max[son_group][start_group_name] = 1
                father_group_ob.trans_start_num[son_group][start_group_name] = -1
                father_group_ob.trans_total_num[son_group] += son_group_ob.feature_num
                father_group_ob.trans_activate_num[son_group][start_group_name] = son_group_ob.feature_num
                son_group_ob.trans_father_total[father_group] = {son_group: {start_group_name: {"0": "0"}}}


def create_mini_group(generator_mini_group, father_mini):
    if len(initial_variable.recycle_mini_list) == 0:
        mini_group_label = str(len(initial_variable.mini_group_dict))
        while mini_group_label in initial_variable.mini_group_dict:
            mini_group_num = int(mini_group_label)
            mini_group_num += 1
            mini_group_label = str(mini_group_num)
        mini_ob = generator_mini_group(mini_group_label, father_mini)
        mini_ob.correct_num = initial_variable.son_mini_mid_correct_num
        initial_variable.mini_group_dict[mini_group_label] = mini_ob
        for father_mini_group in father_mini:
            father_mini_ob = initial_variable.mini_group_dict[father_mini_group]
            if mini_group_label not in father_mini_ob.son_mini:
                if len(father_mini_ob.son_mini) < 6:
                    father_mini_ob.son_mini.append(mini_group_label)
                else:
                    father_mini_ob.son_mini.insert(5, mini_group_label)
    else:
        mini_group_label = initial_variable.recycle_mini_list.pop(0)
        mini_ob = initial_variable.mini_group_dict[mini_group_label]
        mini_ob.correct_num = initial_variable.son_mini_mid_correct_num
        mini_ob.father_mini = father_mini
        for father_mini_group in father_mini:
            father_mini_ob = initial_variable.mini_group_dict[father_mini_group]
            if mini_group_label not in father_mini_ob.son_mini:
                if len(father_mini_ob.son_mini) < 6:
                    father_mini_ob.son_mini.append(mini_group_label)
                else:
                    father_mini_ob.son_mini.insert(5, mini_group_label)
    return mini_group_label


def del_trans_father(son_group_label, trans_father):
    son_group_ob = start_group_dict[son_group_label]
    if trans_father in son_group_ob.trans_father_total:
        father_group_ob = start_group_dict[trans_father]
        for son_group in son_group_ob.trans_father_total[trans_father]:
            for start_group_key in son_group_ob.trans_father_total[trans_father][son_group]:
                for replace_key in son_group_ob.trans_father_total[trans_father][son_group][start_group_key]:
                    group_key = son_group_ob.trans_father_total[trans_father][son_group][start_group_key][replace_key]
                    if son_group_label in father_group_ob.trans_son_total[son_group][start_group_key][replace_key][group_key]:
                        father_group_ob.trans_son_total[son_group][start_group_key][replace_key][group_key].remove(son_group_label)
                    if len(father_group_ob.trans_son_total[son_group][start_group_key][replace_key][group_key]) == 0:
                        father_group_ob.trans_son_total[son_group][start_group_key][replace_key].pop(group_key)
                        if group_key in father_group_ob.trans_num[son_group][start_group_key][replace_key]:
                            father_group_ob.trans_num[son_group][start_group_key][replace_key].pop(group_key)
                    if len(father_group_ob.trans_son_total[son_group][start_group_key][replace_key]) == 0:
                        father_group_ob.trans_son_total[son_group][start_group_key].pop(replace_key)
                        if replace_key in father_group_ob.trans_num[son_group][start_group_key]:
                            father_group_ob.trans_num[son_group][start_group_key].pop(replace_key)
                        if replace_key in father_group_ob.trans_replace_num[son_group][start_group_key]:
                            father_group_ob.trans_replace_num[son_group][start_group_key].pop(replace_key)
        son_group_ob.trans_father_total.pop(trans_father)
    if len(son_group_ob.trans_father_total) == 0 and son_group_ob.delete == 0 \
            and len(son_group_ob.father_mini_dict) == 0:
        del_group(son_group_label)
        # print("delete start group is: " + son_group_label)


def del_group(son_group_label):
    """ delete part(group_label)"""
    # print("delete group_label is: " + son_group_label)
    group_ob = start_group_dict[son_group_label]
    if group_ob.delete == 1:
        return
    del_list = [son_group_label]
    while len(del_list) > 0:
        delete_group_label = del_list.pop()
        del_group_ob = start_group_dict[delete_group_label]
        del_group_ob.delete = 1
        del_group_ob.delete_num += 1
        initial_variable.recycle_start_group_list.append(delete_group_label)
        for father_group in del_group_ob.trans_father_total:
            father_group_ob = start_group_dict[father_group]
            for son_group in del_group_ob.trans_father_total[father_group]:
                for start_group_key in del_group_ob.trans_father_total[father_group][son_group]:
                    for replace_key in del_group_ob.trans_father_total[father_group][son_group][start_group_key]:
                        group_key = del_group_ob.trans_father_total[father_group][son_group][start_group_key][replace_key]
                        if delete_group_label in father_group_ob.trans_son_total[son_group][start_group_key][replace_key][group_key]:
                            father_group_ob.trans_son_total[son_group][start_group_key][replace_key][group_key].remove(delete_group_label)
                        if len(father_group_ob.trans_son_total[son_group][start_group_key][replace_key][group_key]) == 0:
                            father_group_ob.trans_son_total[son_group][start_group_key][replace_key].pop(group_key)
                            if group_key in father_group_ob.trans_num[son_group][start_group_key][replace_key]:
                                father_group_ob.trans_num[son_group][start_group_key][replace_key].pop(group_key)

        for son_group1 in del_group_ob.trans_son_total:
            for start_group_label in del_group_ob.trans_son_total[son_group1]:
                for replace_key in del_group_ob.trans_son_total[son_group1][start_group_label]:
                    for group_key in del_group_ob.trans_son_total[son_group1][start_group_label][replace_key]:
                        for group_label in del_group_ob.trans_son_total[son_group1][start_group_label][replace_key][group_key]:
                            son_group_ob = start_group_dict[group_label]
                            if delete_group_label in son_group_ob.trans_father_total:
                                son_group_ob.trans_father_total.pop(delete_group_label)
                            if len(son_group_ob.trans_father_total) == 0 and son_group_ob.delete == 0 \
                                    and len(son_group_ob.father_mini_dict) == 0 and group_label not in del_list:
                                del_list.append(group_label)
                                # print("delete start group is: " + group_label)

        del_group_ob.delete_num += 1
        del_group_ob.execute = 0
        del_group_ob.success = 0
        del_group_ob.replace_signal = 0
        del_group_ob.result_dict.clear()
        del_group_ob.total_save_dict.clear()
        del_group_ob.activate_dict.clear()
        del_group_ob.image_save_dict.clear()
        del_group_ob.scanning_image_dict.clear()
        del_group_ob.scanning_image_or_dict.clear()
        del_group_ob.result_total.clear()
        del_group_ob.result_dict.clear()
        del_group_ob.activate_dict.clear()
        del_group_ob.replace_save.clear()
        del_group_ob.trans_son_total.clear()
        del_group_ob.trans_son_total_num.clear()
        del_group_ob.trans_num.clear()
        del_group_ob.trans_replace_num.clear()
        del_group_ob.trans_replace_max.clear()
        del_group_ob.trans_start_num.clear()
        del_group_ob.trans_total_num.clear()
        del_group_ob.trans_activate_num.clear()
        del_group_ob.allow_trans_son_dict = {}
        del_group_ob.allow_mini_dict = {}
        del_group_ob.trans_father_keep_dict = {}
        del_group_ob.sort_trans_father = {}

    for mini_group in group_ob.father_mini_dict:
        mini_ob = initial_variable.mini_group_dict[mini_group]
        for start_group_key in group_ob.father_mini_dict[mini_group]:
            for replace_key in group_ob.father_mini_dict[mini_group][start_group_key]:
                group_key = group_ob.father_mini_dict[mini_group][start_group_key][replace_key]
                if start_group_key in mini_ob.small_group and replace_key in mini_ob.small_group[start_group_key] \
                        and group_key in mini_ob.small_group[start_group_key][replace_key]:
                    if son_group_label in mini_ob.small_group[start_group_key][replace_key][group_key]:
                        mini_ob.small_group[start_group_key][replace_key][group_key].remove(son_group_label)
                    if len(mini_ob.small_group[start_group_key][replace_key][group_key]) == 0:
                        mini_ob.small_group[start_group_key][replace_key].pop(group_key)
    group_ob.father_mini_dict.clear()
    group_ob.father_mini.clear()


def del_mini(min_temp):
    # print("delete mini group is: " + min_temp)
    mini_ob_temp = initial_variable.mini_group_dict[min_temp]
    if mini_ob_temp.delete == 1:
        return
    mini_ob_temp.delete = 1
    mini_ob_temp.save_update_group.clear()
    for father_mini_group in mini_ob_temp.father_mini:
        father_mini_ob = initial_variable.mini_group_dict[father_mini_group]
        if min_temp in father_mini_ob.son_mini:
            father_mini_ob.son_mini.remove(min_temp)
    initial_variable.recycle_mini_list.append(min_temp)
    for start_group_key in mini_ob_temp.small_group:
        for replace_key in mini_ob_temp.small_group[start_group_key]:
            for group_key in mini_ob_temp.small_group[start_group_key][replace_key]:
                for delete_group_label in mini_ob_temp.small_group[start_group_key][replace_key][group_key]:
                    # initial_variable.recycle_group_list.append(delete_group_label)
                    del_group_ob = start_group_dict[delete_group_label]
                    if min_temp in del_group_ob.father_mini:
                        del_group_ob.father_mini.remove(min_temp)
                    if min_temp in del_group_ob.father_mini_dict:
                        del_group_ob.father_mini_dict.pop(min_temp)
                    if len(del_group_ob.trans_father_total) == 0 and len(del_group_ob.father_mini_dict) == 0 and \
                            del_group_ob.delete == 0:
                        del_group(delete_group_label)
    mini_ob_temp.small_group.clear()
    mini_ob_temp.small_group_num.clear()
    mini_ob_temp.small_start_num.clear()
    mini_ob_temp.activate_group.clear()
    mini_ob_temp.son_mini_correct.clear()
    mini_ob_temp.activate_small_group.clear()
    mini_ob_temp.small_total_num = 0
    mini_ob_temp.execute = 0
    mini_ob_temp.success = 0