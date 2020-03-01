# *** Writen by Shiquan Mei ***

import numpy as np
import struct
import matplotlib.pyplot as plt
from initial_variable import *
from model_list import *
from part import *
from program_list import recognition
from test_function import recognition_test
import random


train_images_idx3_ubyte_file = 'E:/MNIST/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = 'E:/MNIST/train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = 'E:/MNIST/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = 'E:/MNIST/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((10000, num_rows, num_cols))
    for i in range(10000):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(10000)
    for i in range(10000):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def dataload():
    train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)
    return train_images, train_labels, test_images, test_labels


def construct_recognition():
    train_images, train_labels, test_images, test_labels = dataload()
    filePathGroup = "E:\\DcomNet\\result\\refine_result\\re_refine.txt"
    group_label_list = [9]
    for global_label in group_label_list:
        with open(filePathGroup, 'a') as file_object:
            file_object.write("******** global_label: " + str(global_label) + " ********" + "\n")
        start_group_dict.clear()
        group_dict.clear()
        initial_variable.mini_group_dict.clear()
        initial_variable.recycle_start_group_list.clear()
        initial_variable.recycle_group_list.clear()
        initial_variable.recycle_mini_list.clear()

        first_time = 0
        first_mini_ob = generator_mini_group('0', [])
        initial_variable.mini_group_dict['0'] = first_mini_ob
        initial_variable.activate_length = 0
        initial_variable.save_num = 400
        for number_input_i in range(0, 10000):
            label = train_labels[number_input_i]
            initial_variable.image_number = number_input_i
            initial_variable.number_input = number_input_i
            initial_variable.activate_length += 1
            if label == global_label or number_input_i % 20 == 0:
                image_mid = train_images[number_input_i] - 30
                image_test = image_mid + abs(image_mid)
                image_test_first = np.ceil(image_test / (abs(image_test) + 0.001))
                image_test_list, image_test_total, image_name_dict, num, test_num \
                    = scanning_input(image_test_first)
                if len(image_test_list) == 0:
                    continue
                if first_time == 0 and label == global_label:
                    print("number_input is: %d" % number_input_i + ", execute create group, label is: " + str(global_label))
                    create_group(['0'], image_test_list)
                    first_time = 1
                elif first_time == 1:
                    if initial_variable.activate_length >= initial_variable.save_num:
                        if initial_variable.activate_length > 20:
                            construct_test_recognition(global_label, test_images, test_labels, '0', 0.15)
                        if initial_variable.save_num <= 3300:
                            if initial_variable.save_num < 2500:
                                initial_variable.save_num += 400
                            else:
                                initial_variable.save_num += 500
                        else:
                            break
                    print("******** prepare execute next time **********")
                    print("number_input is: %d" % number_input_i)
                    initial_variable.number_input = number_input_i
                    success_signal = recognition(image_test_list, image_test_total, image_name_dict, num,
                                                 ['0'], global_label, label)
            if initial_variable.save_num > 3301:
                break
        with open(filePathGroup, 'a') as file_object:
            file_object.write("\n")
            file_object.write("\n")


def construct_test_recognition(global_label, test_images, test_labels, first_mini_label, success_ratio):
    total_test_num = 0
    recognition_test_success = 0
    total_test_error_num = 0
    recognition_test_error_success = 0
    max_comparator_num = 0  # count peak number of comparison
    max_multiplier_num = 0  # count peak number of multiplication
    max_adder_num = 0  # count peak number of addition
    initial_variable.total_comparator_num = 0
    initial_variable.total_adder_num = 0
    initial_variable.total_multiplier_num = 0
    initial_variable.total_num = 0
    for number_input_i in range(0, 2000):
        label = test_labels[number_input_i]
        if label == global_label or number_input_i % 10 == 0:
            image_mid = test_images[number_input_i] - 30
            image_test = image_mid + abs(image_mid)
            image_test_first = np.ceil(image_test / (abs(image_test) + 0.001))
            initial_variable.copy_image = image_test_first
            image_test_list, image_test_total, image_name_dict, num, test_num \
                = scanning_input(image_test_first)
            success_signal = recognition_test(image_test_total, image_name_dict, num,
                                              [first_mini_label], success_ratio)
            if label == global_label:
                total_test_num += 1
                if success_signal is True:
                    recognition_test_success += 1
            else:
                total_test_error_num += 1
                if success_signal is False:
                    recognition_test_error_success += 1
            if initial_variable.test_signal == 1:
                # We can set initial_variable.test_signal to count the number of comparisons, multiplications,
                # and additions
                initial_variable.total_comparator_num += initial_variable.comparator_num
                initial_variable.total_adder_num += initial_variable.adder_num
                initial_variable.total_multiplier_num += initial_variable.multiplier_num
                initial_variable.total_num += 1
                if initial_variable.comparator_num > max_comparator_num:
                    max_comparator_num = initial_variable.comparator_num
                if initial_variable.adder_num > max_adder_num:
                    max_adder_num = initial_variable.adder_num
                if initial_variable.multiplier_num > max_multiplier_num:
                    max_multiplier_num = initial_variable.multiplier_num
                initial_variable.comparator_num = 0
                initial_variable.adder_num = 0
                initial_variable.multiplier_num = 0
    print("recognition_test_success is %d, total_num is: %d" % (recognition_test_success, total_test_num))
    print("recognition test success ratio is: %f" % (recognition_test_success / total_test_num))
    print("recognition_test_error_success is %d, total_error_num is: %d" %
          (recognition_test_error_success, total_test_error_num))
    print("recognition test error success ratio is: %f" % (recognition_test_error_success / total_test_error_num))
    filePathGroup = "E:\\DcomNet\\result\\refine_result\\re_refine.txt"
    with open(filePathGroup, 'a') as file_object:
        first_mini_ob = initial_variable.mini_group_dict['0']
        not_update = 0
        file_object.write("global_label: " + str(global_label) + "\n")
        for son_mini in first_mini_ob.son_mini:
            son_mini_ob = initial_variable.mini_group_dict[son_mini]
            if not_update < 3:
                file_object.write("son_mini: " + son_mini +
                                  ", small_group: " + str(son_mini_ob.small_group) + "\n")
                file_object.write("small_group: " + str(son_mini_ob.small_group_num) + "\n")
                not_update += 1

        file_object.write("activate_length: " + str(initial_variable.activate_length) + "\n")
        file_object.write("recognition_test_success is: " + str(recognition_test_success)
                          + ", total_num is: " + str(total_test_num) + "\n")
        file_object.write("recognition test success ratio is: " + str(recognition_test_success / total_test_num) + "\n")
        file_object.write("recognition_test_error_success is: " + str(recognition_test_error_success)
                          + ", total_error_num is: " + str(total_test_error_num) + "\n")
        file_object.write("recognition test error success ratio is: " + str(recognition_test_error_success / total_test_error_num) + "\n")
        file_object.write("\n")
        file_object.write("\n")



if __name__ == '__main__':
    construct_recognition()
