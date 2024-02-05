# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
# import tensorflow as tf
# import mquat as mq
#
#
# def my_create_quanziter_func(parent_name, index):
#     return mq.FlexPointQuantizer(parent_name + "_" + str(index), 4, std_keep_factor=0.0,
#                                  debug=True)
#
# myquant = mq.PerChannelQuantizer("myquant_per_channel", create_quanziter_func=my_create_quanziter_func,splits=2,split_axis=1, debug=True)
# data = tf.constant([[-0.25, -1], [-0.13, 0], [0.01, 2.3], [0.19, 5.6]], shape=[4, 2], dtype=tf.float32)
# data2 = tf.constant(np.array([[400.0, -201.0], [-801.0, 401.0]]))
# data2 = tf.constant(np.array([[400.0, -201.0], [-801.0, 401.0]]))
# #data2 = np.transpose(data2, axes=[1,0])
#
# #data2 = np.array([[400.0, 801.0], [201.0, 401.0]])
# cond_num = tf.cast(tf.numpy_function(lambda x: np.linalg.cond(x), [data2], [tf.float32]), dtype=tf.float32)
# print(cond_num)
# exit(0)


# flex_qat_mse = []
# flex_qat_bits = []
# flex_ptq_mse = []
# flex_ptq_bits = []
#
# for file in os.listdir("mats_flex_qat"):
#     if "_mse.csv" not in file:
#         continue
#     filepath = "mats_flex_qat/" + file
#     desc = file[file.index("(")+1:]
#     desc = desc[:desc.index(")")]
#     desc_num = desc[4:]
#     mse = np.loadtxt(filepath)
#     print(desc, mse, desc_num)
#     flex_qat_mse.append(float(mse))
#     flex_qat_bits.append(int(desc_num))
#
# for file in os.listdir("mats_flex_ptq"):
#     if "_mse.csv" not in file:
#         continue
#     filepath = "mats_flex_ptq/" + file
#     desc = file[file.index("(")+1:]
#     desc = desc[:desc.index(")")]
#     desc_num = desc[4:]
#     mse = np.loadtxt(filepath)
#     print(desc, mse, desc_num)
#     flex_ptq_mse.append(float(mse))
#     flex_ptq_bits.append(int(desc_num))
#
#
# plt.scatter(flex_qat_bits, np.clip(flex_qat_mse, 0, mse_max*0.99), marker="o", color="green", edgecolors="blue", label="qat")
# plt.scatter(flex_ptq_bits, np.clip(flex_ptq_mse, 0, mse_max*0.99), marker="^", color="green", edgecolors="orange", label="ptq")
# plt.ylim([0, mse_max])
# plt.legend()
# plt.xlabel("bits/adders for weights")
# plt.ylabel("val mse loss")
# plt.xticks(flex_qat_bits)
# plt.title("float_to_flex")
# plt.savefig("figs/float_to_flex.png")
# plt.show()
# plt.scatter(flex_qat_bits, np.clip(flex_qat_mse,0,1.95), marker="X", color="red", label="qat")
# plt.scatter(flex_ptq_bits, np.clip(flex_ptq_mse,0,1.95), marker=".", color="black", label="ptq")
# plt.ylim([0,2])
# plt.legend()
# plt.xlabel("bits for all weights")
# plt.ylabel("val mse loss")
# plt.xticks(flex_qat_bits)
# plt.show()


# flex_qat_mse = []
# flex_qat_bits = []
# flex_ptq_mse = []
# flex_ptq_bits = []
#
# # for file in os.listdir("mats"):
# #     if "_mse.csv" not in file:
# #         continue
# #     filepath = "mats_flex_qat/" + file
# #     desc = file[file.index("(")+1:]
# #     desc = desc[:desc.index(")")]
# #     desc_num = desc[4:]
# #     mse = np.loadtxt(filepath)
# #     print(desc, mse, desc_num)
# #     flex_qat_mse.append(float(mse))
# #     flex_qat_bits.append(int(desc_num))
#
# for file in os.listdir("mats"):
#     if "_mse.csv" not in file:
#         continue
#     filepath = "mats/" + file
#     desc = file[file.index("(")+1:]
#     desc = desc[:desc.index(")")]
#     desc_num = desc[0:2]
#     mse = np.loadtxt(filepath)
#     print(desc, mse, desc_num)
#     flex_ptq_mse.append(float(mse))
#     flex_ptq_bits.append(int(desc_num))
#
# plt.scatter(flex_qat_bits, np.clip(flex_qat_mse,0,1.95), marker="X", color="red", label="qat")
# plt.scatter(flex_ptq_bits, np.clip(flex_ptq_mse,0,1.95), marker=".", color="black", label="ptq")
# plt.ylim([0,2])
# plt.legend()
# plt.xlabel("bits for all weights")
# plt.ylabel("val mse loss")
# plt.xticks(flex_ptq_bits)
# plt.show()
# # exit(0)

# strategies = ["5flex_to_add", "8flex_to_add", "11flex_to_add", "float_to_add", "float64_to_add64", "flex64",
#               "ds_float_to_flex", "ds_float_to_add", "float_to_channel_add", "ds_float_to_channel_add", "float_to_channel_flex",
#               "float_to_channel_flex3"]
#strategies = ["float_to_channel_flex3"]

strategies = ["float_to_channel_flex", "16flex_to_channel_add", "12flex_to_channel_add", "8flex_to_channel_add", "5flex_to_channel_add",
                  "6flex_to_channel_add", "16flex_to_add", "12flex_to_add", "8flex_to_add", "5flex_to_add", "6flex_to_add"]
strat_nums = [16,12,8,6,5]
strat_nums = sorted(strat_nums)
dnn_strat_meth_dict = {}

dnns = ["3", "5", "10"] # , "5", "10"
mse_max = 8.5

main_folder = "C:\\Users\\fdai0217\\Desktop\\ARCH ergs neu\\"

for dnn in dnns:
    if dnn not in dnn_strat_meth_dict:
        dnn_strat_meth_dict[dnn] = {}
    for strat in strategies:
        if strat not in dnn_strat_meth_dict[dnn]:
            dnn_strat_meth_dict[dnn][strat] = {}

        for method in ["ptq", "qat"]:
            if method not in dnn_strat_meth_dict[dnn][strat]:
                dnn_strat_meth_dict[dnn][strat][method] = {}
            foldername = main_folder + "dnn("+dnn+")\\mats_dnn(" + dnn + ")_"+strat+"_" + method
            mses = []
            bits = []
            for file in os.listdir(foldername):
                if "_mse.csv" not in file:
                    continue
                filepath = foldername + "\\" + file
                desc = file[file.index("(")+1:]
                desc = desc[:desc.index(")")] # All between the "network_ptq_q(BRACKETS HERE!!!!)_mse.csv" => "0 Adder_12 Flex"
                desc_num = desc[0:2]
                mse = np.loadtxt(filepath)
                # print(strat, method, desc, mse, desc_num)
                mses.append(float(mse))
                bits.append(int(desc_num))
            zipped_lists = zip(bits, mses)
            sorted_pairs = sorted(zipped_lists)
            tuples = zip(*sorted_pairs)
            bits, mses = [list(tuple) for tuple in tuples]
            dnn_strat_meth_dict[dnn][strat][method] = [mses, bits]

for dnn in dnns:
    flex_qat_mse, flex_qat_bits = dnn_strat_meth_dict[dnn]["float_to_channel_flex"]["qat"]
    flex_ptq_mse, flex_ptq_bits = dnn_strat_meth_dict[dnn]["float_to_channel_flex"]["ptq"]
    plt.scatter(flex_qat_bits, np.clip(flex_qat_mse,0,mse_max*0.99), marker="o", color="green", edgecolors="blue", label="qat")
    plt.scatter(flex_ptq_bits, np.clip(flex_ptq_mse,0,mse_max*0.99), marker="^", color="green", edgecolors="orange", label="ptq")
    plt.ylim([-0.1, mse_max])
    plt.legend()
    plt.xlabel("bits for weights and biases")
    plt.ylabel("validation mse loss")
    plt.xticks(flex_ptq_bits)
    plt.title("dnn("+dnn+")_mse")
    #plt.savefig(foldername+"\\"+strat+"_mse.png")
    plt.savefig("figs\\dnn("+dnn+")_Fixedpoint_mse.png")
    plt.show()

for dnn in dnns:
    for method in ["ptq", "qat"]:
        for i, num in enumerate(strat_nums):
            all_add_channel_mse = dnn_strat_meth_dict[dnn][str(num) + "flex_to_channel_add"][method][0]
            all_add_layer_mse = dnn_strat_meth_dict[dnn][str(num) + "flex_to_add"][method][0]
            add0_channel_mse = [all_add_channel_mse[0]]
            add1_channel_mse = [all_add_channel_mse[1]]
            add2_channel_mse = [all_add_channel_mse[2]]
            plt.scatter(np.ones_like(add0_channel_mse) * i, np.clip(add0_channel_mse,0,mse_max*0.99), marker="o", color="green", edgecolors="blue", label="0 Add p. c.")
            plt.scatter(np.ones_like(add1_channel_mse) * i, np.clip(add1_channel_mse,0,mse_max*0.99), marker="^", color="green", edgecolors="blue", label="1 Add p. c.")
            plt.scatter(np.ones_like(add2_channel_mse) * i, np.clip(add2_channel_mse,0,mse_max*0.99), marker="x", color="green", edgecolors="blue", label="2 Add p. c.")
            plt.scatter(np.ones_like(all_add_layer_mse) * i, np.clip(all_add_layer_mse,0,mse_max*0.99), marker="*", color="green", edgecolors="orange", label="2 Add p. l.")
            if i == 0:
                plt.legend()
        plt.ylim([-0.1, mse_max])
        plt.xlabel("word length for weights and biases")
        plt.ylabel("validation mse loss")
        plt.xticks(range(len(strat_nums)), strat_nums)
        name = "dnn("+dnn+")_"+method+"_mse"
        plt.title(name)
        #plt.savefig(foldername+"\\"+strat+"_mse.png")
        plt.savefig("figs\\"+name+".png")
        plt.show()


