# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

# import random
# import csv
# all_valls = []
# with open('ds/val.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in spamreader:
#         print(row[0])
#         all_valls.append(row[0])
#
# x = [1,2,3,4,5,6]
# random.shuffle(all_valls)
# with open('ds/val20k.csv', 'a') as the_file:
#     for i, line in enumerate(all_valls):
#         the_file.write(line + '\n')
#         if i == 20000:
#             exit(0)
# exit(0)

import matplotlib.lines as mlines
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



import matplotlib.patches as mpatches

strategies = ["float_to_channel_flex"]
strat_nums = [4,5,6,7,8,9,12,16]
strat_nums = sorted(strat_nums)
for num in strat_nums:
    strategies.append(str(num) + "flex_to_channel_add")
    strategies.append(str(num) + "flex_to_add")
dnn_strat_meth_dict = {}

dnns = ["3", "5"] # , "5", "10"
mse_max = 0.5
widht=0.2
plt.rcParams.update({'font.size': 22})
fp32_mse=0.12194367498159409
main_folder = "C:\\Users\\fdai0217\\Desktop\\ARCH ergs neu\\"

def fixstuff():
    fig = plt.gcf()
    size = np.array([16, 8]) * 0.7
    fig.set_size_inches(size[0], size[1])
    plt.subplots_adjust(left=0.1, right=0.99, top=0.97, bottom=0.15)

def toColor(tab, mse_tab):
    result = []
    for n, mse in zip(tab, mse_tab):
        if mse >= mse_max:
            result.append(0.0) # 0.0
        elif n == 0:
            result.append(3.2)
        else:
            result.append(0.0)
    return result

def toFilter(tab, color):
    result = []
    for n in tab:
        if n < mse_max:
            result.append(color)
        else:
            #result.append("white")#"white")
            result.append(color)  # "white")
    return result

def toFilter2(tab, color):
    result = []
    for n in tab:
        if n < mse_max:
            result.append(color)
        else:
            result.append("white")#"white")
    return result

# def toFilterAlpha(tab):
#     result = []
#     for n in tab:
#         if n < mse_max:
#             result.append(1.0)
#         else:
#             result.append(0.5)
#     return result


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
                desc = desc[:desc.index(")")] # All between the "network_ptq_q(BRACKETS HERE!!!!)_mse.csv" => "Cost-0er_12 Flex"
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

dnn_strat_meth_dict["3"]["float_to_channel_flex"]["ptq"].append([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
dnn_strat_meth_dict["3"]["float_to_channel_flex"]["qat"].append([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
dnn_strat_meth_dict["5"]["float_to_channel_flex"]["ptq"].append([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
dnn_strat_meth_dict["5"]["float_to_channel_flex"]["qat"].append([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# dnn_strat_meth_dict["10"]["float_to_channel_flex"]["ptq"].append([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# dnn_strat_meth_dict["10"]["float_to_channel_flex"]["qat"].append([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

for dnn in dnns:
    flex_qat_mse, flex_qat_bits, flex_qat_stable = dnn_strat_meth_dict[dnn]["float_to_channel_flex"]["qat"]
    flex_ptq_mse, flex_ptq_bits, flex_ptq_stable = dnn_strat_meth_dict[dnn]["float_to_channel_flex"]["ptq"]
    # print("flex_qat_mse", [round(flex_qat_mse, 4)  ])
    # print("flex_qat_bits", flex_qat_bits)
    plt.bar(np.array(flex_qat_bits) - widht*0.6, np.clip(flex_qat_mse,0,mse_max*0.99), width=widht, color=toFilter(flex_qat_mse,"#ffd700"), edgecolor="red", linewidth=toColor(flex_qat_stable, flex_qat_mse), label="qat")# marker="o",
    plt.bar(np.array(flex_ptq_bits) + widht*0.6, np.clip(flex_ptq_mse,0,mse_max*0.99), width=widht, color=toFilter(flex_ptq_mse,"#0066cc"), edgecolor="red", linewidth=toColor(flex_ptq_stable, flex_ptq_mse), label="ptq")# marker="^",
    for i, mse in enumerate(flex_ptq_mse):
        if mse >= mse_max:
            plt.arrow(flex_ptq_bits[i] + widht*0.6, mse_max-0.04, 0, 0.03, length_includes_head=True,
                  head_width=0.15, head_length=0.01, facecolor="black", edgecolor="black")
    for i, mse in enumerate(flex_qat_mse):
        if mse >= mse_max:
            plt.arrow(flex_qat_bits[i] - widht*0.6, mse_max-0.04, 0, 0.03, length_includes_head=True,
                  head_width=0.15, head_length=0.01, facecolor="black", edgecolor="black")
    #plt.scatter()
    #plt.scatter(flex_ptq_bits, np.clip(flex_ptq_mse,0,mse_max*0.99), marker="^", color=toColor(flex_ptq_stable), label="ptq")
    plt.ylim([0, mse_max])
    qat_legend = mpatches.Patch(color='#0066cc', label='PTQ')
    ptq_legend = mpatches.Patch(color='#ffd700', label='QAT')
    fp32_legend = plt.hlines([fp32_mse], -1000, 1000, colors=["grey"], linestyles="solid", label="FP32")
    leg = plt.legend(handles=[qat_legend, ptq_legend, fp32_legend],frameon=False)
    leg.get_frame().set_linewidth(0.0)
    plt.xlabel("word length for weights and biases")
    plt.ylabel("validation mse loss")
    plt.hlines([fp32_mse], -1000, 1000, colors=["grey"], linestyles="dotted")

    xticks = flex_ptq_bits
    plt.xticks(xticks)
    plt.xlim([np.min(xticks) - 1, np.max(xticks) + 1])

    #plt.title("dnn("+dnn+")_mse")
    fixstuff()
    #plt.savefig(foldername+"\\"+strat+"_mse.png")
    plt.savefig("figs\\dnn("+dnn+")_Fixedpoint_mse.png")
    plt.show()

exit(0)

stable_map = {}
for dnn in dnns:
    stable_map[dnn] = {}
    for method in ["ptq", "qat"]:
        stable_map[dnn][method] = []
#stable_dnn3_qat_add_map
stable_map["3"]["qat"]=[
[0, 0, 0, 0, 0, 0],#4
[1, 0, 0, 0 ,1, 1],#5
[1, 1, 1, 1, 1, 1],#6
[1, 1, 1, 1, 1, 1],#7
[1, 1, 1, 1, 1, 1],#8
[1, 1, 0, 1, 1, 1],#9
[1, 1, 1, 1, 1, 1],#12
[1, 1, 1, 1, 1, 1],] #16

stable_map["3"]["ptq"] = [
[0, 0, 0, 0, 0, 0], #4
[1, 0, 0, 0, 0, 1], #5
[1, 0, 1, 0, 1, 1], #6
[1, 0, 1, 0, 1, 1], #7
[1, 0, 1, 1, 0, 1], #8
[1, 1, 0, 1, 1, 1], #9
[1, 1, 1, 1, 1, 1], #12
[0, 1, 0, 1, 1, 1],] #16

stable_map["5"]["qat"] = [
[1, 0, 0, 0, 0, 0],#4
[1, 1, 0, 0, 0, 1],#5
[1, 1, 1, 1, 1, 1],#6
[0, 1, 1, 1, 1, 1],#7
[1, 1, 1, 1, 1, 1],#8
[1, 1, 1, 1, 1, 1],#9
[1, 1, 1, 1, 1, 1],#12
[1, 1, 1, 1, 1, 1],] #16

stable_map["5"]["ptq"] = [
[0, 0, 1, 0, 0, 0],#4
[0, 0, 1, 0, 0, 1],#5
[1, 1, 1, 1, 0, 1],#6
[1, 1, 1, 0, 1, 1],#7
[1, 1, 1, 0, 1, 1],#8
[1, 1, 1, 1, 1, 1],#9
[1, 1, 1, 1, 1, 1],#12
[1, 1, 1, 1, 1, 1],] #16

widht /= 1
space = 0.5

for dnn in dnns:
    for method in ["ptq", "qat"]:
        for i, num in enumerate(strat_nums):
            all_add_channel_mse = dnn_strat_meth_dict[dnn][str(num) + "flex_to_channel_add"][method][0]
            all_add_layer_mse = dnn_strat_meth_dict[dnn][str(num) + "flex_to_add"][method][0]
            all_add0_layer_mse = [all_add_layer_mse[0]]
            all_add1_layer_mse = [all_add_layer_mse[1]]
            all_add2_layer_mse = [all_add_layer_mse[2]]
            add0_channel_mse = [all_add_channel_mse[0]]
            add1_channel_mse = [all_add_channel_mse[1]]
            add2_channel_mse = [all_add_channel_mse[2]]
            add0_channel_stable = [stable_map[dnn][method][i][0]]
            add1_channel_stable = [stable_map[dnn][method][i][1]]
            add2_channel_stable = [stable_map[dnn][method][i][2]]
            all_add0_layer_stable = [stable_map[dnn][method][i][3]]
            all_add1_layer_stable = [stable_map[dnn][method][i][4]]
            all_add2_layer_stable = [stable_map[dnn][method][i][5]]
            plt.bar(np.ones_like(add0_channel_mse) * i * 2 - widht * 2.4 * 1.5, np.clip(add0_channel_mse, 0, mse_max * 0.99), width=widht, color=toFilter(add0_channel_mse, "blue"), edgecolor="red", linewidth=toColor(add0_channel_stable, add0_channel_mse), label="ptq")  # marker="^",
            plt.bar(np.ones_like(add1_channel_mse) * i * 2 - widht * 2.4 * 0.9, np.clip(add1_channel_mse, 0, mse_max * 0.99), width=widht, color=toFilter(add1_channel_mse, "lime"), edgecolor="red", linewidth=toColor(add1_channel_stable, add1_channel_mse), label="ptq")  # marker="^",
            plt.bar(np.ones_like(add2_channel_mse) * i * 2 - widht * 2.4 * 0.3, np.clip(add2_channel_mse, 0, mse_max * 0.99), width=widht, color=toFilter(add2_channel_mse, "orange"), edgecolor="red", linewidth=toColor(add2_channel_stable, add2_channel_mse), label="ptq")  # marker="^",
            #plt.bar(np.ones_like(all_add0_layer_stable) * i * 2 + widht * 2.4 * 0.3, np.clip(all_add0_layer_mse, 0, mse_max * 0.99), width=widht, color=toFilter(all_add0_layer_mse, "violet"), edgecolor="red", linewidth=toColor(all_add0_layer_stable, all_add0_layer_mse), label="ptq")  # marker="^",
            plt.bar(np.ones_like(all_add1_layer_stable) * i * 2 + widht * 2.4 * 0.3, np.clip(all_add1_layer_mse, 0, mse_max * 0.99), width=widht, color=toFilter(all_add1_layer_mse, "lightgrey"), edgecolor="red", linewidth=toColor(all_add1_layer_stable, all_add1_layer_mse), label="ptq")  # marker="^",
            plt.bar(np.ones_like(all_add2_layer_stable) * i * 2 + widht * 2.4 * 0.9, np.clip(all_add2_layer_mse, 0, mse_max * 0.99), width=widht, color=toFilter(all_add2_layer_mse, "aquamarine"), edgecolor="red", linewidth=toColor(all_add2_layer_stable, all_add2_layer_mse), label="ptq")  # marker="^",
            # plt.scatter(np.ones_like(add0_channel_mse) * i, np.clip(add0_channel_mse,0,mse_max*0.99), marker="o", color=toColor(add0_channel_stable), label="Cost-0 p. c.")
            # plt.scatter(np.ones_like(add1_channel_mse) * i, np.clip(add1_channel_mse,0,mse_max*0.99), marker="^", color=toColor(add1_channel_stable), label="Cost-1 p. c.")
            # plt.scatter(np.ones_like(add2_channel_mse) * i, np.clip(add2_channel_mse,0,mse_max*0.99), marker="x", color=toColor(add2_channel_stable), label="Cost-2 p. c.")
            # plt.scatter(np.ones_like(all_add_layer_mse) * i, np.clip(all_add_layer_mse,0,mse_max*0.99), marker="*", color=toColor(all_add_layer_stable), label="Cost-2 p. l.")
            for mult, mses in [[-widht * 2.4 * 1.5, add0_channel_mse], [-widht * 2.4 * 0.9, add1_channel_mse], [-widht * 2.4 * 0.3, add2_channel_mse],
                               [widht * 2.4 * 0.3, all_add1_layer_mse], [widht * 2.4 * 0.9, all_add2_layer_mse]]:
                for j, mse in enumerate(mses):
                    if mse >= mse_max:
                        print("arrow at ", i * 2 + mult, mse_max - 0.04, add0_channel_mse)
                        plt.arrow(i * 2 + mult, mse_max - 0.04, 0, 0.03, length_includes_head=True,
                                  head_width=0.15, head_length=0.01, facecolor="black", edgecolor="black")

        plt.ylim([0, mse_max])
        add0_legend = mpatches.Patch(color='blue', label='Cost-0 p. c.')
        add1_legend = mpatches.Patch(color='lime', label='Cost-1 p. c.')
        add2_legend = mpatches.Patch(color='orange', label='Cost-2 p. c.')
        #add0l_legend = mpatches.Patch(color='violet', label='Cost-0 p. l.')
        add1l_legend = mpatches.Patch(color='lightgrey', label='Cost-1 p. l.')
        add2l_legend = mpatches.Patch(color='aquamarine', label='Cost-2 p. l.')
        fp32_legend = plt.hlines([fp32_mse], -1000, 1000, colors=["grey"], linestyles="solid", label="FP32")
        plt.legend(handles=[add0_legend, add1_legend, add2_legend, add1l_legend, add2l_legend, fp32_legend])
        plt.xlabel("word length for weights and biases")
        plt.ylabel("validation mse loss")
        xticks = np.array(range(len(strat_nums)))*2
        plt.xticks(xticks, strat_nums)
        plt.xlim([np.min(xticks)-1, np.max(xticks)+1])
        name = "dnn("+dnn+")_"+method+"_mse"
        #plt.title(name)
        #plt.savefig(foldername+"\\"+strat+"_mse.png")
        plt.savefig("figs\\"+name+".png")

        fixstuff()
        plt.savefig("figs\\dnn("+dnn+")_"+method+"_mse.png")
        plt.show()


