import numpy as np
import os
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches
import json


mse_max = 0.5
widht=0.2
fp32_mse=0.12194367498159409
dnn = "3"

def toFilter(tab, color):
    result = []
    for n in tab:
        if n < mse_max:
            result.append(color)
        else:
            #result.append("white")#"white")
            result.append(color)  # "white")
    return result


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


def fixstuff():
    fig = plt.gcf()
    size = np.array([16, 8]) * 0.7
    fig.set_size_inches(size[0], size[1])
    plt.subplots_adjust(left=0.1, right=0.99, top=0.97, bottom=0.15)


def get_path(model_name, desc, train_type, quant_depth="", weight_bits="", bias_bits="", adder=""):
    SAVE_PREFIX = {-1: lambda: None,
                   "float": lambda: f"{model_name}/float/{desc}/" ,
                   "fixed": lambda: f"{model_name}/{quant_depth}_fixed/{weight_bits}w_{bias_bits}b_{None}a/{desc}/" ,
                   "adder": lambda: f"{model_name}/{quant_depth}_adder/{adder}_adder_{weight_bits}w_{bias_bits}b_{None}a/{desc}/" }

    model_saved_path = SAVE_PREFIX[train_type]()
    return model_saved_path

def read_val_mse(model_name, desc, train_type, quant_depth="", weight_bits="", bias_bits="", adder=""):
    path = get_path(model_name, desc, train_type, quant_depth, weight_bits, bias_bits, adder)
    with open(path + "results.txt", "r") as f:
        data = json.loads(f.read().replace("'", "\""))
    return data["val_loss"]


val_mse_map = {}

for quant_depth in ["channel-wise", "layer-wise"]:
    if quant_depth not in val_mse_map:
        val_mse_map[quant_depth] = {}
    for train_type in ["fixed"]:
        if train_type not in val_mse_map[quant_depth]:
            val_mse_map[quant_depth][train_type] = {}
        for bits in range(3, 17):
            # if bits not in val_mse_map[quant_depth][train_type]:
            #     val_mse_map[quant_depth][train_type][bits] = {}
            val_mse_map[quant_depth][train_type][bits] = read_val_mse("m3", "default", train_type, quant_depth, bits, bits, None)
            print(train_type, quant_depth, bits, val_mse_map[quant_depth][train_type][bits])

for quant_depth in ["channel-wise", "layer-wise"]:
    if quant_depth not in val_mse_map:
        val_mse_map[quant_depth] = {}
    for train_type in ["adder"]:
        if train_type not in val_mse_map[quant_depth]:
            val_mse_map[quant_depth][train_type] = {}
        for bits in range(3, 17):
            for adder in [10, 11, 12]:
                if bits not in val_mse_map[quant_depth][train_type]:
                    val_mse_map[quant_depth][train_type][bits] = {}
                val_mse_map[quant_depth][train_type][bits][adder] = read_val_mse("m3", "default", train_type, quant_depth, bits, bits, adder)
            print(train_type, quant_depth, bits, val_mse_map[quant_depth][train_type][bits])




quant_depth = "layer-wise"
train_type = "fixed"


# for bits in range(3, 17):
# # flex_qat_mse, flex_qat_bits, flex_qat_stable = dnn_strat_meth_dict[dnn]["float_to_channel_flex"]["qat"]
# # flex_ptq_mse, flex_ptq_bits, flex_ptq_stable = dnn_strat_meth_dict[dnn]["float_to_channel_flex"]["ptq"]
# # print("flex_qat_mse", [round(flex_qat_mse, 4)  ])
# # print("flex_qat_bits", flex_qat_bits)
# plt.bar(np.array(flex_qat_bits) - widht*0.6, np.clip(flex_qat_mse,0,mse_max*0.99), width=widht, color=toFilter(flex_qat_mse,"#ffd700"), edgecolor="red", linewidth=toColor(flex_qat_stable, flex_qat_mse), label="qat")# marker="o",
# plt.bar(np.array(flex_ptq_bits) + widht*0.6, np.clip(flex_ptq_mse,0,mse_max*0.99), width=widht, color=toFilter(flex_ptq_mse,"#0066cc"), edgecolor="red", linewidth=toColor(flex_ptq_stable, flex_ptq_mse), label="ptq")# marker="^",
# for i, mse in enumerate(flex_ptq_mse):
#     if mse >= mse_max:
#         plt.arrow(flex_ptq_bits[i] + widht*0.6, mse_max-0.04, 0, 0.03, length_includes_head=True,
#               head_width=0.15, head_length=0.01, facecolor="black", edgecolor="black")
# for i, mse in enumerate(flex_qat_mse):
#     if mse >= mse_max:
#         plt.arrow(flex_qat_bits[i] - widht*0.6, mse_max-0.04, 0, 0.03, length_includes_head=True,
#               head_width=0.15, head_length=0.01, facecolor="black", edgecolor="black")
# #plt.scatter()
# #plt.scatter(flex_ptq_bits, np.clip(flex_ptq_mse,0,mse_max*0.99), marker="^", color=toColor(flex_ptq_stable), label="ptq")
# plt.ylim([0, mse_max])
# qat_legend = mpatches.Patch(color='#0066cc', label='PTQ')
# ptq_legend = mpatches.Patch(color='#ffd700', label='QAT')
# fp32_legend = plt.hlines([fp32_mse], -1000, 1000, colors=["grey"], linestyles="solid", label="FP32")
# leg = plt.legend(handles=[qat_legend, ptq_legend, fp32_legend],frameon=False)
# leg.get_frame().set_linewidth(0.0)
# plt.xlabel("word length for weights and biases")
# plt.ylabel("validation mse loss")
# plt.hlines([fp32_mse], -1000, 1000, colors=["grey"], linestyles="dotted")
#
# xticks = flex_ptq_bits
# plt.xticks(xticks)
# plt.xlim([np.min(xticks) - 1, np.max(xticks) + 1])
#
# #plt.title("dnn("+dnn+")_mse")
# fixstuff()
# #plt.savefig(foldername+"\\"+strat+"_mse.png")
# plt.savefig("figs\\dnn("+dnn+")_Fixedpoint_mse.png")
# plt.show()