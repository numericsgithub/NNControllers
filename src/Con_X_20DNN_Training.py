import os
import sys


sys.path.insert(1, os.path.join(sys.path[0], '../..'))
# import tensorflow as tf
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

from training import QTraining
import mquat as mq  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import csv

import Con_10_20DNN as con_10_20  # noqa: E402
import Con_5_20DNN as con_5_20  # noqa: E402
import Con_3_20DNN as con_3_20  # noqa: E402
dtype = tf.float32



def func_load_inital_model(model, args):
    pass

# -model m3 -tt float -desc debug -lr 0.000001 -no-skip
# -model m3 -tt fixed -desc debug -qd layer-wise -b-bits 8 -w-bits 8 -a-bits 8 -lr 0.000001 --checkpoint "m3/float/debug/model.npz" -no-skip
# -model m3 -tt adder -desc debug -qd layer-wise -adder-type 10 -b-bits 5 -w-bits 5 -a-bits 5 -lr 0.000001 --checkpoint "m3/float/debug/model.npz" -no-skip


# quantize_portion = tf.Variable(1.0, trainable=False)
# quantize_portion_act = tf.Variable(1.0, trainable=False)

def func_quantize_model(model, selected_train_type, weights_bits_total, bias_bits_total,
                        activation_bits_total, adder, std_keep_factor, quantize_actication, model_load_path, args):
    print("SELECTED selected_train_type", selected_train_type, f"and adder is \"{adder}\"")
    def get_quantizer(name, i="all", j="all", no_chan=False):
        suffix =  "" if no_chan else f"per_chan_{i}_{j}_f"
        if selected_train_type == "fixed":
            return mq.FlexPointQuantizer(name + suffix, weights_bits_total)
        return mq.AddQuantizer(name + suffix, f"{adder}Add{weights_bits_total}")

    # ["layer-wise", "channel-wise", "kernel-wise"]
    quant_depth = args["quant-depth"]

    def quant_Dense(dense, dense_name, neurons, use_adder):
        if args["no-weight"] != True:
            if quant_depth == "channel-wise" or quant_depth == "kernel-wise":
                dense.w.quant_out = mq.PerChannelQuantizer(f"{dense_name}_cw_w", get_quantizer, 1, neurons)
            elif quant_depth == "layer-wise":
                dense.w.quant_out = get_quantizer(f"{dense_name}_w", no_chan=True)
            else:
                raise Exception(f"Unkwon quant_depth {quant_depth}")
        if args["no-bias"] != True:
            dense.b.quant_out = mq.FlexPointQuantizer(f"{dense_name}_b", bias_bits_total)
        if args["no-activation"] != True:
            dense.quant_out = mq.FlexPointQuantizer(f"{dense_name}_out", activation_bits_total)

    if selected_train_type == "fixed" or selected_train_type == "adder":
        use_adder = selected_train_type == "adder"
        for dlayer in model.dense_list_help:
            dlayer: mq.DenseLayer = dlayer
            quant_Dense(dlayer, dlayer.name, dlayer.units, use_adder=use_adder)
            # if only_layerwise_quant:
            #     if adder == 666:
            #         dlayer.w.quant_out = mq.FlexPointQuantizer("dense" + i + "_w_fixed_quant", total_bits, debug=True,
            #                                                    extra_flex_shift=extra_flex_shift, dtype=dtype)
            #     else:
            #         dlayer.w.quant_out = mq.AddQuantizer("dense" + i + "_w_adder_quant", f"{adder}Add{total_bits}",
            #                                              pre_flex_bits=12, debug=True, dtype=dtype)
            # else:
            #     dlayer.w.quant_out = mq.PerChannelQuantizer("dense" + i + "_w_adder_quant",
            #                                                 create_quanziter_func=my_create_quanziter_func,
            #                                                 split_axis=1, splits=dlayer.units, debug=True, dtype=dtype)
            # dlayer.b.quant_out = mq.FlexPointQuantizer("dense" + i + "_b_fixed_quant", total_bits, debug=True,
            #                                            extra_flex_shift=extra_flex_shift, dtype=dtype)
    elif selected_train_type == "float":
        pass
    else:
        raise Exception("UNKNOWN train type!")


def func_compile_model(model, lr_base, INPUT_SHAPE, args):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_base, #0.000001,
                                         amsgrad=True)  # learning_rate=LEARN_RATE, amsgrad=True, decay=1e-6
    metrics = []
    model.compile(loss="mse", optimizer=optimizer, metrics=metrics)
    model.build([None] + INPUT_SHAPE)


def func_get_model(INPUT_SHAPE, args, batch_norm=False):
    OUTPUT_SHAPE = [1]
    TARGET_SHAPE = [1]
    WEIGHT_DECAY = 0.0
    if args["model"] == "m3":
        model = con_3_20.Con_3_20DNN(INPUT_SHAPE, OUTPUT_SHAPE, TARGET_SHAPE, WEIGHT_DECAY, dtype=dtype)
    elif args["model"] == "m5":
        model = con_5_20.Con_5_20DNN(INPUT_SHAPE, OUTPUT_SHAPE, TARGET_SHAPE, WEIGHT_DECAY, dtype=dtype)
    elif args["model"] == "m10":
        model = con_10_20.Con_10_20DNN(INPUT_SHAPE, OUTPUT_SHAPE, TARGET_SHAPE, WEIGHT_DECAY, dtype=dtype)
    else:
        raise Exception("Unkown model type " + args["model"])
    return model


def func_epochs_and_lrs(selected_train_type, lr_base):
    if selected_train_type == "float":
        EPOCHS_LRS = [(250, lr_base*10000), (400, lr_base*1000), (50, lr_base*100), (50, lr_base*10)]
    elif selected_train_type == "fixed":
        EPOCHS_LRS = [(20, lr_base*10000), (20, lr_base*1000), (10, lr_base*100), (10, lr_base*10)]
    elif selected_train_type == "adder":
        EPOCHS_LRS = [(20, lr_base*10000), (20, lr_base*1000), (10, lr_base*100), (10, lr_base*10)] # [(10, lr_base/100), (30, lr_base/10), (30, lr_base/100), (10, lr_base/1000)] # (5, lr_base/100), (5, lr_base/10),
    else:
        raise Exception("UNKNOWN train type " + str(selected_train_type))
    return EPOCHS_LRS


def func_get_dataset(args, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, dataset_path):
    csv_test_file = "../ds/val.csv"
    csv_train_file = "../ds/train.csv"
    # if old_val_ds:
    #     csv_test_file = "ds/val.csv"
    # else:
    #     csv_test_file = "ds/val2.csv"
    # if old_train_ds:
    #     csv_train_file = "ds/train.csv"
    # else:
    #     csv_train_file = "ds/train2.csv"

    def csv_val_gen():
        with open(csv_test_file) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                row = [float(x) for x in row]
                yield row[:-1], [row[-1]]

    def csv_train_gen():
        with open(csv_train_file) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                row = [float(x) for x in row]
                yield row[:-1], [row[-1]]
    train_dataset = tf.data.Dataset.from_generator(csv_train_gen, output_signature=(
        tf.TensorSpec(shape=(5), dtype=dtype),
        tf.TensorSpec(shape=(1),
                      dtype=dtype)))  # .take(250000) #.take((int)((40000*TRAIN_BATCH_SIZE)/10)).shard(4, random.randint(0,3))
    train_dataset = train_dataset.shuffle(128).batch(TRAIN_BATCH_SIZE).cache().prefetch(1)  # todo is shuffling correct here?

    test_dataset = tf.data.Dataset.from_generator(csv_val_gen, output_signature=(
        tf.TensorSpec(shape=(5), dtype=dtype),
        tf.TensorSpec(shape=(1),
                      dtype=dtype)))  # .take(250000) #.take((int)((40000*TRAIN_BATCH_SIZE)/10)).shard(4, random.randint(0,3))
    test_dataset = test_dataset.batch(TEST_BATCH_SIZE).cache().prefetch(1)
    return train_dataset, test_dataset


training = QTraining(
    INPUT_SHAPE=[1, 1, 5],
    batch_size=2048, # MUCH HIGHER NOW! This is the standard value when no batch size is provided!
    func_get_dataset=func_get_dataset,
    func_get_model=func_get_model,
    func_quantize_model=func_quantize_model,
    func_compile_model=func_compile_model,
    func_epochs_and_lrs=func_epochs_and_lrs,
    func_load_inital_model=func_load_inital_model,
    quantize_actication = True
)