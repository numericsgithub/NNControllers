import os
import sys
import json

from keras.callbacks import ReduceLROnPlateau
from tensorflow.python.ops.gen_dataset_ops import PrefetchDataset


sys.path.insert(1, os.path.join(sys.path[0], '../..'))


import mquat as mq  # noqa: E402
from tensorflow import keras
import datetime  # noqa: E402
import tensorflow.keras.callbacks as callbacks
import tensorflow as tf
from argparse import ArgumentParser
import time

# selected_train_type_index = int(sys.argv[1])
# model_load_path = sys.argv[2]
# bias_bits_total = int(sys.argv[3])  # 8
# weights_bits_total = int(sys.argv[4])  # 3
# activation_bits_total = int(sys.argv[5])  # 8
# adder = int(sys.argv[6])  # 3

# --train-type float
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'



class Logger(object):
    def __init__(self, target, output_file):
        self.terminal = target
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


parser = ArgumentParser()
parser.add_argument("-desc", dest="desc", default="",
                    help="Short description. A folder with the description as a name will contain all files generated")
parser.add_argument("--checkpoint", dest="checkpoint",
                    help="Checkpoint file to load")
parser.add_argument("-tt", "--train-type", choices=["float", "fixed", "adder"], dest="train-type", required=True,
                    help="Set what kind of quantizers you want to choose. Either none at all, fixed or adder aware")

parser.add_argument("-qd", "--quant-depth", choices=["layer-wise", "channel-wise", "kernel-wise"], dest="quant-depth",
                    help="Set how to split up each quantization. Quantize each layer, channel or kernel with a seperate quantizer")
parser.add_argument('-b-bits', '--bias-bits', type=int, choices=range(1, 21), dest="bias-bits")
parser.add_argument('-w-bits', '--weight-bits', type=int, choices=range(1, 21), dest="weight-bits")
parser.add_argument('-a-bits', '--activation-bits', type=int, choices=range(1, 21), dest="activation-bits")

parser.add_argument('-no-b-quant', '--no-bias-quant', action='store_true', dest="no-bias")
parser.add_argument('-no-w-quant', '--no-weight-quant', action='store_true', dest="no-weight")
parser.add_argument('-no-a-quant', '--no-activation-quant', action='store_true', dest="no-activation")
parser.add_argument('-no-skip', '--no-skip', action='store_true', dest="no-skip")

parser.add_argument('-adder-type', dest="adder-type")
parser.add_argument("--dataset-path", dest="dataset_path", default=None,
                    help="Path to the tensorflow_datasets folder")
parser.add_argument('-bsize', type=int, dest="bsize", default=None)
parser.add_argument('-lr', type=float, dest="lr_base", default=None)
parser.add_argument('-opt', choices=["adam", "sgd"], dest="opt", required=False, default="adam", help="Set optimizer")

parser.add_argument("-model", "--model", choices=["m3", "m5", "m10"], dest="model", required=True,
                    help="Set what model to load")
_ = parser.parse_args()
args = {}

for arg in vars(_):
    args[arg] = getattr(_, arg)
print(args)


if args["train-type"] == "float":
    has_no_quant_settings = args["bias-bits"] == None and args["weight-bits"] == None and args["activation-bits"] == None
    has_no_quant_settings = has_no_quant_settings and args["quant-depth"] == None and args["adder-type"] == None
    has_no_quant_settings = has_no_quant_settings and args["no-bias"] == False and args["no-weight"] == False and args["no-activation"] == False
    if not has_no_quant_settings:
        raise Exception("Wrong Arguments! When training without quantization, quantization arguments are not allowed!")
else:
    if args["bias-bits"] == None and args["no-bias"] == False:
        raise Exception("Quantization of the bias is not specified! Either set --no-b-quant flag or set bits via --b-bits 8")
    if args["weight-bits"] == None and args["no-weight"] == False:
        raise Exception("Quantization of the weight is not specified! Either set --no-w-quant flag or set bits via --w-bits 8")
    if args["activation-bits"] == None and args["no-activation"] == False:
        raise Exception("Quantization of the activation is not specified! Either set --no-a-quant flag or set bits via --a-bits 8")
    if args["quant-depth"] == None:
        raise Exception("Quantization depth is not specified! Set this via --quant-depth")
    if args["train-type"] == "adder":
        if args["adder-type"] == None:
            raise Exception("Adder type is not specified! Set this via --adder-type")
    if args["train-type"] == "fixed":
        if args["adder-type"] != None:
            raise Exception("Adder type is specified! But train type is \"fixed\"!")

train_type = args["train-type"]
quant_depth = args["quant-depth"]
model_load_path = args["checkpoint"]
bias_bits_total = args["bias-bits"]
weights_bits_total = args["weight-bits"]
activation_bits_total = args["activation-bits"]
adder = args["adder-type"]
general_desc = args["desc"]
dataset_path = args["dataset_path"]
time_start = time.time()
lr_base = args["lr_base"]

class QTraining:

    def __init__(self, INPUT_SHAPE, batch_size,
                 func_get_dataset, func_get_model, func_quantize_model, func_compile_model,
                 func_epochs_and_lrs, func_load_inital_model, quantize_actication = True):
        try:
            model_name = args["model"]
            batch_size = batch_size if args["bsize"] is None else args["bsize"]
            print("Batch size is ", batch_size)

            TRAIN_BATCH_SIZE = batch_size # 256
            TEST_BATCH_SIZE = batch_size # 256

            INPUT_SHAPE = INPUT_SHAPE # [28, 28, 1]
            # INPUT_SHAPE = [32, 32, 3]

            # ["float", "fixed", "adder"]
            SAVE_PREFIX = {-1: lambda: None,
                           "float": lambda: f"{model_name}/float/{general_desc}/" + "{}",
                           "fixed": lambda: f"{model_name}/{quant_depth}_fixed/{weights_bits_total}w_{bias_bits_total}b_{activation_bits_total}a/{general_desc}/" + "{}",
                           "adder": lambda: f"{model_name}/{quant_depth}_adder/{adder}_adder_{weights_bits_total}w_{bias_bits_total}b_{activation_bits_total}a/{general_desc}/" + "{}"}


            model_saved_path = SAVE_PREFIX[train_type]()
            MODEL_SAVEPATH = model_saved_path.format("model.npz")
            MODEL_SAVEPATH_Q = model_saved_path.format("model_q.npz")
            MODEL_SAVEPATH_BEST = model_saved_path.format("model_best.npz")
            MODEL_SAVEPATH_Q_BEST = model_saved_path.format("model_best_q.npz")
            MODEL_SAVEPATH_Q_FUSEB = model_saved_path.format("model_q_fusebatch.npz")
            os.makedirs(model_saved_path.format(""), exist_ok=True)
            sys.stdout = Logger(sys.stdout, model_saved_path.format("stdout.log"))
            sys.stderr = Logger(sys.stderr, model_saved_path.format("stderr.log"))
            if os.path.exists(MODEL_SAVEPATH):
                print(f"Already done! Skipping! Because {MODEL_SAVEPATH} already exists")
                if args["no-skip"] == True:
                    print("SKIPPING PREVENTED DUE TO no-skip FLAG!")
                else:
                    exit(0)
            # if ".npz" not in model_load_path.lower():
            #     model_load_path = SAVE_PREFIX[selected_train_type_index - 1]()


            train_dataset, test_dataset = func_get_dataset(args, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, dataset_path)

            std_keep_factor = 0.0
            model = func_get_model(INPUT_SHAPE, args, True)

            func_quantize_model(model, train_type, weights_bits_total, bias_bits_total, activation_bits_total, adder,
                                std_keep_factor, quantize_actication, model_load_path, args)


            func_compile_model(model, lr_base, INPUT_SHAPE, args)

            if model_load_path != None:
                model.loadVariablesNPZ(model_load_path, try_load_quantizers=False) # loss: 0.0137 - top 1: 0.9960 - val_loss: 0.0302 - val_top 1: 0.9908
            else:
                func_load_inital_model(model, args)
            for q in model.getQuantizers():
                q.reset()
            for v in model.getVariables():
                v: mq.Variable = v
                v()

            # model.evaluate(train_dataset) # initialize the activation quantization

            #train_dataset = train_dataset.cache("/run/determined/workdir/shared/datasets/img2012_train")

            evaluation_results = model.evaluate(test_dataset, verbose=2)
            with open(model_saved_path.format("ptq.txt"), "w") as f:
                f.write(str(evaluation_results))
            print("\n\tTYPE", train_type, "\n\tmodel_load_path", model_load_path, "\n\tbias_bits_total", bias_bits_total,
                  "\n\tweights_bits_total", weights_bits_total, "\n\tactivation_bits_total", activation_bits_total, "\n\tadder", adder)
            with open("before_train_results.txt", "a") as f:
                f.write(f"{MODEL_SAVEPATH} | {model_load_path} | {str(evaluation_results)}\r\n")
            print(evaluation_results)


            self.best_val_loss = 999999999999.0
            self.best_val_loss_results = None
            self.best_val_loss_epoch = None
            self.epochs_counter = 1
            self.epoch_start = time.time()

            def func_get_model_FUSEB(model, MODEL_SAVEPATH, lr_base, INPUT_SHAPE, args):
                model.saveVariablesNPZ(MODEL_SAVEPATH, quantisize=True, fuse_batch_norm=True)
                del model
                fb_model = func_get_model(INPUT_SHAPE, args, False)
                func_compile_model(fb_model, lr_base, INPUT_SHAPE, args)
                fb_model.loadVariablesNPZ(MODEL_SAVEPATH)
                return fb_model


            class LocalCallbacks(callbacks.Callback):
                early_stopped = False

                def __init__(self, qTraining):
                    self.qTraining = qTraining
                    pass

                # def on_batch_end(self, batch, logs=None):
                #     print("MEMORY USAGE", args["bsize"], batch, tf.config.experimental.get_memory_info("GPU:0"))
                #     exit(0)
                #     # if batch > 60:
                #     #     exit(0)

                # def on_epoch_begin(self, epoch, logs={}):
                #     for logger in all_logger:
                #         logger.takeSnapshot(epoch)

                def on_epoch_end(self, epoch, logs={}):
                    cur_val_loss = logs["val_loss"]
                    with open(model_saved_path.format("epochs_log.txt"), "a") as f:
                        f.write(f"{self.qTraining.epochs_counter} | {epoch} | {model.optimizer.lr.numpy()} | {str(logs)} | {self.qTraining.best_val_loss > cur_val_loss} | {time.time()-self.qTraining.epoch_start}\n")
                    self.qTraining.epochs_counter += 1
                    self.qTraining.epoch_start = time.time()
                    if self.qTraining.best_val_loss > cur_val_loss:
                        print()
                        print(f"Found better one! {self.qTraining.best_val_loss} -> {cur_val_loss}")
                        self.qTraining.best_val_loss = cur_val_loss
                        self.qTraining.best_val_loss_results = logs
                        self.qTraining.best_val_loss_epoch = self.qTraining.epochs_counter
                        with open(model_saved_path.format("results.txt"), "w") as f:
                            f.write(str(logs))
                        with open(model_saved_path.format("config.json"), "w") as f:
                            f.write(str(args))
                        model.saveVariablesNPZ(MODEL_SAVEPATH, quantisize=False)
                        model.saveVariablesNPZ(MODEL_SAVEPATH_Q, quantisize=True)
                    #     model.saveVariablesNPZ(MODEL_SAVEPATH_BEST, quantisize=False)
                    #     model.saveVariablesNPZ(MODEL_SAVEPATH_Q_BEST, quantisize=True)
                    #     #model.saveVariablesNPZ(MODEL_SAVEPATH_Q_FUSEB, fuse_batch_norm=True, quantisize=True)
                    # model.saveVariablesNPZ(MODEL_SAVEPATH, quantisize=False)
                    # model.saveVariablesNPZ(MODEL_SAVEPATH_Q, quantisize=True)
                    # #model.saveVariablesNPZ(MODEL_SAVEPATH_Q_FUSEB, fuse_batch_norm=True, quantisize=True)



            EPOCHS_LRS = func_epochs_and_lrs(train_type, lr_base)
            #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tensorboard_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, profile_batch=(1,20)) # tensorboard --logdir tensorboard_logs
            for run_id, (epochs, lr) in enumerate(EPOCHS_LRS):
                if epochs == "FUSEB":
                    MODEL_SAVEPATH = model_saved_path.format("model_fuseb.npz")
                    MODEL_SAVEPATH_Q = model_saved_path.format("model_fuseb_q.npz")
                    model = func_get_model_FUSEB(model, MODEL_SAVEPATH, lr_base, INPUT_SHAPE, args)
                    model.saveVariablesNPZ(MODEL_SAVEPATH_Q,quantisize=True, fuse_batch_norm=False)
                    LocalCallbacks.early_stopped = False
                    continue
                if LocalCallbacks.early_stopped:
                    print("EARLY STOP DETECTED!")
                    continue
                model.optimizer.lr.assign(lr)

                callbacks_list = []

                # if run_id == 0:
                #     logs = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S " + f"{model_name} {train_type} {quant_depth}")
                #     tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                #                                                      histogram_freq=1,
                #                                                      profile_batch='100,120')
                #     callbacks_list.append(tboard_callback)

                # with tf.profiler.experimental.Profile('logdir'):
                model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, use_multiprocessing=True, verbose=2,
                          validation_freq=1, callbacks=[LocalCallbacks(self), *callbacks_list])#, callbacks=[reduce_lr])#, callbacks=[LocalCallbacks()])
                if os.path.exists(MODEL_SAVEPATH):
                    model.loadVariablesNPZ(MODEL_SAVEPATH)
                else:
                    if model_load_path != None:
                        model.loadVariablesNPZ(model_load_path, try_load_quantizers=False)  # loss: 0.0137 - top 1: 0.9960 - val_loss: 0.0302 - val_top 1: 0.9908
                    else:
                        func_load_inital_model(model, args)
                # old_lr = model.optimizer.lr.read_value()
                # model.optimizer.lr.assign(old_lr * 0.1)

            print("finished training with std_keep_factor=", std_keep_factor)
            #model.saveVariablesNPZ("LeNet_pre_trained")
            extra_info = {"epochs_count": str(self.epochs_counter), "time": str(time.time()-time_start), "best_epoch": str(self.best_val_loss_epoch)}
            evaluation_results = model.evaluate(test_dataset, verbose=2)
            with open("after_train_results.txt", "a") as f:
                f.write(f"{MODEL_SAVEPATH} | {model_load_path} | {str(args)} | {str(evaluation_results)}\n")
            with open("after_train_best_results.txt", "a") as f:
                f.write(f"{MODEL_SAVEPATH} | {model_load_path} | {str(args)} | {str(self.best_val_loss_results)} | {str(extra_info)}\n")
            with open(model_saved_path.format("extra_info.txt"), "w") as f:
                f.write(str(extra_info))
            print(evaluation_results)
        except Exception as ex:
            raise ex