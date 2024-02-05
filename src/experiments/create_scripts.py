import numpy as np
import os


python_script_name = "python Con_X_20DNN_Training.py"
model_name = "m3"
no_skip = "-no-skip" # or "" if you do not want to FORCE the experiment to re-run
do_the_new_training = True

# just a script that has all created runs in one. Just a practical thing
all_in_one_script_name = "all_experiments"
if os.path.exists("all_in_one_script_name.sh"):
    os.remove("all_in_one_script_name.sh")

# -model m3 -tt float -desc debug -lr 0.000001 -no-skip
# -model m3 -tt fixed -desc debug -qd layer-wise -b-bits 8 -w-bits 8 -a-bits 8 -lr 0.000001 --checkpoint "m3/float/debug/model.npz" -no-skip
# -model m3 -tt adder -desc debug -qd layer-wise -adder-type 10 -b-bits 5 -w-bits 5 -a-bits 5 -lr 0.000001 --checkpoint "m3/float/debug/model.npz" -no-skip



def create_run_and_append(run_list, lr, desc, train_type, quant_depth=None, bias_bits=None, weight_bits=None, checkpoint=None, adder=None, batch_size=2048):
    SAVE_PREFIX = {-1: lambda: None,
                   "float": lambda: f"{model_name}/float/{desc}/" + "{}",
                   "fixed": lambda: f"{model_name}/{quant_depth}_fixed/{weight_bits}w_{bias_bits}b_{None}a/{desc}/" + "{}",
                   "adder": lambda: f"{model_name}/{quant_depth}_adder/{adder}_adder_{weight_bits}w_{bias_bits}b_{None}a/{desc}/" + "{}"}

    model_saved_path = SAVE_PREFIX[train_type]()
    resulting_checkpoint_file = model_saved_path.format("model_q.npz")

    run = f"{python_script_name} -model {model_name} -lr {lr} -desc {desc} {no_skip}"
    if train_type == "float":
        run += " -tt float"
    elif train_type == "fixed" or train_type == "adder":
        run += f" -tt {train_type} -qd {quant_depth} -b-bits {bias_bits} -w-bits {weight_bits} --no-activation-quant --checkpoint {checkpoint} -bsize {batch_size}"
    else:
        raise Exception("UNKOWN TRAIN TYPE! float, fixed or adder!")
    if train_type == "adder":
        run += f" -adder-type {adder}"

    run_list.append(run)
    return resulting_checkpoint_file

def create_experiment(run_list, experiment_name):
    with open(experiment_name + ".sh", "w") as f:
        for run in run_list:
            f.write(run + "\n")
    with open(all_in_one_script_name + ".sh", "a") as f:
        f.write("# " + experiment_name + "\n")
        for run in run_list:
            f.write(run + "\n")
        f.write("" + "\n")
        f.write("" + "\n")

# Training the DNN in float
all_runs = []
float_checkpoint = create_run_and_append(all_runs, "0.000001", "default", "float")
create_experiment(all_runs, "float_training")


quant_checkpoint_map = {}


for quant_depth in ["layer-wise", "channel-wise"]:
    all_runs = []
    last_checkpoint = float_checkpoint
    quant_checkpoint_map[quant_depth] = {}
    for bits in [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]:
        batch_size = 2048
        if not do_the_new_training: # in the past the code was not well optimized... Memory leaks were present back in that time :-(
            batch_size = 512
        last_checkpoint = create_run_and_append(all_runs, "0.000001", "default", "fixed",
                                                 quant_depth=quant_depth, bias_bits=str(bits), weight_bits=str(bits),
                                                 checkpoint=last_checkpoint, batch_size=batch_size)
        quant_checkpoint_map[quant_depth][bits] = last_checkpoint
        if do_the_new_training:
            last_checkpoint = float_checkpoint # With bigger batch size we do not need to train it down anymore!
    create_experiment(all_runs, f"fxp_{quant_depth}_training")


for quant_depth in ["layer-wise", "channel-wise"]:
    for adder in ["10", "11", "12"]: # cost-0, cost-1 and cost-2
        all_runs = []
        for bits in [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]:
            if do_the_new_training:
                checkpoint = float_checkpoint # With bigger batch size we do not need to train it down anymore!
            else:
                checkpoint = quant_checkpoint_map[quant_depth][bits]
            create_run_and_append(all_runs, "0.000001", "default", "adder", adder=adder,
                                  quant_depth=quant_depth, bias_bits=str(bits), weight_bits=str(bits), checkpoint=checkpoint)
        create_experiment(all_runs, f"adder_{quant_depth}_cost_{adder}_training")