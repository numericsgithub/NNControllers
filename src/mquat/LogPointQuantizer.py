# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import Variable
import numpy as np

from .Utilities import to_variable, to_value
from .Quantizer import Quantizer
import random
import time
from .QuantizerBase import DEFAULT_DATATYPE

# TODO Add variable for the self.flex_inferences stuff
# TODO Deactivate the dynamic change of the bits
# TODO Add threshold (TODO count outlier)
# TODO Implement percentile keep stuff...

class LogPointQuantizer(Quantizer):
    """fixed point quantisizer with self adjusting bits_before and bits_after

    it is using a signed fixed point representation and is s based on the
    Quantizer.
    the quantisation range and number of steps is calculated by the
    two bitsizes.

    Parameters:
        name (string):
            the name of the layer in the TensorFlow graph.
        total_bits (int):
            the amount of bits for the fixed point quantisizer
        leak_clip (float):
            leak factor for backpropagation.
            applied when values are clipped to quantization range.
        dtype (tf.dtypes.DType):
            datatype of the layers's operations.
            default is float32.
    """
    total_bits: tf.Variable
    flex_inferences: tf.Variable
    std_keep_factor: tf.Variable

    INT_FRAC_EXTENSION = 10
    PLOTTING = False
    #test_summary_writer = tf.summary.create_file_writer('logs/fit/' + "LogPointQuantizer")


    def __init__(self, name, total_bits, has_signed_bit=True, std_keep_factor=0.0, flex_inferences=1, leak_clip=0.0,
                 dtype=DEFAULT_DATATYPE, debug=False, extra_flex_shift=0, print_log_num=False, bits_left_of_comma = 4,
                 base=2.0):
        super().__init__(name, dtype=dtype)
        use_two_complement = False
        self.base = tf.Variable(base, trainable=False)
        self.extra_flex_shift = extra_flex_shift
        self.total_bits = to_variable(total_bits, tf.int32)
        self.flex_inferences = to_variable(flex_inferences, tf.int32)
        self.std_keep_factor = to_variable(std_keep_factor, tf.float32)
        self.debug = debug
        self.leak_clip = tf.cast(leak_clip, dtype)
        self.inference_counter = tf.Variable(0, name=self.name+"_inference_counter", trainable=False, dtype=tf.int32)
        self.input_sample = tf.Variable([], name=self.name+"_input_sample", shape=[None], trainable=False, dtype=dtype, validate_shape=False)
        self.maybe_inversed_step_diff = tf.Variable(0, name=self.name+"_maybe_inversed_step_diff", trainable=False, dtype=dtype)
        self.min_value_exponent = tf.Variable(0, name=self.name + "_min_value", trainable=False, dtype=dtype)
        self.max_value_exponent = tf.Variable(0, name=self.name + "_max_value", trainable=False, dtype=dtype)
        self.min_value = tf.Variable(-1.0, name=self.name+"_min_value", trainable=False, dtype=dtype) # todo implement the has_sign case -> [0;1] or [-1;1]
        self.max_value = tf.Variable(1.0, name=self.name+"_max_value", trainable=False, dtype=dtype) # todo also implement the use_two_complement case -> [-1;1] or [-inf;+inf] and [0;inf] or [0;1]
        self.b_frc = tf.Variable(0, name=self.name+"_b_frc", trainable=False, dtype=dtype)
        self.b_frc_trend_setter = tf.Variable(0, name=self.name+"_b_frc_trend_setter", trainable=False, dtype=dtype)
        self.print_log_num = to_variable(print_log_num, tf.bool)
        self.use_two_complement = to_variable(use_two_complement, tf.bool)
        self.has_signed_bit = to_variable(has_signed_bit, tf.bool)
        self.preset_b_int = bits_left_of_comma
        b_int = tf.cast(self.preset_b_int, tf.float64)  # tf.maximum(tf.reduce_min(q_low), tf.reduce_min(q_high))
        self.setBits(b_int)


    def reset(self):
        """
        Resets the Quantizer. The next inference triggers the coefficient search again.
        Returns:

        """
        self.inference_counter.assign(0)
        self.input_sample.assign([])
        self.maybe_inversed_step_diff.assign(0)
        self.min_value_exponent.assign(0)
        self.max_value_exponent.assign(0)
        self.b_frc.assign(0)
        #self.b_frc_trend_setter.assign(0)


    def getCoeffs(self):
        inputs = tf.range(-1.0, limit=1.0, delta=0.000001)
        quanted_inputs = self(inputs)
        uniq = tf.unique(quanted_inputs).y
        tf.print(self.name, "getCoeffs",tf.size(uniq) , uniq)
        return uniq.numpy()

    def isNonUniform(self):
        return True

    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend([self.inference_counter, self.maybe_inversed_step_diff, self.min_value_exponent, self.max_value_exponent,
                          self.b_frc, self.b_frc_trend_setter])
        return variables


    def collect_input_sample(self, inputs):
        tf.cond(tf.reduce_sum(tf.where(inputs > 0, 1, 1)) == 0,
                lambda :self.inference_counter.assign_add(-1),
                lambda :self.inference_counter.assign_add(0))
        self.input_sample.assign(tf.concat([tf.reshape(inputs, [-1]), self.input_sample], 0))
        return 0

    def __filterInputs(self, inputs, std_keep_factor):
        std_keep_factor = tf.cast(std_keep_factor, self.dtype)
        # if std_keep_factor == 0.0:
        #     return inputs, inputs
        std_keep = tf.math.reduce_std(inputs) * std_keep_factor # TODO Unique verändert das Ergebnis
        kept = tf.sort(inputs[tf.math.abs(inputs - tf.math.reduce_mean(inputs)) <= std_keep])
        kept_s = tf.size(kept)
        dropped = tf.sort(inputs[tf.math.abs(inputs - tf.math.reduce_mean(inputs)) > std_keep])
        dropped_s = tf.size(dropped)
        # if self.debug:
        #     tf.print("FlexPointQuant: ", self.name,
        #              "kept", tf.round(100 * (kept_s / tf.size(inputs))), "%", kept_s, kept,
        #              "dropped", tf.round(100 * (dropped_s / tf.size(inputs))), "%", dropped_s, dropped, "input size is", tf.size(inputs))
        clip_min = tf.reduce_min(kept)
        clip_max = tf.reduce_max(kept)
        clipped = tf.where(tf.math.abs(inputs - tf.math.reduce_mean(inputs)) <= std_keep,
                           inputs,
                           tf.clip_by_value(inputs, clip_min, clip_max))
        return clipped #kept#, clipped


    def warn_not_quantizing(self):
        tf.print("FlexPointQuant:", self.name, "is not quantizing this inference! Still creating the sample! ",
                 self.inference_counter, "/", self.flex_inferences, "inputs collected")
        return 0

    def FatalNumericalError(self):
        tf.print("FlexPointQuant:", self.name,"ERROR: TOO MANY BITS! MORE THAN 63 BITS ARE NOT SUPPORTED!")
        return 0

    def WarningNumericalError(self):
        tf.print("FlexPointQuant:", self.name,"WARNING: TOO MANY BITS! More than 24 bits are not yet tested for numerical errors!")
        return 0

    def setBitsPlot(self, stds, errors, chosen, std_errors):
        # if "_b_" in self.name:
        #     return 0.0
        # if "conv_f_fixed_quant" != self.name:
        #     return 0.0
        # time.sleep(random.uniform(0, 1))
        # while LogPointQuantizer.PLOTTING:
        #     time.sleep(random.uniform(0, 1))
        # LogPointQuantizer.PLOTTING = True
        # chosen = chosen.numpy()
        # plt.plot(stds, errors, color="blue", label="quant error")
        # plt.plot(stds, std_errors, color="orange", label="clip error")
        # plt.vlines([stds[chosen]], ymin=0, ymax=np.max(errors), colors=["red"], label="chosen")
        # plt.title(self.name + "with " + str(self.total_bits.numpy()) + "bits")
        # plt.legend()
        # plt.xlabel("std factor")
        # plt.ylabel("absolute error")
        # print(self.name, "stds[chosen] red", stds[chosen].numpy(), errors[chosen].numpy())
        # plt.show()
        #
        # LogPointQuantizer.PLOTTING = False
        return 0.0

    @tf.autograph.experimental.do_not_convert
    def setBitsBeforeAndAfter(self):
        # all_std_filters = tf.range(3.0, 10, 0.2, dtype=tf.float32)
        # all_clipping_error, all_rounding_errors = tf.map_fn(self.setBitsBeforeAndAfter_std,
        #                      [all_std_filters],
        #                         fn_output_signature=[tf.float32, tf.float32], parallel_iterations=1)
        # all_errors_ez_vs_lns = all_clipping_error + all_rounding_errors
        # selected = tf.argmin(all_errors_ez_vs_lns)
        # self.std_keep_factor = all_std_filters[selected]
        # tf.py_function(func=self.setBitsPlot,  # stds, errors, chosen, std_errors
        #                    inp=[all_std_filters, all_rounding_errors,
        #                         selected, all_clipping_error],
        #                    Tout=[tf.float32])
        self.setBitsBeforeAndAfter_std(0.0)

        org_inputs = self.input_sample
        org_inputs = tf.reshape(org_inputs, [-1])
        org_input_no_zeros = tf.where(org_inputs == 0, False, True)
        org_inputs = tf.math.log(tf.abs(org_inputs)) / tf.math.log(tf.cast(self.base, self.dtype))  # get log2(inputs)
        org_inputs = org_inputs[org_input_no_zeros]
        clipped_inputs = self.__filterInputs(org_inputs, self.std_keep_factor)

        if self.debug:
            tf.print("FlexPointQuant:", self.name, "all kept unique inputs", tf.unique(clipped_inputs)[0])
        tmp = tf.clip_by_value(clipped_inputs, self.min_value_exponent, self.max_value_exponent)
        tmp = (tmp - self.min_value_exponent)
        tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff,
                      lambda: tmp / self.maybe_inversed_step_diff)
        tmp = tf.floor(tmp + 0.5)
        quant_input = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
                              lambda: tmp * self.maybe_inversed_step_diff) + self.min_value_exponent

        direct_hit = tf.reduce_sum(tf.where(quant_input == tmp, 1, 0))
        sample_size = tf.reduce_sum(tf.where(quant_input > 0, 1, 1))
        tf.cond(tf.logical_and(tf.abs(self.b_frc) < 64, (self.total_bits + self.INT_FRAC_EXTENSION) < 64),
                lambda: 0,
                lambda: self.FatalNumericalError())
        tf.cond(tf.logical_and(tf.abs(self.b_frc) < 24, (self.total_bits + self.INT_FRAC_EXTENSION) < 24),
                lambda: 0,
                lambda: self.WarningNumericalError())
        if self.debug:
            tf.print("FlexPointQuant:", self.name, "got", sample_size, "weights with", direct_hit, "direct matches (",
                 (direct_hit / sample_size) * 100.0, "%)")

        # LogPointQuantizer.error_statistic.assign(tf.concat([tf.reshape(inputs - tmp, [-1]), LogPointQuantizer.error_statistic], 0))

        if self.debug:
            tf.print("LogPointQuant:", self.name, "setting bits based on sample size",
                     sample_size, "[", tf.reduce_min(org_inputs), ";", tf.reduce_max(org_inputs), "]",
                     " bits set to", self.b_frc + tf.cast(self.total_bits, tf.float32), self.b_frc, "[", self.min_value_exponent, ";", self.max_value_exponent, "]")
            tf.print("FlexPointQuant:", self.name, "quantized sample with unique is", "[", tf.reduce_min(org_inputs), ";",
                     tf.reduce_max(org_inputs), "]", tf.unique(org_inputs))

        self.input_sample.assign([])
        return 0.0

    def getRange(self):
        steps = tf.cast(tf.range(0, tf.pow(2, self.total_bits)), tf.float32)
        steps = tf.cond(self.b_frc >= 0, lambda: steps / self.maybe_inversed_step_diff, lambda: steps * self.maybe_inversed_step_diff) + self.min_value_exponent

        input_zeros = tf.where(steps == self.min_value_exponent, True, False)  # All values with zero are encoded to the smallest value
        lns_steps = tf.pow(self.base, steps)   # reverse the log2(inputs) by calculating 2^(log2(inputs))
        lns_steps = tf.where(input_zeros, 0.0, lns_steps)
        return lns_steps

    def getMin(self, inputs):
        t = tf.cast(self.total_bits, tf.float64)
        _05 = tf.cast(0.5, dtype=tf.float64)
        _1 = tf.cast(1, dtype=tf.float64)
        _2 = tf.cast(2, dtype=tf.float64)
        result = -tf.pow(_2, inputs - _1)
        return tf.cond(self.use_two_complement, lambda: result, lambda: -(tf.pow(_2, inputs) - tf.pow(_05, t - inputs)))

    def getMax(self, inputs):
        t = tf.cast(self.total_bits, tf.float64)
        _05 = tf.cast(0.5, dtype=tf.float64)
        _1 = tf.cast(1, dtype=tf.float64)
        _2 = tf.cast(2, dtype=tf.float64)
        result = tf.pow(_2, inputs - _1) - tf.pow(_05, t - inputs)
        return tf.cond(self.use_two_complement, lambda: result, lambda: tf.zeros_like(result))

    def setBits(self, b_int):
        t = tf.cast(self.total_bits, tf.float64)
        # _05 = tf.cast(0.5, dtype=tf.float64)
        # _1 = tf.cast(1, dtype=tf.float64)
        # _2 = tf.cast(2, dtype=tf.float64)
        b_frc = t - b_int
        min = self.getMin(b_int) # -tf.pow(_2, b_int - _1)
        max = self.getMax(b_int) # tf.pow(_2, b_int - _1) - tf.pow(_05, t - b_int)
        inv_step = tf.cast(tf.pow(tf.cast(2, dtype=tf.int64), tf.cast(tf.abs(b_frc), dtype=tf.int64)), dtype=tf.float64)
        self.maybe_inversed_step_diff.assign(tf.cast(inv_step, dtype=self.dtype))
        self.min_value_exponent.assign(tf.cast(min, dtype=self.dtype))
        self.max_value_exponent.assign(tf.cast(max, dtype=self.dtype))
        self.b_frc.assign(tf.cast(b_frc, dtype=self.dtype))


    def setBitsBeforeAndAfter_std(self, std_filter):
        org_inputs = self.input_sample
        org_inputs = tf.reshape(org_inputs, [-1])
        org_input_no_zeros = tf.where(org_inputs == 0, False, True)
        org_inputs = tf.math.log(tf.abs(org_inputs)) / tf.math.log(tf.cast(self.base, dtype=self.dtype))  # get log2(inputs)
        org_inputs = org_inputs[org_input_no_zeros]
        #org_inputs = org_inputs/tf.reduce_max(tf.abs(org_inputs))

        # if self.debug:
        #     tf.print("FlexPointQuant:", self.name, "all samples unique inputs",inputs)
        clipped_inputs = tf.cond(std_filter != tf.constant(0.0), lambda: self.__filterInputs(org_inputs, std_filter), lambda: org_inputs)
        clipping_error = tf.reduce_sum(tf.abs(clipped_inputs - org_inputs))

        int_frac_extension = tf.cast(LogPointQuantizer.INT_FRAC_EXTENSION, dtype=tf.float64)



        t = tf.cast(self.total_bits, tf.float64)
        T = tf.cast(tf.range(1 - int_frac_extension, t + 1 + int_frac_extension), dtype=tf.float64)
        S = tf.cast(clipped_inputs, tf.float64)
        _05 = tf.cast(0.5, dtype=tf.float64)
        _1 = tf.cast(1, dtype=tf.float64)
        _2 = tf.cast(2, dtype=tf.float64)

        #q_low = tf.where((-tf.pow(_2, T - _1)) <= tf.reduce_min(S))
        q_low = tf.where(self.getMin(T) <= tf.reduce_min(S))
        q_low = tf.reshape(tf.gather(T, q_low), [-1])
        q_low = tf.concat([[t + int_frac_extension], q_low], axis=0)

        #q_high = tf.where(((tf.pow(_2, T - _1) - tf.pow(_05, tf.cast(t - T, dtype=tf.float64)))) >= tf.reduce_max(S))
        q_high = tf.where(self.getMax(T) >= tf.reduce_max(S))
        q_high = tf.reshape(tf.gather(T, q_high), [-1])
        q_high = tf.concat([[t + int_frac_extension], q_high], axis=0)

        b_int = tf.cast(self.preset_b_int, tf.float64) #tf.maximum(tf.reduce_min(q_low), tf.reduce_min(q_high))
        self.setBits(b_int)

        # quant_factor = tf.reduce_max(tf.abs(clipped_inputs))
        # tmp = clipped_inputs / quant_factor
        # tmp = tmp * 128.0
        # quant_input = tf.round(tmp)
        # quant_input = quant_input / 128.0
        # quant_input = quant_input * quant_factor

        tmp = tf.clip_by_value(clipped_inputs, self.min_value_exponent, self.max_value_exponent)
        tmp = (tmp - self.min_value_exponent)
        tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff,
                      lambda: tmp / self.maybe_inversed_step_diff)
        tmp = tf.floor(tmp + 0.5)
        quant_input = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
                      lambda: tmp * self.maybe_inversed_step_diff) + self.min_value_exponent

        rounding_errors = tf.reduce_sum(tf.abs(quant_input - clipped_inputs))

        # tmp = tf.clip_by_value(inputs, self.min_value_exponent, self.max_value_exponent)
        # tmp = (tmp - self.min_value_exponent)
        # tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff,
        #               lambda: tmp / self.maybe_inversed_step_diff)
        # tmp = tf.floor(tmp + 0.5)
        # tmp = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
        #               lambda: tmp * self.maybe_inversed_step_diff) + self.min_value_exponent

        # # the check for numerical errors works by converting back the quantized number to an integer number
        # # quantized * step_diff == integer number
        # # checker = quantized * step_diff
        # # checker_int = int(checker)
        # # checker_int != checker -> numerical error!
        # checker = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff, lambda: tmp / self.maybe_inversed_step_diff)
        # checker_int = tf.cast(tf.cast(checker, tf.int64), dtype=tf.float32)
        # tf.print("FlexPointQuant:", self.name, "checking sample for numerical errors", inputs, tmp, self.maybe_inversed_step_diff)
        # tf.print("FlexPointQuant:", self.name, "checking sample for numerical errors", checker_int, checker)
        # tf.print("FlexPointQuant:", self.name, "checking sample for numerical errors", tf.reduce_all(tf.math.equal(checker_int, checker)))
        # tf.cond(tf.reduce_all(tf.math.equal(checker_int, checker)),
        #         lambda: 0,
        #         lambda: self.FatalNumericalError())


        # direct_hit = tf.reduce_sum(tf.where(quant_input == tmp, 1, 0))
        # sample_size = tf.reduce_sum(tf.where(quant_input > 0, 1, 1))
        # tf.cond(tf.logical_and(tf.abs(b_frc) < 64, (self.total_bits + self.INT_FRAC_EXTENSION) < 64),
        #         lambda: 0,
        #         lambda: self.FatalNumericalError())
        # tf.cond(tf.logical_and(tf.abs(b_frc) < 24, (self.total_bits + self.INT_FRAC_EXTENSION) < 24),
        #         lambda: 0,
        #         lambda: self.WarningNumericalError())
        #
        # tf.print("FlexPointQuant:", self.name, "got", sample_size, "weights with", direct_hit, "direct matches (",(direct_hit/sample_size)*100.0,"%)")
        #
        # #LogPointQuantizer.error_statistic.assign(tf.concat([tf.reshape(inputs - tmp, [-1]), LogPointQuantizer.error_statistic], 0))
        #
        # if self.debug:
        #     tf.print("FlexPointQuant:", self.name, "setting bits based on sample size",
        #              sample_size, "[", tf.reduce_min(org_inputs), ";", tf.reduce_max(org_inputs), "]",
        #              " bits set to", b_int, b_frc, "[", min, ";", max, "]")
        #     tf.print("FlexPointQuant:", self.name, "quantized sample with unique is", "[", tf.reduce_min(tmp), ";",
        #              tf.reduce_max(tmp), "]", tf.unique(tmp))
        return [clipping_error, rounding_errors]

    def assign_add_counter(self):
        self.inference_counter.assign_add(1)
        return 0

    def print_log2(self, log2_fixed, text=""):
        tf.print("LogPointQuant:", self.name, text, "[", tf.reduce_min(log2_fixed), ";", tf.reduce_max(log2_fixed), "]", log2_fixed, summarize=-1)
        return 0.0

    def convertToLNS(self, inputs):

        input_signs = tf.where(inputs >= 0, tf.cast(1.0, dtype=self.dtype), tf.cast(-1.0, dtype=self.dtype))
        inputs_log2 = tf.math.log(tf.abs(inputs)) / tf.math.log(tf.cast(self.base, self.dtype))  # get log2(inputs)
        # tf.cond(self.print_log_num, lambda: self.print_log2(inputs_log2, "log2(inputs)"), lambda: 0.0)
        tmp = tf.clip_by_value(inputs_log2, self.min_value_exponent, self.max_value_exponent)
        #tmp = (tmp - self.min_value_exponent)

        tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff,
                      lambda: tmp / self.maybe_inversed_step_diff)
        # tmp = tf.round(tmp)
        tmp = tf.floor(tmp + 0.5)

        tf.cond(self.print_log_num, lambda: self.print_log2(tmp, "FxP expo steps"), lambda: 0.0)

        log2_fixed = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
                             lambda: tmp * self.maybe_inversed_step_diff) #+ self.min_value_exponent
        return log2_fixed, input_signs

    def convertToNormal(self, log2_fixed, input_signs):
        #tf.cond(self.print_log_num, lambda: self.print_log2(log2_fixed, "FxP expo"), lambda: 0.0)
        #tmp = tf.where(input_zeros, self.min_value_exponent, log2_fixed)  # All values with zero are encoded to the smallest value
        tmp = log2_fixed
        tmp = tf.clip_by_value(tmp, self.min_value_exponent, self.max_value_exponent)
        input_zeros = tf.where(tmp == self.min_value_exponent, True, False)  # All values with zero are encoded to the smallest value
        tmp = tf.pow(tf.cast(self.base, dtype=self.dtype), tmp)  # reverse the log2(inputs) by calculating 2^(log2(inputs))

        tmp = tf.cond(self.has_signed_bit, lambda: tmp * input_signs, lambda: tmp)
        tmp = tf.where(input_zeros, tf.cast(0.0, dtype=self.dtype), tmp)
        return tmp

    def convertToNormalExtra(self, log2_fixed, input_signs):
        #tf.cond(self.print_log_num, lambda: self.print_log2(log2_fixed, "FxP expo"), lambda: 0.0)
        #tmp = tf.where(input_zeros, self.min_value_exponent, log2_fixed)  # All values with zero are encoded to the smallest value
        tmp = log2_fixed
        tmp = tf.clip_by_value(tmp, self.min_value_exponent, self.max_value_exponent)
        was_clipped = tf.where(tmp == log2_fixed, False, True)
        input_zeros = tf.where(tmp == self.min_value_exponent, True, False)  # All values with zero are encoded to the smallest value
        tmp = tf.pow(tf.cast(self.base, dtype=self.dtype),
                     tmp) * input_signs  # reverse the log2(inputs) by calculating 2^(log2(inputs))
        tmp = tf.where(input_zeros, tf.cast(0.0, dtype=self.dtype), tmp)
        return tmp, was_clipped

    def quant(self, inputs):
        """quantisation function

        applies the quantization to the input.

        Parameters:
            inputs (list of tensors):
                list of all input tensors.

        Returns:
            (tensor):
                the output of layer.
        """

        @tf.custom_gradient
        # custom function, that modifies the gradient computation in the
        # backward pass
        def _quant(inputs): # how to remove those parameters without errors?
            self.inference_counter.assign_add(1)
            # inputs = tf.cast(inputs, self.dtype) #todo maybe remove this?
            #tf.cond(self.print_log_num, lambda: self.print_log2(inputs, "inputs"), lambda: 0.0)

            unique_inputs = tf.reshape(inputs, [-1])
            unique_inputs = tf.floor(unique_inputs * 1024.0 + 0.5) / 1024.0
            unique_inputs = tf.unique(unique_inputs).y

            tf.cond(tf.logical_and(self.inference_counter <= self.flex_inferences, self.inference_counter >= 0),
                    lambda: self.collect_input_sample(unique_inputs),
                    lambda: 0)
            tf.cond(tf.equal(self.inference_counter, self.flex_inferences),
                    lambda: self.setBitsBeforeAndAfter(),
                    lambda: 0.0)
            if self.debug:
                tf.cond(self.inference_counter < self.flex_inferences,
                        lambda: self.warn_not_quantizing(),
                        lambda: 0)
            log2_fixed, input_signs = self.convertToLNS(inputs)
            tmp = self.convertToNormal(log2_fixed, input_signs)

            # tmp = tf.clip_by_value(inputs, self.min_value_exponent, self.max_value_exponent)
            # tmp = (tmp - self.min_value_exponent)
            # tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff,
            #               lambda: tmp / self.maybe_inversed_step_diff)
            # tmp = tf.floor(tmp + 0.5)
            # tmp = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
            #               lambda: tmp * self.maybe_inversed_step_diff) + self.min_value_exponent

            tmp = tf.cond(self.inference_counter < self.flex_inferences, lambda: inputs, lambda: tmp)
            # if self.debug:
            #     tf.print("LogPointQuant:", self.name, "abs error is", tf.reduce_sum(tf.abs(tmp - inputs))/tf.cast(tf.size(inputs), tf.float32),
            #              "inputs[", tf.reduce_min(inputs), ";", tf.reduce_max(inputs), "]  ",
            #              "q(inputs)[", tf.reduce_min(tmp), ";", tf.reduce_max(tmp), "]")

            #tf.cond(self.print_log_num, lambda: self.print_log2(tmp, "output float"), lambda: 0.0)
            # if self.debug:
            #     tf.print("LogPointQuant:", self.name, "output is", tmp)
            #     tf.print("LogPointQuant:", self.name, "input is", inputs)

            # tf.print(self.name, "stds[chosen] red  control", tf.reduce_sum(tf.abs(inputs - tmp)))

            # define the gradient calculation
            @tf.function
            def grad(dy):
                # test for every element of a if it is out of the bounds of
                # the quantisation range
                if self.has_signed_bit:
                    is_out_of_range = tf.logical_or(inputs < -1.0, inputs > 1.0)
                else:
                    is_out_of_range = tf.logical_or(inputs < 0.0, inputs > 1.0)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return tf.where(is_out_of_range, self.leak_clip * dy, dy)

            return tmp, grad

        return _quant(inputs)
