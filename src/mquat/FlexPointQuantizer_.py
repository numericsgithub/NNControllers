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

# TODO Add variable for the self.flex_inferences stuff
# TODO Deactivate the dynamic change of the bits
# TODO Add threshold (TODO count outlier)
# TODO Implement percentile keep stuff...

class FlexPointQuantizer(Quantizer):
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
    #test_summary_writer = tf.summary.create_file_writer('logs/fit/' + "FlexPointQuantizer")

    def __init__(self, name, total_bits, std_keep_factor=0.0, flex_inferences=1, leak_clip=0.0, dtype=DEFAULT_DATATYPE, debug=False, extra_flex_shift=0):
        super().__init__(name, dtype=dtype)
        self.extra_flex_shift = extra_flex_shift
        self.total_bits = to_variable(total_bits, tf.int32)
        self.flex_inferences = to_variable(flex_inferences, tf.int32)
        self.std_keep_factor = to_variable(std_keep_factor, tf.float32)
        self.debug = debug
        self.leak_clip = tf.cast(leak_clip, dtype)
        self.inference_counter = tf.Variable(0, name=self.name+"_inference_counter", trainable=False, dtype=tf.int32)
        self.input_sample = tf.Variable([], name=self.name+"_input_sample", shape=[None], trainable=False, dtype=dtype, validate_shape=False)
        self.maybe_inversed_step_diff = tf.Variable(0, name=self.name+"_maybe_inversed_step_diff", trainable=False, dtype=dtype)
        self.min_value = tf.Variable(0, name=self.name+"_min_value", trainable=False, dtype=dtype)
        self.max_value = tf.Variable(0, name=self.name+"_max_value", trainable=False, dtype=dtype)
        self.b_frc = tf.Variable(0, name=self.name+"_b_frc", trainable=False, dtype=dtype)
        self.b_frc_trend_setter = tf.Variable(0, name=self.name+"_b_frc_trend_setter", trainable=False, dtype=dtype)


    def reset(self):
        """
        Resets the Quantizer. The next inference triggers the coefficient search again.
        Returns:

        """
        self.inference_counter.assign(0)
        self.input_sample.assign([])
        self.maybe_inversed_step_diff.assign(0)
        self.min_value.assign(0)
        self.max_value.assign(0)
        self.b_frc.assign(0)
        #self.b_frc_trend_setter.assign(0)


    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend([self.inference_counter, self.maybe_inversed_step_diff, self.min_value, self.max_value,
                          self.b_frc, self.b_frc_trend_setter])
        return variables


    def collectinputsample(self, inputs):
        tf.cond(tf.reduce_sum(tf.where(inputs > 0, 1, 1)) == 0,
                lambda :self.inference_counter.assign_add(-1),
                lambda :self.inference_counter.assign_add(0))
        self.input_sample.assign(tf.concat([tf.reshape(inputs, [-1]), self.input_sample], 0))
        return 0

    def filterInputs(self, inputs, std_keep_factor):
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


    def warnnotquantizing(self):
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
        if "_b_" in self.name:
            return 0.0
        if "conv_f_fixed_quant" != self.name:
            return 0.0
        time.sleep(random.uniform(0, 1))
        while FlexPointQuantizer.PLOTTING:
            time.sleep(random.uniform(0, 1))
        FlexPointQuantizer.PLOTTING = True
        chosen = chosen.numpy()
        plt.plot(stds, errors, color="blue", label="quant error")
        plt.plot(stds, std_errors, color="orange", label="clip error")
        plt.vlines([stds[chosen]], ymin=0, ymax=np.max(errors), colors=["red"], label="chosen")
        plt.title(self.name + "with " + str(self.total_bits.numpy()) + "bits")
        plt.legend()
        plt.xlabel("std factor")
        plt.ylabel("absolute error")
        print(self.name, "stds[chosen] red", stds[chosen].numpy(), errors[chosen].numpy())
        plt.show()

        FlexPointQuantizer.PLOTTING = False
        return 0.0


    # def setBitsBeforeAndAfter(self):
    #     all_std_filters = tf.range(3.0, 22, 0.2, dtype=tf.float32)
    #     all_error, all_b_int, all_std_errors = tf.map_fn(self.setBitsBeforeAndAfterTestStd, [all_std_filters, tf.zeros_like(all_std_filters, dtype=tf.float32), tf.zeros_like(all_std_filters, dtype=tf.float32)],
    #                                     fn_output_signature=[tf.float32, tf.float32, tf.float32], parallel_iterations=1)
    #     all_error_min = tf.reduce_min(all_error)
    #     all_error_argmin = tf.argmin(all_error)
    #     #best_index = tf.reduce_max(tf.where(all_error == all_error_min))
    #     # all_error_cross = tf.where(all_std_errors <= all_error, all_error, tf.reduce_max(all_error))
    #     # all_error_cross = tf.argmin(all_error_cross)
    #     best_index = all_error_argmin
    #
    #     best_std = tf.reshape(tf.gather(all_std_filters, [best_index]), ())
    #     if self.debug:
    #         tf.print("FlexPointQuant: ", self.name, "std search got lowest error ", tf.gather(all_error, best_index), "with std filter of", tf.gather(all_std_filters, best_index))
    #     #error, best_b_int = self.setBitsBeforeAndAfterTestStd([1.0, 0.0])
    #
    #     input_sample_size = tf.reduce_sum(tf.where(self.input_sample > 0, 1.0, 1.0))
    #
    #     tf.py_function(func=self.setBitsPlot,
    #                    inp=[all_std_filters, all_error ,
    #                         best_index, all_std_errors , ]
    #                    , Tout=[tf.float32])
    #     self.std_keep_factor.assign(best_std)
    #     self.setBitsBeforeAndAfter_old()
    #     return 0
    #
    # @tf.function()
    # def setBitsBeforeAndAfterTestStd(self, std_keep_factor):
    #     std_keep_factor = std_keep_factor[0]
    #     inputs = self.input_sample
    #     inputs = tf.reshape(inputs, [-1])
    #
    #     # if self.debug:
    #     #     tf.print("FlexPointQuant:", self.name, "all samples unique inputs",inputs)
    #     inputs = self.__filterInputs(inputs, std_keep_factor)
    #     #inputs = clipped
    #     if self.debug:
    #         tf.print("FlexPointQuant:", self.name, "all kept unique inputs", tf.unique(inputs)[0])
    #     int_frac_extension = tf.cast(FlexPointQuantizer.INT_FRAC_EXTENSION, dtype=tf.float64)
    #     beta = tf.cast(1.00, dtype=tf.float64)
    #
    #     t = tf.cast(self.total_bits, tf.float64)
    #     T = tf.cast(tf.range(1 - int_frac_extension, t + 1 + int_frac_extension), dtype=tf.float64)
    #     S = tf.cast(inputs, tf.float64)
    #     _05 = tf.cast(0.5, dtype=tf.float64)
    #     _1 = tf.cast(1, dtype=tf.float64)
    #     _2 = tf.cast(2, dtype=tf.float64)
    #
    #
    #     q_low = tf.where((beta * -tf.pow(_2, T - _1)) <= tf.reduce_min(S))
    #     q_low = tf.reshape(tf.gather(T, q_low), [-1])
    #     q_low = tf.concat([[t + int_frac_extension], q_low], axis=0)
    #
    #     q_high = tf.where((beta * (tf.pow(_2, T - _1) - tf.pow(_05, tf.cast(t - T, dtype=tf.float64)))) >= tf.reduce_max(S))
    #     q_high = tf.reshape(tf.gather(T, q_high), [-1])
    #     q_high = tf.concat([[t + int_frac_extension], q_high], axis=0)
    #
    #     b_int = tf.maximum(tf.reduce_min(q_low), tf.reduce_min(q_high))
    #     b_frc = t - b_int
    #     min = -tf.pow(_2, b_int - _1)
    #     max = tf.pow(_2, b_int - _1) - tf.pow(_05, t - b_int)
    #     inv_step = tf.cast(tf.pow(tf.cast(2,dtype=tf.int64),tf.cast(tf.abs(b_frc),dtype=tf.int64)),dtype=tf.float64)
    #     # self.maybe_inversed_step_diff.assign(tf.cast(inv_step, dtype=self.dtype))
    #     # self.min_value.assign(tf.cast(min, dtype=self.dtype))
    #     # self.max_value.assign(tf.cast(max, dtype=self.dtype))
    #     # self.b_frc.assign(tf.cast(b_frc, dtype=tf.float32))
    #     # self.input_sample.assign([])
    #     self_maybe_inversed_step_diff = tf.cast(inv_step, dtype=self.dtype)
    #     self_min_value = tf.cast(min, dtype=self.dtype)
    #     self_max_value = tf.cast(max, dtype=self.dtype)
    #     self_b_frc = tf.cast(b_frc, dtype=tf.float32)
    #
    #     tmp = tf.clip_by_value(self.input_sample, self_min_value, self_max_value)
    #     tmp = (tmp - self_min_value)
    #     tmp = tf.cond(self_b_frc >= 0, lambda: tmp * self_maybe_inversed_step_diff, lambda: tmp / self_maybe_inversed_step_diff)
    #     tmp = tf.floor(tmp + 0.5)
    #     tmp = tf.cond(self_b_frc >= 0, lambda: tmp / self_maybe_inversed_step_diff, lambda: tmp * self_maybe_inversed_step_diff) + self_min_value
    #     # # the check for numerical errors works by converting back the quantized number to an integer number
    #     # # quantized * step_diff == integer number
    #     # # checker = quantized * step_diff
    #     # # checker_int = int(checker)
    #     # # checker_int != checker -> numerical error!
    #     # checker = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff, lambda: tmp / self.maybe_inversed_step_diff)
    #     # checker_int = tf.cast(tf.cast(checker, tf.int64), dtype=tf.float32)
    #     # tf.print("FlexPointQuant:", self.name, "checking sample for numerical errors", inputs, tmp, self.maybe_inversed_step_diff)
    #     # tf.print("FlexPointQuant:", self.name, "checking sample for numerical errors", checker_int, checker)
    #     # tf.print("FlexPointQuant:", self.name, "checking sample for numerical errors", tf.reduce_all(tf.math.equal(checker_int, checker)))
    #     # tf.cond(tf.reduce_all(tf.math.equal(checker_int, checker)),
    #     #         lambda: 0,
    #     #         lambda: self.FatalNumericalError())
    #     # tf.cond(tf.logical_and(tf.abs(b_frc) < 64, (self.total_bits + self.INT_FRAC_EXTENSION) < 64),
    #     #         lambda: 0,
    #     #         lambda: self.FatalNumericalError())
    #     # tf.cond(tf.logical_and(tf.abs(b_frc) < 25, (self.total_bits + self.INT_FRAC_EXTENSION) < 25),
    #     #         lambda: 0,
    #     #         lambda: self.WarningNumericalError())
    #
    #     if self.debug:
    #         tf.print("FlexPointQuant: ", self.name, "setting bits based on sample size",
    #                  tf.reduce_sum(tf.where(inputs > 0, 1, 1)), "[", tf.reduce_min(inputs), ";", tf.reduce_max(inputs), "]",
    #                  " bits set to", b_int, b_frc, "[", min, ";", max, "]")
    #         tf.print("FlexPointQuant: ", self.name, "quantized sample with unique is","[", tf.reduce_min(tmp), ";", tf.reduce_max(tmp), "]", tf.unique(tmp))
    #     error = tf.cast(tf.reduce_sum(tf.pow(self.input_sample - tmp, 2)), dtype=tf.float32)
    #     std_error = tf.reduce_sum(tf.pow(self.input_sample - inputs, 2))
    #     return [error, tf.cast(b_int, dtype=tf.float32), std_error]

    @tf.autograph.experimental.do_not_convert
    def setBitsBeforeAndAfter(self):
        all_std_filters = tf.range(3.0, 10, 0.2, dtype=tf.float32)
        all_clipping_error, all_rounding_errors = tf.map_fn(self.setBitsBeforeAndAfterstd,
                                                            [all_std_filters],
                                                            fn_output_signature=[tf.float32, tf.float32], parallel_iterations=1)
        all_errors = all_clipping_error + all_rounding_errors
        selected = tf.argmin(all_errors)
        self.std_keep_factor = all_std_filters[selected]
        tf.py_function(func=self.setBitsPlot,  # stds, errors, chosen, std_errors
                           inp=[all_std_filters, all_rounding_errors,
                                selected, all_clipping_error],
                           Tout=[tf.float32])


        org_inputs = self.input_sample
        org_inputs = tf.reshape(org_inputs, [-1])
        clipped_inputs = self.filterInputs(org_inputs, self.std_keep_factor)
        if self.debug:
            tf.print("FlexPointQuant:", self.name, "all kept unique inputs", tf.unique(clipped_inputs)[0])
        tmp = tf.clip_by_value(clipped_inputs, self.min_value, self.max_value)
        tmp = (tmp - self.min_value)
        tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff,
                      lambda: tmp / self.maybe_inversed_step_diff)
        tmp = tf.floor(tmp + 0.5)
        quant_input = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
                              lambda: tmp * self.maybe_inversed_step_diff) + self.min_value

        direct_hit = tf.reduce_sum(tf.where(quant_input == tmp, 1, 0))
        sample_size = tf.reduce_sum(tf.where(quant_input > 0, 1, 1))
        tf.cond(tf.logical_and(tf.abs(self.b_frc) < 64, (self.total_bits + self.INT_FRAC_EXTENSION) < 64),
                lambda: 0,
                lambda: self.FatalNumericalError())
        tf.cond(tf.logical_and(tf.abs(self.b_frc) < 24, (self.total_bits + self.INT_FRAC_EXTENSION) < 24),
                lambda: 0,
                lambda: self.WarningNumericalError())

        tf.print("FlexPointQuant:", self.name, "got", sample_size, "weights with", direct_hit, "direct matches (",
                 (direct_hit / sample_size) * 100.0, "%)")

        # FlexPointQuantizer.error_statistic.assign(tf.concat([tf.reshape(inputs - tmp, [-1]), FlexPointQuantizer.error_statistic], 0))

        if self.debug:
            tf.print("FlexPointQuant:", self.name, "setting bits based on sample size",
                     sample_size, "[", tf.reduce_min(org_inputs), ";", tf.reduce_max(org_inputs), "]",
                     " bits set to", self.b_frc + tf.cast(self.total_bits, tf.float32), self.b_frc, "[", self.min_value, ";", self.max_value, "]")
            tf.print("FlexPointQuant:", self.name, "quantized sample with unique is", "[", tf.reduce_min(quant_input), ";",
                     tf.reduce_max(quant_input), "]", tf.unique(quant_input))

        self.input_sample.assign([])
        return 0.0


    def setBitsBeforeAndAfterstd(self, std_filter):
        org_inputs = self.input_sample
        org_inputs = tf.reshape(org_inputs, [-1])
        #org_inputs = org_inputs/tf.reduce_max(tf.abs(org_inputs))

        # if self.debug:
        #     tf.print("FlexPointQuant:", self.name, "all samples unique inputs",inputs)
        clipped_inputs = tf.cond(std_filter != tf.constant(0.0), lambda: self.filterInputs(org_inputs, std_filter), lambda: org_inputs)
        clipping_error = tf.reduce_sum(tf.abs(clipped_inputs - org_inputs))

        int_frac_extension = tf.cast(FlexPointQuantizer.INT_FRAC_EXTENSION, dtype=tf.float64)
        beta = tf.cast(1.05, dtype=tf.float64)

        t = tf.cast(self.total_bits, tf.float64)
        T = tf.cast(tf.range(1 - int_frac_extension, t + 1 + int_frac_extension), dtype=tf.float64)
        S = tf.cast(clipped_inputs, tf.float64)
        _05 = tf.cast(0.5, dtype=tf.float64)
        _1 = tf.cast(1, dtype=tf.float64)
        _2 = tf.cast(2, dtype=tf.float64)

        q_low = tf.where((beta * -tf.pow(_2, T - _1)) <= tf.reduce_min(S))
        q_low = tf.reshape(tf.gather(T, q_low), [-1])
        q_low = tf.concat([[t + int_frac_extension], q_low], axis=0)

        q_high = tf.where(
            (beta * (tf.pow(_2, T - _1) - tf.pow(_05, tf.cast(t - T, dtype=tf.float64)))) >= tf.reduce_max(S))
        q_high = tf.reshape(tf.gather(T, q_high), [-1])
        q_high = tf.concat([[t + int_frac_extension], q_high], axis=0)

        b_int = tf.maximum(tf.reduce_min(q_low), tf.reduce_min(q_high))
        b_frc = t - b_int
        min = -tf.pow(_2, b_int - _1)
        max = tf.pow(_2, b_int - _1) - tf.pow(_05, t - b_int)
        inv_step = tf.cast(tf.pow(tf.cast(2, dtype=tf.int64), tf.cast(tf.abs(b_frc), dtype=tf.int64)), dtype=tf.float64)
        self.maybe_inversed_step_diff.assign(tf.cast(inv_step, dtype=self.dtype))
        self.min_value.assign(tf.cast(min, dtype=self.dtype))
        self.max_value.assign(tf.cast(max, dtype=self.dtype))
        self.b_frc.assign(tf.cast(b_frc, dtype=tf.float32))

        # quant_factor = tf.reduce_max(tf.abs(clipped_inputs))
        # tmp = clipped_inputs / quant_factor
        # tmp = tmp * 128.0
        # quant_input = tf.round(tmp)
        # quant_input = quant_input / 128.0
        # quant_input = quant_input * quant_factor

        tmp = tf.clip_by_value(clipped_inputs, self.min_value, self.max_value)
        tmp = (tmp - self.min_value)
        tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff,
                      lambda: tmp / self.maybe_inversed_step_diff)
        tmp = tf.floor(tmp + 0.5)
        quant_input = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
                      lambda: tmp * self.maybe_inversed_step_diff) + self.min_value

        rounding_errors = tf.reduce_sum(tf.abs(quant_input - clipped_inputs))

        # tmp = tf.clip_by_value(inputs, self.min_value, self.max_value)
        # tmp = (tmp - self.min_value)
        # tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff,
        #               lambda: tmp / self.maybe_inversed_step_diff)
        # tmp = tf.floor(tmp + 0.5)
        # tmp = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
        #               lambda: tmp * self.maybe_inversed_step_diff) + self.min_value

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
        # #FlexPointQuantizer.error_statistic.assign(tf.concat([tf.reshape(inputs - tmp, [-1]), FlexPointQuantizer.error_statistic], 0))
        #
        # if self.debug:
        #     tf.print("FlexPointQuant:", self.name, "setting bits based on sample size",
        #              sample_size, "[", tf.reduce_min(org_inputs), ";", tf.reduce_max(org_inputs), "]",
        #              " bits set to", b_int, b_frc, "[", min, ";", max, "]")
        #     tf.print("FlexPointQuant:", self.name, "quantized sample with unique is", "[", tf.reduce_min(tmp), ";",
        #              tf.reduce_max(tmp), "]", tf.unique(tmp))
        return [clipping_error, rounding_errors]


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
        def _quant(inputs):
            self.inference_counter.assign_add(1)

            tf.cond(tf.logical_and(self.inference_counter <= self.flex_inferences, self.inference_counter >= 0, name="should_collect_inputs_internal"),
                    lambda: self.collectinputsample(inputs),
                    lambda: 0, name="should_collect_inputs")
            tf.cond(tf.equal(self.inference_counter, self.flex_inferences, name="should_set_bits"),
                    lambda: self.setBitsBeforeAndAfter(),
                    lambda: 0.0)
            if self.debug:
                tf.cond(self.inference_counter < self.flex_inferences,
                        lambda: self.warnnotquantizing(),
                        lambda: 0, name="should_warn_not_quantizing")

            tmp = tf.clip_by_value(inputs, self.min_value, self.max_value)
            tmp = (tmp - self.min_value)
            tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff, lambda: tmp / self.maybe_inversed_step_diff, name="b_frac_diff1")
            tmp = tf.floor(tmp + 0.5)

            # with FlexPointQuantizer.test_summary_writer.as_default():
            #     step = tf.cast(self.inference_counter, tf.int64)
            #     tf.summary.object(self.name.replace("/","_")+"/decayed amount mins", tf.reduce_sum(tf.where(tmp == tf.reduce_min(tmp), 1, 0)), step=step)
            #     tf.summary.object(self.name.replace("/","_")+"/decayed amount maxs", tf.reduce_sum(tf.where(tmp == tf.reduce_max(tmp), 1, 0)), step=step)
            #     tf.summary.object(self.name.replace("/","_")+"/decayed at min", tf.cond(self.b_frc >= 0, lambda: tf.reduce_min(tmp) / self.maybe_inversed_step_diff, lambda: tf.reduce_min(tmp) * self.maybe_inversed_step_diff) + self.min_value, step=step)
            #     tf.summary.object(self.name.replace("/","_")+"/decayed at max", tf.cond(self.b_frc >= 0, lambda: tf.reduce_max(tmp) / self.maybe_inversed_step_diff, lambda: tf.reduce_max(tmp) * self.maybe_inversed_step_diff) + self.min_value, step=step)
            #     self.test_summary_writer.flush()
            # with self.test_summary_writer.as_default():
            #     tf.summary.object(self.name+"test", self.inference_counter, step=1)
            #     tf.summary.object(self.name+"test2", self.inference_counter, step=tf.cast(self.inference_counter, tf.int64))
            #     tf.summary.object(self.name+"decayed amount mins", tf.reduce_sum(tf.where(tmp == tf.reduce_min(tmp), 1, 0)), step=1)
            #     tf.summary.object(self.name+"decayed amount maxs", tf.reduce_sum(tf.where(tmp == tf.reduce_max(tmp), 1, 0)), step=tf.cast(self.inference_counter, tf.int64))
            #     tf.summary.object(self.name+"decayed at min", tf.cond(self.b_frc >= 0, lambda: tf.reduce_min(tmp) / self.maybe_inversed_step_diff, lambda: tf.reduce_min(tmp) * self.maybe_inversed_step_diff) + self.min_value, step=tf.cast(self.inference_counter, tf.int64))
            #     tf.summary.object(self.name+"decayed at max", tf.cond(self.b_frc >= 0, lambda: tf.reduce_max(tmp) / self.maybe_inversed_step_diff, lambda: tf.reduce_max(tmp) * self.maybe_inversed_step_diff) + self.min_value, step=tf.cast(self.inference_counter, tf.int64))
            #     self.test_summary_writer.flush()

            # error_without_selwei = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff, lambda: tmp * self.maybe_inversed_step_diff) + self.min_value
            # error_without_selwei = tf.reduce_sum(tf.abs(error_without_selwei - inputs))
            # allowed_selwei_error = error_without_selwei * 0.25
            # allowed_selwei_steps = allowed_selwei_error / tf.cond(self.b_frc >= 0, lambda: 1 / self.maybe_inversed_step_diff, lambda: 1 * self.maybe_inversed_step_diff)
            # allowed_selwei_steps = tf.round(allowed_selwei_steps)
            # tf.print("FlexPointQuant:", self.name, "total error", error_without_selwei, "allowed selwei error to add", allowed_selwei_error, "allowed selwei steps", allowed_selwei_steps)
            # all_roundings = tf.range(0, tf.round((tf.reduce_max(tmp) - tf.reduce_min(tmp))/4.0)) # Nur 1/4 aller quantisierten Werte kann gerundet werden
            #                                                                                      # 1/4 im Sinne von 1/4 aller einzigartigen Werte können um 1 * step_size verschoben werden
            # all_round_down_steps = tf.map_fn(fn=lambda cur_steps: tf.reduce_sum(tf.where(tmp >= tf.reduce_max(tmp)-cur_steps, 1.0, 0.0)),
            #                                  elems=[all_roundings], dtype=tf.float32)
            # tf.print("FlexPointQuant:", self.name, "all_roundings", all_roundings, "all_round_down_steps", all_round_down_steps)


            ##selective quant weight decay
            # tmp = tf.where(tmp == tf.reduce_min(tmp), tmp + 1, tmp) # the smallest values get 1 step_div bigger
            # tmp = tf.where(tmp == tf.reduce_max(tmp), tmp - 1, tmp) # the biggest values get 1 step_div smaller

            tmp = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff, lambda: tmp * self.maybe_inversed_step_diff, name="b_frac_diff2") + self.min_value
            tmp = tf.cond(self.inference_counter < self.flex_inferences, lambda: inputs, lambda: tmp, name="should_use_quantized")

            # tf.print(self.name, "stds[chosen] red  control", tf.reduce_sum(tf.abs(inputs - tmp)))

            # define the gradient calculation
            @tf.function
            def grad(dy):
                # test for every element of a if it is out of the bounds of
                # the quantisation range
                is_out_of_range = tf.logical_or(inputs < self.min_value, inputs > self.max_value)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return tf.where(is_out_of_range, self.leak_clip * dy, dy)

            return tmp, grad

        return _quant(inputs)
