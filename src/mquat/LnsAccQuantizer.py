# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import Variable
import numpy as np

from .Utilities import to_variable, to_value
from .Quantizer import Quantizer
from .LogPointQuantizer import LogPointQuantizer
from .LnsQuantizer import LnsQuantizer
import random
import time
from .QuantizerBase import DEFAULT_DATATYPE


class LnsAccQuantizer(Quantizer):
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
    def __init__(self, name, msb, lsb, exp_lsb, is_last_layer, std_keep_factor=0.0, leak_clip=0.0,
                 dtype=DEFAULT_DATATYPE,
                 debug=False, extra_flex_shift=0, print_log_num=False, preset_b_int=4):
        super().__init__(name, dtype=dtype)
        self.std_keep_factor = to_variable(std_keep_factor, tf.float32)
        self.debug = debug
        self.leak_clip = tf.cast(leak_clip, dtype)
        self.msb = msb
        self.lsb = lsb
        self.exp_lsb = exp_lsb
        self.inputLogSize = msb - lsb + 1
        self.is_last_layer = is_last_layer
        # self.q_sum_after_all = LogPointQuantizer(name+"fc1_quant_in_log_quant", 4, bits_left_of_comma=3, has_signed_bit=False)
        self.q_sum_out = LnsQuantizer("mynewlog", int(self.msb), int(self.lsb), int(self.exp_lsb), "w")


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
        # self.b_frc_trend_setter.assign(0)

    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend([])
        return variables
    #
    # def xlin2log(self, x):
    #     mask = tf.cast((1 << self.inputLogSize) - 1, dtype=tf.float32)
    #     shifted_log_part1 = tf.pow(2.0, -self.lsb)  # (2**-self.lsb)
    #     shifted_log_part2 = tf.math.log(tf.abs(x)) / tf.math.log(2.0)  # math.log2(abs(w))
    #     shifted_log = tf.floor(-shifted_log_part2 * shifted_log_part1 + 0.5)
    #     # shifted_log = int(round(-math.log2(abs(w)) * (2**-self.lsb)))
    #     logw = tf.minimum(shifted_log, mask)  # min(shifted_log, mask)
    #     result = logw
    #     return tf.where(x == 0, mask, result)
    #
    # def xlin2log2lin(self, x):
    #     mask = tf.cast((1 << self.inputLogSize) - 1, dtype=tf.float32)
    #     shifted_log_part1 = tf.pow(2.0, -self.lsb)  # (2**-self.lsb)
    #     shifted_log_part2 = tf.math.log(tf.abs(x)) / tf.math.log(2.0)  # math.log2(abs(w))
    #     shifted_log = tf.floor(-shifted_log_part2 * shifted_log_part1 + 0.5)
    #     # shifted_log = int(round(-math.log2(abs(w)) * (2**-self.lsb)))
    #     logw = tf.minimum(shifted_log, mask)  # min(shifted_log, mask)
    #     result = logw
    #     lx = tf.where(x == 0, mask, result)
    #
    #     lx = tf.floor(lx)
    #     shifted_lx = -lx * tf.pow(2.0, self.lsb)
    #     lin = tf.pow(2.0, shifted_lx)
    #     lin = tf.where(x == 0, 0.0, lin)
    #     return lin
    #
    # def xlog2lin(self, lx):
    #     lx = tf.floor(lx)
    #     shifted_lx = -lx * tf.pow(2.0, self.lsb)
    #     return tf.pow(2.0, shifted_lx)
    #
    # def wlin2log(self, w):
    #     mask = tf.cast((1 << self.inputLogSize) - 1, dtype=tf.float32)
    #     # if w == 0:
    #     #     return mask
    #     neg = tf.where(w < 0.0, tf.pow(2.0, self.inputLogSize), 0.0)  # (w < 0) << self.inputLogSize
    #     shifted_log_part1 = tf.pow(2.0, -self.lsb)  # (2**-self.lsb)
    #     shifted_log_part2 = tf.math.log(tf.abs(w)) / tf.math.log(2.0)  # math.log2(abs(w))
    #     shifted_log = tf.floor(-shifted_log_part2 * shifted_log_part1 + 0.5)
    #     # shifted_log = int(round(-math.log2(abs(w)) * (2**-self.lsb)))
    #     logw = tf.minimum(shifted_log, mask)  # min(shifted_log, mask)
    #     # tf.print("img wlin2logTEST", logw, summarize=-1)
    #     # tf.print("img neg", neg, summarize=-1)
    #     result = tf.where(tf.logical_and(w < 0, logw >= 0), logw + neg, logw)  # neg | logw
    #     result = tf.where(w == 0, mask, result)
    #     return result
    #
    # def wlin2log2lin(self, w):
    #     mask = tf.cast((1 << self.inputLogSize) - 1, dtype=tf.float32)
    #     # if w == 0:
    #     #     return mask
    #     neg = tf.where(w < 0.0, tf.pow(2.0, self.inputLogSize), 0.0)  # (w < 0) << self.inputLogSize
    #     shifted_log_part1 = tf.pow(2.0, -self.lsb)  # (2**-self.lsb)
    #     shifted_log_part2 = tf.math.log(tf.abs(w)) / tf.math.log(2.0)  # math.log2(abs(w))
    #     shifted_log = tf.floor(-shifted_log_part2 * shifted_log_part1 + 0.5)
    #     # shifted_log = int(round(-math.log2(abs(w)) * (2**-self.lsb)))
    #     logw = tf.minimum(shifted_log, mask)  # min(shifted_log, mask)
    #     # tf.print("img wlin2logTEST", logw, summarize=-1)
    #     # tf.print("img neg", neg, summarize=-1)
    #     result = tf.where(tf.logical_and(w < 0, logw >= 0), logw + neg, logw)  # neg | logw
    #     lw = tf.where(w == 0, mask, result)
    #
    #     lw = tf.floor(lw)  # int(lw)
    #     mask = (1 << self.inputLogSize) - 1
    #     negw = tf.floor(lw / tf.pow(2.0, self.inputLogSize))  # lw >> self.inputLogSize
    #     # tf.print("negw", negw[0], summarize=-1)
    #     logw = -tf.where(negw != 0, lw - tf.cast(1 << self.inputLogSize, dtype=tf.float32) * negw, lw)  # -(lw & mask)
    #     # tf.print("logw2", logw[0], summarize=-1)
    #     logw = logw * tf.pow(2.0, self.lsb)  # float(logw) * 2 ** self.lsb
    #     lin = (1.0 - 2.0 * negw) * (tf.pow(2.0, logw))  # (1 - 2 * negw) * (math.pow(2, logw))
    #     lin = tf.where(w == 0, 0.0, lin)
    #     return lin
    #
    # def wlog2lin(self, lw):
    #     lw = tf.floor(lw)  # int(lw)
    #     mask = (1 << self.inputLogSize) - 1
    #     negw = tf.floor(lw / tf.pow(2.0, self.inputLogSize))  # lw >> self.inputLogSize
    #     # tf.print("negw", negw[0], summarize=-1)
    #     logw = -tf.where(negw != 0, lw - tf.cast(1 << self.inputLogSize, dtype=tf.float32) * negw, lw)  # -(lw & mask)
    #     # tf.print("logw2", logw[0], summarize=-1)
    #     logw = logw * tf.pow(2.0, self.lsb)  # float(logw) * 2 ** self.lsb
    #     w = (1.0 - 2.0 * negw) * (tf.pow(2.0, logw))  # (1 - 2 * negw) * (math.pow(2, logw))
    #     return w
    #
    #     # lw = tf.floor(lw)
    #     # mask = tf.cast((1 << self.inputLogSize) - 1, dtype=tf.float32)
    #     # negw = tf.floor(lw / tf.pow(2.0, self.inputLogSize)) #lw >> self.inputLogSize
    #     # logw = -(tf.where(lw > mask, mask, lw)) #-(lw & mask)
    #     # logw = logw * tf.pow(2.0, self.lsb) #float(logw) * 2 ** self.lsb
    #     # w = ((negw * 2) - 1) * tf.pow(2.0, logw) #(1 - 2 * negw) * (math.pow(2, logw))
    #     # return w

    def tran_back(self, sum, valFilter):
        accFloat = sum * tf.pow(2.0, self.exp_lsb)  # accFloat = ((double)acc) * Math.Pow(2, expLsb); // exact
        # sum_lns = tf.where(accFloat < 0.0, 0.0, accFloat)
        loga = tf.math.log(tf.abs(accFloat)) / tf.math.log(tf.cast(2.0, self.dtype))  # loga = Math.Log2(accFloat); // rounding @lsb=-53
        loga *= -tf.pow(2.0, -self.lsb) # loga *= Math.Pow(2, -lsb);
        a = tf.floor(loga + 0.5) # a = (long)(Math.Round(-loga)); // >0, unsigned, rounding @lsb

        # if (a > valFilter)
        # {
        #     // cout << "Overflow ! Diff is: " << -a - valFilter << endl;
        #     a = valFilter;
        # }
        sum_lns = tf.minimum(a, valFilter)
        return sum_lns


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
        def _quant(inputs):  # how to remove those parameters without errors?
            # self.q_sum_after_all(tf.constant([1.0]))

            acc = inputs
            if not self.is_last_layer:
                # if (acc > ((long)1 << -expLsb)) // High cut
                #   acc = (long)1 << -expLsb;
                upper_limit = tf.cast(1 << -int(self.exp_lsb), dtype=tf.float32)
                inputLogSize = int(self.msb - self.lsb + 1)
                valFilter = tf.cast((1 << inputLogSize) - 1, dtype=tf.float32)
                acc = tf.minimum(acc, upper_limit)
                a = tf.where(acc <= 0.0, valFilter, self.tran_back(acc, valFilter))
                tmp = a
                tmp = self.q_sum_out.xlog2lin(tmp)
                #tmp = tf.where(acc <= 0.0, 0.0, tmp)
                #tmp = tmp / upper_limit
            else:
                acc = tf.where(acc <= 0, 0.0, acc)
                tmp = acc

            # define the gradient calculation
            @tf.function
            def grad(dy):
                # test for every element of a if it is out of the bounds of
                # the quantisation range
                is_out_of_range = tf.logical_or(inputs < -99999.0, inputs > 99999.0)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return tf.where(is_out_of_range, self.leak_clip * dy, dy)

            return tmp, grad

        return _quant(inputs)
