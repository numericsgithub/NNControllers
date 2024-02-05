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



class LnsQuantizer(Quantizer):
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
    test_summary_writer = tf.summary.create_file_writer('logs/fit/' + "LogPointQuantizer")

    def __init__(self, name, msb, lsb, exp_lsb, quant_on_call, std_keep_factor=0.0, leak_clip=0.0, dtype=DEFAULT_DATATYPE,
                 debug=False, extra_flex_shift=0, print_log_num=False, preset_b_int=4):
        super().__init__(name, dtype=dtype)
        self.std_keep_factor = to_variable(std_keep_factor, tf.float32)
        self.debug = debug
        self.leak_clip = tf.cast(leak_clip, dtype)
        self.msb = msb
        self.lsb = lsb
        self.exp_lsb = exp_lsb
        self.inputLogSize = msb - lsb + 1
        self.quant_on_call = quant_on_call
        if quant_on_call == "x":
            self.internalQuant = lambda input: self.xlin2log2lin(input)
        elif quant_on_call == "w":
            self.internalQuant = lambda input: self.wlin2log2lin(input)
        elif quant_on_call == "w_conv":
            self.internalQuant = lambda input: self.wlin2log2lin_for_conv(input)
        else:
            raise Exception("Unknown parameter value for quant_on_call. Should be x or w but is", quant_on_call)

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


    def xlin2log(self, x):
        mask = tf.cast((1 << self.inputLogSize) - 1, dtype=tf.float32)
        shifted_log_part1 = tf.pow(2.0, -self.lsb) # (2**-self.lsb)
        shifted_log_part2 = tf.math.log(tf.abs(x)) / tf.math.log(2.0) # math.log2(abs(w))
        shifted_log = tf.floor(-shifted_log_part2 * shifted_log_part1 + 0.5)
        # shifted_log = int(round(-math.log2(abs(w)) * (2**-self.lsb)))
        logw = tf.minimum(shifted_log, mask) #min(shifted_log, mask)
        result = logw
        return tf.where(x == 0, mask, result)

    def xlin2log2lin(self, x):
        mask = tf.cast((1 << self.inputLogSize) - 1, dtype=tf.float32)
        shifted_log_part1 = tf.pow(2.0, -self.lsb) # (2**-self.lsb)
        shifted_log_part2 = tf.math.log(tf.abs(x)) / tf.math.log(2.0) # math.log2(abs(w))
        shifted_log = tf.floor(-shifted_log_part2 * shifted_log_part1 + 0.5)
        # shifted_log = int(round(-math.log2(abs(w)) * (2**-self.lsb)))
        logw = tf.minimum(shifted_log, mask) #min(shifted_log, mask)
        result = logw
        lx = tf.where(x == 0, mask, result)
        lx = tf.floor(lx)
        shifted_lx = -lx * tf.pow(2.0, self.lsb)
        lin = tf.pow(2.0, shifted_lx)
        #lin = tf.where(x == 0, 0.0, lin)
        return lin

    def xlog2lin(self, lx):
        lx = tf.floor(lx)
        shifted_lx = -lx * tf.pow(2.0, self.lsb)
        return tf.pow(2.0, shifted_lx)

    def wlin2log(self, w):
        mask = tf.cast((1 << self.inputLogSize) - 1, dtype=tf.float32)
        # if w == 0:
        #     return mask
        neg = tf.where(w < 0.0, tf.pow(2.0, self.inputLogSize), 0.0)# (w < 0) << self.inputLogSize
        shifted_log_part1 = tf.pow(2.0, -self.lsb) # (2**-self.lsb)
        shifted_log_part2 = tf.math.log(tf.abs(w)) / tf.math.log(2.0) # math.log2(abs(w))
        shifted_log = tf.floor(-shifted_log_part2 * shifted_log_part1 + 0.5)
        # shifted_log = int(round(-math.log2(abs(w)) * (2**-self.lsb)))
        logw = tf.minimum(shifted_log, mask) #min(shifted_log, mask)
        result = tf.where(tf.logical_and(w < 0, logw >= 0), logw + neg, logw) # neg | logw
        result = tf.where(w == 0, mask, result)
        return result

    def wlin2log2lin_for_conv(self, w):
        sign = tf.sign(w)
        x = tf.math.log(tf.abs(w)) / tf.math.log(2.0)  # translate to log domain
        x = tf.round(x * (2 ** -self.lsb))  # round in our encoding
        max_log = 2.0 ** (self.msb - self.lsb + 1.0) - 1.0
        log_cap = 1.0 - tf.sign(tf.round(tf.abs(x) / (2.0 * max_log)))
        x = tf.pow(2.0, x / (2.0 ** -self.lsb)) * sign * log_cap  # translate back to lin domain
        return x

    def wlin2log2lin(self, w):
        mask = tf.cast((1 << self.inputLogSize) - 1, dtype=tf.float32)
        # if w == 0:
        #     return mask
        neg = tf.where(w < 0.0, tf.pow(2.0, self.inputLogSize), 0.0)# (w < 0) << self.inputLogSize
        shifted_log_part1 = tf.pow(2.0, -self.lsb) # (2**-self.lsb)
        shifted_log_part2 = tf.math.log(tf.abs(w)) / tf.math.log(2.0) # math.log2(abs(w))
        shifted_log = tf.floor(-shifted_log_part2 * shifted_log_part1 + 0.5)
        # shifted_log = int(round(-math.log2(abs(w)) * (2**-self.lsb)))
        logw = tf.minimum(shifted_log, mask) #min(shifted_log, mask)
        #tf.print("img wlin2logTEST", logw, summarize=-1)
        # tf.print("img neg", neg, summarize=-1)
        result = tf.where(tf.logical_and(w < 0, logw >= 0), logw + neg, logw) # neg | logw
        lw = tf.where(w == 0, mask, result)

        lw = tf.floor(lw)  # int(lw)
        mask = (1 << self.inputLogSize) - 1
        negw = tf.floor(lw / tf.pow(2.0, self.inputLogSize))  # lw >> self.inputLogSize
        # tf.print("negw", negw[0], summarize=-1)
        logw = -tf.where(negw != 0, lw - tf.cast(1 << self.inputLogSize, dtype=tf.float32) * negw, lw)  # -(lw & mask)
        # tf.print("logw2", logw[0], summarize=-1)
        logw = logw * tf.pow(2.0, self.lsb)  # float(logw) * 2 ** self.lsb
        lin = (1.0 - 2.0 * negw) * (tf.pow(2.0, logw))  # (1 - 2 * negw) * (math.pow(2, logw))
        lin = tf.where(w == 0, 0.0, lin)
        return lin

    def wlog2lin(self, lw):
        lw = tf.floor(lw) #int(lw)
        mask = (1 << self.inputLogSize) - 1
        negw = tf.floor(lw / tf.pow(2.0, self.inputLogSize)) #lw >> self.inputLogSize
        # tf.print("negw", negw[0], summarize=-1)
        logw = -tf.where(negw != 0, lw - tf.cast(1 << self.inputLogSize, dtype=tf.float32) * negw, lw) #-(lw & mask)
        # tf.print("logw2", logw[0], summarize=-1)
        logw = logw * tf.pow(2.0, self.lsb) #float(logw) * 2 ** self.lsb
        w = (1.0 - 2.0 * negw) * (tf.pow(2.0, logw))#(1 - 2 * negw) * (math.pow(2, logw))
        return w

        # lw = tf.floor(lw)
        # mask = tf.cast((1 << self.inputLogSize) - 1, dtype=tf.float32)
        # negw = tf.floor(lw / tf.pow(2.0, self.inputLogSize)) #lw >> self.inputLogSize
        # logw = -(tf.where(lw > mask, mask, lw)) #-(lw & mask)
        # logw = logw * tf.pow(2.0, self.lsb) #float(logw) * 2 ** self.lsb
        # w = ((negw * 2) - 1) * tf.pow(2.0, logw) #(1 - 2 * negw) * (math.pow(2, logw))
        # return w

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
            tmp = self.internalQuant(inputs)

            # define the gradient calculation
            @tf.function
            def grad(dy):
                # test for every element of a if it is out of the bounds of
                # the quantisation range
                is_out_of_range = tf.logical_or(inputs < -1.0, inputs > 1.0) # TODO is_out_of_range = tf.logical_or(tmp < -1.0, tmp > 1.0)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return tf.where(is_out_of_range, self.leak_clip * dy, dy)

            return tmp, grad

        return _quant(inputs)
