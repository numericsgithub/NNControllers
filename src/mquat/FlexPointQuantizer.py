# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import Variable
import numpy as np

#from . import Logging
from .Utilities import to_variable, to_value
from .Quantizer import Quantizer
import random
import time
from .QuantizerBase import DEFAULT_DATATYPE

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

    INT_FRAC_EXTENSION = 7
    PLOTTING = False

    def __init__(self, name, total_bits, leak_clip=0.0, dtype=DEFAULT_DATATYPE, debug=False, extra_flex_shift=0, set_b_int=None, is_symmetric=False, pre_filters=[], post_filters=[]):
        super().__init__(name, dtype=dtype, pre_filters=pre_filters, post_filters=post_filters)
        self.extra_flex_shift = extra_flex_shift
        self.total_bits = to_variable(total_bits, tf.int32)
        self.debug = debug
        self.leak_clip = tf.cast(leak_clip, dtype)
        self.maybe_inversed_step_diff = tf.Variable(0, name=self.name+"_maybe_inversed_step_diff", trainable=False, dtype=dtype)
        self.min_value = tf.Variable(0, name=self.name+"_min_value", trainable=False, dtype=dtype)
        self.max_value = tf.Variable(0, name=self.name+"_max_value", trainable=False, dtype=dtype)
        self.b_frc = tf.Variable(0, name=self.name+"_b_frc", trainable=False, dtype=dtype)
        self.b_int_override = set_b_int
        self.is_symmetric = is_symmetric
        if set_b_int is not None:
            self.b_int_override = tf.constant(self.b_int_override, dtype=tf.float64)

    def reset(self):
        """
        Resets the Quantizer. The next inference triggers the coefficient search again.
        Returns:

        """
        self.maybe_inversed_step_diff.assign(0)
        self.min_value.assign(0)
        self.max_value.assign(0)
        self.b_frc.assign(0)
        # self.setBitsBeforeAndAfter(tf.constant([-8, -4, -2, -1, 0, 1, 2, 4, 8], dtype=tf.float32) / 32.0)
        #self.b_frc_trend_setter.assign(0)


    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend([self.maybe_inversed_step_diff, self.min_value, self.max_value,
                          self.b_frc])
        return variables

    def getCoeffs(self):
        step_size = tf.cond(self.b_frc >= 0, lambda: 1.0 / self.maybe_inversed_step_diff, lambda: 1.0 * self.maybe_inversed_step_diff)
        coeffs = tf.range(self.min_value, self.max_value + step_size, step_size)
        # tf.print("coeffs asd", coeffs, self.min_value, self.max_value, step_size)
        return coeffs

    def getCoeffsScaling(self):
        return (1.0 / tf.reduce_max(tf.abs(self.getCoeffs()))) * tf.pow(2.0, float(self.total_bits) - 1.0)

    def isNonUniform(self):
        return False

    @tf.function(autograph=False)
    def setBitsBeforeAndAfter(self, input_sample):
        inputs = input_sample
        inputs = tf.reshape(inputs, [-1])

        int_frac_extension = tf.cast(FlexPointQuantizer.INT_FRAC_EXTENSION, dtype=tf.float64)
        beta = tf.cast(1.00, dtype=tf.float64)

        t = tf.cast(self.total_bits, tf.float64)
        T = tf.cast(tf.range(1 - int_frac_extension, t + 1 + int_frac_extension), dtype=tf.float64)
        S = tf.cast(inputs, tf.float64)
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
        if self.b_int_override is not None:
            b_int = self.b_int_override
        b_frc = t - b_int
        min = -tf.pow(_2, b_int - _1)
        if not self.is_symmetric:
            max = tf.pow(_2, b_int - _1) - tf.pow(_05, t - b_int)
        else:
            max = -min
        inv_step = tf.cast(tf.pow(tf.cast(2, dtype=tf.int64), tf.cast(tf.abs(b_frc), dtype=tf.int64)), dtype=tf.float64)
        self.maybe_inversed_step_diff.assign(tf.cast(inv_step, dtype=self.dtype))
        self.min_value.assign(tf.cast(min, dtype=self.dtype))
        self.max_value.assign(tf.cast(max, dtype=self.dtype))
        self.b_frc.assign(tf.cast(b_frc, dtype=tf.float32))

        tmp = tf.clip_by_value(inputs, self.min_value, self.max_value)
        tmp = (tmp - self.min_value)
        tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff,
                      lambda: tmp / self.maybe_inversed_step_diff)
        tmp = tf.floor(tmp + 0.5)
        tmp = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
                      lambda: tmp * self.maybe_inversed_step_diff) + self.min_value
        # tf.cond(tf.logical_and(tf.abs(b_frc) < 64, (self.total_bits + self.INT_FRAC_EXTENSION) < 64),
        #         lambda: 0,
        #         lambda: self.FatalNumericalError())
        # tf.cond(tf.logical_and(tf.abs(b_frc) < 24, (self.total_bits + self.INT_FRAC_EXTENSION) < 24),
        #         lambda: 0,
        #         lambda: self.WarningNumericalError())

        # direct_hit = tf.reduce_sum(tf.where(inputs == tmp, 1, 0))
        # # Logging.Log(self.name + "_quant_inputs", tmp)
        # # Logging.Log(self.name + "_direct_hits", direct_hit)
        # sample_size = tf.reduce_sum(tf.where(inputs > 0, 1, 1))
        # tf.print("FlexPointQuant:", self.name, "got", sample_size, "weights with", direct_hit, "direct matches (",(direct_hit/sample_size)*100.0,"%)", "[", min, ";", max, "]")

        #FlexPointQuantizer.error_statistic.assign(tf.concat([tf.reshape(inputs - tmp, [-1]), FlexPointQuantizer.error_statistic], 0))

        # if self.debug:
        #     tf.print("FlexPointQuant:", self.name, "setting bits based on sample size",
        #              tf.reduce_sum(tf.where(inputs > 0, 1, 1)), "[", tf.reduce_min(inputs), ";", tf.reduce_max(inputs), "]",
        #              " bits set to", b_int, b_frc, "[", min, ";", max, "]")
        #     tf.print("FlexPointQuant:", self.name, "quantized sample with unique is", "[", tf.reduce_min(tmp), ";",
        #              tf.reduce_max(tmp), "]", tf.unique(tmp))
        return 0

    @tf.function(jit_compile=True)
    def quant_forward(self, f_inputs):
        tmp = tf.clip_by_value(f_inputs, self.min_value, self.max_value)
        tmp = (tmp - self.min_value)
        tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff,
                      lambda: tmp / self.maybe_inversed_step_diff)
        tmp = tf.floor(tmp + 0.5)

        tmp = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
                      lambda: tmp * self.maybe_inversed_step_diff) + self.min_value
        return tmp

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
        # tf.stop_gradient(tf.cond(self.min_value == self.max_value,
        #                          lambda: self.setBitsBeforeAndAfter(inputs),
        #                          lambda: 0))
        # tf.cond(self.min_value == self.max_value,
        #         lambda: self.setBitsBeforeAndAfter(inputs),
        #         lambda: 0)
        #
        # tmp = self.quant_forward(inputs) # tf.stop_gradient(self.quant_forward(inputs))
        # return tmp
        # #return tf.stop_gradient(self.quant_forward(inputs))
        # custom function, that modifies the gradient computation in the
        # backward pass
        @tf.custom_gradient
        def _quant(inputs): # how to remove those parameters without errors?
            tmp = inputs
            # for pre_filter in self.pre_filters:
            #     tmp = pre_filter(tmp, inputs)
            tf.cond(self.min_value == self.max_value,
                    lambda: self.setBitsBeforeAndAfter(tmp),
                    lambda: 0)

            tmp = self.quant_forward(tmp)

            # for post_filters in self.post_filters:
            #     tmp = post_filters(tmp, inputs)

            # define the gradient calculation
            def grad(dy, variables=None):
                # test for every element of a if it is out of the bounds of
                # the quantisation range
                is_out_of_range = tf.logical_or(inputs < self.min_value, inputs > self.max_value)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return tf.where(is_out_of_range, self.leak_clip * dy, dy)

            return tmp, grad

        return _quant(inputs)
