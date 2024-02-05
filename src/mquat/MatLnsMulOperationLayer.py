# -*- coding: utf-8 -*-

import tensorflow as tf

from .FixedToLogQuantizer import FixedToLogQuantizer
from .OperationLayer import OperationLayer
from .QuantizerBase import NON_QUANT
from .FlexPointQuantizer import FlexPointQuantizer
from .LnsQuantizer import LnsQuantizer
from .LogPointQuantizer import LogPointQuantizer
from .QuantizerBase import DEFAULT_DATATYPE


class MatLnsMulOperationLayer(OperationLayer):
    """matrix multiplication operation layer

    the matrix multiplication is performed in two stages, first the multiplication
    is executed, the results are passed through the multiplication quantisizer,
    then results are summed up, while applying the summation quantisizer after
    each addtion.

    Parameters:
        name (string):
            the name of the layer in the Tensorflow graph.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations.
            default is float32.

    Attributes:
        quant_mul (Quantizer):
            quantisizer that is applied after each multiplication.
        quant_sum (Quantizer):
            quantisizer that is applied after each addition.
    """

    def __init__(self, name, msb, lsb, expLsb, splits, is_conv, dtype=DEFAULT_DATATYPE):
        super().__init__(name, "matmul", dtype=dtype)
        self.quant_mul = NON_QUANT
        self.quant_sum = NON_QUANT
        self.msb = msb
        self.lsb = lsb
        self.expLsb = expLsb
        self.q_before_sum = FlexPointQuantizer(name + "_matmul_log_quant", 2 - self.expLsb + 1, symmetric=True, set_b_int=2 - self.expLsb + 1)
        self.q_before_sum_t = 2 - self.expLsb + 1
        #self.q_before_sum = FlexPointQuantizer(name + "_matmul_log_quant", 10, symmetric=True, set_b_int=10)
        self.q_lns = FixedToLogQuantizer(name + "fixed_to_log", self.msb, self.lsb, self.expLsb)
        # self.lns_test = LnsQuantizer(name +"fc1_w_log_quant", self.msb, self.lsb, expLsb, "w")
        self.splits = splits
        self.is_conv = is_conv

        b_int = tf.cast(self.q_before_sum_t, tf.float64)
        b_frc = tf.cast(0.0, dtype=tf.float64)
        _2 = tf.cast(2.0, dtype=tf.float64)
        _1 = tf.cast(1.0, dtype=tf.float64)
        _05 = tf.cast(0.5, dtype=tf.float64)
        min = -tf.pow(_2, b_int - _1)
        max = -min
        inv_step = tf.cast(tf.pow(tf.cast(2, dtype=tf.int64), tf.cast(tf.abs(b_frc), dtype=tf.int64)), dtype=tf.float64)
        self.maybe_inversed_step_diff = tf.cast(inv_step, dtype=self.dtype)
        self.min_value = tf.cast(min, dtype=self.dtype)
        self.max_value = tf.cast(max, dtype=self.dtype)
        self.b_frc = tf.cast(b_frc, dtype=tf.float32)

    def isQuantisizing(self):
        """test if the layer performs some quantisized operations

        Returns:
            (bool):
                true if an operation is quantisized.
        """
        #return self.quant_mul.isQuantisizing() or self.quant_sum.isQuantisizing() or self.quant_out.isQuantisizing()
        return True

    @tf.function
    def call(self, inputs):
        """forward propagation

        apply the layer operation in the forward pass.

        Parameters:
            inputs (list):
                list of the two input tensors.
        Returns:
            (tensor):
                output of the layer.
        """
        x, y = inputs
        self.q_before_sum(tf.constant([1.0]))

        # if self.is_conv:
        #     y_abs = y# * y_signs
        #     y_abs_ext = tf.repeat(tf.expand_dims(y_abs, 0), repeats=tf.cast(tf.shape(x)[0]/self.splits, dtype=tf.int32), axis=0)
        #     sign = tf.sign(y_abs_ext)
        #     max_log = 2.0 ** (self.msb - self.lsb + 1.0) - 1.0
        #     log_cap_divider = 2.0 ** -self.lsb
        #
        #     @tf.function(jit_compile=True)
        #     def over_x(cur_x):
        #         cur_x = tf.repeat(tf.expand_dims(cur_x, 2), repeats=tf.shape(y_abs)[1], axis=2)  # [1024 27] -> [1024 27 128]
        #         over_x_result = cur_x + y_abs_ext
        #         log_cap = 1.0 - tf.sign(tf.round(tf.abs(over_x_result) / (2.0 * max_log)))
        #         over_x_result = tf.pow(2.0, over_x_result / log_cap_divider)  # translate back to lin domain
        #         over_x_result = over_x_result * sign * log_cap  # translate back to lin domain
        #         return tf.reduce_sum(over_x_result, 1)
        #
        #     splitted = tf.split(x, self.splits)
        #     splitted = tf.reshape(splitted, tf.shape(splitted))  # tf.transpose(splitted, [1, 0, 2])
        #     added_floats = tf.stop_gradient(tf.map_fn(over_x, splitted))
        #     added_floats = tf.squeeze(tf.concat(tf.split(added_floats, self.splits, axis=0), axis=1))
        #     result = added_floats#tf.reduce_sum(added_floats, axis=1)# added_floats
        #
        #     return result

        y_signs = tf.where(y < 0.0, -1.0, 1.0)
        y_abs = y * y_signs

        @tf.function(jit_compile=True)
        def over_x2(cur_x):
            matmuled = tf.einsum('ij,jk->ijk', cur_x, y_abs)  # lwx = lw + lx; // -log(wx)

            #matmuled = self.q_lns(matmuled)
            inputs_zeros = tf.where(matmuled == 0.0, 0.0, 1.0)
            lwx = tf.where(matmuled != 0.0, tf.math.log(tf.abs(matmuled)) / tf.math.log(2.0), 0.0)  # get log2(inputs)
            lwx = lwx * tf.pow(2.0, -self.lsb) # round log2 number
            lwx = tf.floor(lwx + 0.5)          # round log2 number
            lwxFloat = lwx * tf.pow(2.0, self.lsb)  # round log2 number
            wxFloat = tf.pow(2.0 , lwxFloat) # wxFloat = Math.Pow(2, lwxFloat); // rounding @lsb=-53, 2**x
            wxFloat = wxFloat * tf.pow(2.0, -self.expLsb) # wxFloat *= Math.Pow(2, -expLsb); // exact, alignment
            summand = wxFloat * inputs_zeros  # filter NaN values
            matmuled = tf.floor(summand + 0.5) # summand = (long)Math.Floor(wxFloat + 0.5f); // rounding @expLsb

            #matmuled = self.q_before_sum(matmuled)
            tmp = tf.clip_by_value(matmuled, self.min_value, self.max_value)
            tmp = (tmp - self.min_value)
            tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff, lambda: tmp / self.maybe_inversed_step_diff)
            tmp = tf.floor(tmp + 0.5)
            matmuled = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
                          lambda: tmp * self.maybe_inversed_step_diff) + self.min_value

            matmuled *= y_signs
            return tf.reduce_sum(matmuled, axis=1)
            #
            # cur_x = tf.repeat(tf.expand_dims(cur_x, 2), repeats=tf.shape(y_abs)[1], axis=2)  # [1024 27] -> [1024 27 128]
            # over_x_result = cur_x + y_abs_ext
            # log_cap = 1.0 - tf.sign(tf.round(tf.abs(over_x_result) / (2.0 * max_log)))
            # over_x_result = tf.pow(2.0, over_x_result / log_cap_divider)  # translate back to lin domain
            # over_x_result = over_x_result * sign * log_cap  # translate back to lin domain
            # return tf.reduce_sum(over_x_result, 1)

        splitted = tf.split(x, self.splits)
        splitted = tf.reshape(splitted, tf.shape(splitted))#tf.stack(splitted, axis=0)  # tf.transpose(splitted, [1, 0, 2])
        added_floats = tf.stop_gradient(tf.map_fn(over_x2, splitted))
        added_floats = tf.squeeze(tf.concat(tf.split(added_floats, self.splits, axis=0), axis=1), axis=0)
        result = added_floats#tf.reduce_sum(added_floats, axis=1)# added_floats
        return result

        # y_signs = tf.where(y < 0.0, -1.0, 1.0)
        # y_abs = y * y_signs
        #
        # added_floats = tf.einsum('ij,jk->ijk', x, y_abs)  # lwx = lw + lx; // -log(wx)
        #
        # added_floats = self.q_lns(added_floats)
        # added_floats = self.q_before_sum(added_floats)
        #
        # added_floats *= y_signs
        # sum = tf.reduce_sum(added_floats, axis=1)
        # return sum

    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.MatMulOperationLayer"
        return config