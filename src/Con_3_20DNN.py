# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import mquat as mq
import onnx as onnx
from onnx import numpy_helper



class Con_3_20DNN(mq.QNetBaseModel):
    def __init__(self, input_shape, output_shape, target_shape, weight_decay, dtype=tf.float32):
        super().__init__("Con_3_20DNN", input_shape, output_shape, target_shape, dtype=dtype)
        regularizer = tf.keras.regularizers.l2(weight_decay)
        self._dtype = dtype

        self.flatten = mq.FlattenLayer("flatten", dtype=self._dtype)
        self.dense1 = mq.DenseLayer("dense1", 20, weight_regularizer=regularizer,
                                   bias_regularizer=regularizer,
                                   do_batch_norm=False, activation_func=tf.nn.relu, tainable_bias=True, dtype=self._dtype)

        self.dense2 = mq.DenseLayer("dense2", 20, weight_regularizer=regularizer,
                                   bias_regularizer=regularizer,
                                   do_batch_norm=False, activation_func=tf.nn.relu, tainable_bias=True, dtype=self._dtype)

        self.dense3 = mq.DenseLayer("dense3", 20, weight_regularizer=regularizer,
                                   bias_regularizer=regularizer,
                                   do_batch_norm=False, activation_func=tf.nn.relu, tainable_bias=True, dtype=self._dtype)

        self.dense4 = mq.DenseLayer("dense4", output_shape[0], weight_regularizer=regularizer,
                                   bias_regularizer=regularizer,
                                   do_batch_norm=False, activation_func=tf.identity, tainable_bias=True, dtype=self._dtype)
        self.dense_list_help = [self.dense1, self.dense2, self.dense3, self.dense4]
        self.sub = tf.Variable([0.0, 0.0, 0.0, 0.0, 0.0], trainable=False, dtype=self._dtype)

    # def py_cheat(self, value):
    #     self.last_output6.assign(self.last_output5)
    #     self.last_output5.assign(self.last_output4)
    #     self.last_output4.assign(self.last_output3)
    #     self.last_output3.assign(self.last_output2)
    #     self.last_output2.assign(self.last_output1)
    #     self.last_output1.assign(tf.reshape(value, ()))
    #     return 0.0

    def call(self, inputs):
        tmp = inputs
        tmp = self.flatten(tmp)
        tmp = tmp - self.sub
        #tmp = tf.pad(tmp, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        tmp = self.dense1(tmp)
        tmp = self.dense2(tmp)
        tmp = self.dense3(tmp)
        tmp = self.dense4(tmp)
        # tf.py_function(self.py_cheat, inp=[tmp], Tout=[tf.float32])
        #self.last_output.assign(tf.reshape(tmp, ()))
        return tmp

    def load_pretrained_weights(self, h5_file=None):
        MODEL_PATH = "controller_3_20.onnx"
        _model = onnx.load(MODEL_PATH)
        weights = _model.graph.initializer
        #Init block
        self.sub.assign(tf.reshape(numpy_helper.to_array(weights[0]), [-1]))
        self.dense1.w.var.assign(tf.transpose(numpy_helper.to_array(weights[1]), perm=[1,0]))
        self.dense2.w.var.assign(tf.transpose(numpy_helper.to_array(weights[3]), perm=[1,0]))
        self.dense3.w.var.assign(tf.transpose(numpy_helper.to_array(weights[5]), perm=[1,0]))
        self.dense4.w.var.assign(tf.transpose(numpy_helper.to_array(weights[7]), perm=[1,0]))
        self.dense1.b.var.assign(numpy_helper.to_array(weights[2]))
        self.dense2.b.var.assign(numpy_helper.to_array(weights[4]))
        self.dense3.b.var.assign(numpy_helper.to_array(weights[6]))
        self.dense4.b.var.assign(numpy_helper.to_array(weights[8]))
        #self.sub.assign(tf.reshape(numpy_helper.to_array(weights[0]), [-1]))
        # self.dense1.w.var.assign(tf.cast(tf.transpose(numpy_helper.to_array(weights[1]), perm=[1,0]), dtype=self.dtype))
        # self.dense2.w.var.assign(tf.cast(tf.transpose(numpy_helper.to_array(weights[3]), perm=[1,0]), dtype=self.dtype))
        # self.dense3.w.var.assign(tf.cast(tf.transpose(numpy_helper.to_array(weights[5]), perm=[1,0]), dtype=self.dtype))
        # self.dense4.w.var.assign(tf.cast(tf.transpose(numpy_helper.to_array(weights[7]), perm=[1,0]), dtype=self.dtype))
        # self.dense1.b.var.assign(tf.cast(numpy_helper.to_array(weights[2]), dtype=self.dtype))
        # self.dense2.b.var.assign(tf.cast(numpy_helper.to_array(weights[4]), dtype=self.dtype))
        # self.dense3.b.var.assign(tf.cast(numpy_helper.to_array(weights[6]), dtype=self.dtype))
        # self.dense4.b.var.assign(tf.cast(numpy_helper.to_array(weights[8]), dtype=self.dtype))







