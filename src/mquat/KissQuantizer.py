import logging
import mquat as mq
import numpy as np
import tensorflow as tf

from .PostTrainingQuantizer import PostTrainingQuantizer, BitDistribution, LayerBitDistribution
from .QuantizerBase import DEFAULT_DATATYPE

class KissQuantizer(PostTrainingQuantizer):

    def __init__(self, debug=False):
        """
        Constructor
        Args:
            debug: Enables logging of debug messages if True, else it's disabled
        """
        if debug is True:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    def quantize(self, net, word_width, validation_data, favor_integer=True):
        """
        This method quantizes the passed neural network to the set word width.

        Args:
            net: Neuronal network which shall be quantized
            word_width: Sets the quantization word width
            validation_data: Validation data of the training dataset which was used for training.
            favor_integer: Specifies whether the integer part is to be favored

        Returns:
            Quantized neuronal network
        """

        int_frac_dist_list = self.__calculate_int_frac_bit_dist_list(net, word_width, validation_data, favor_integer)
        self.__set_quantizer(net, int_frac_dist_list)
        net.buildGraphs()

    def __calculate_int_frac_bit_dist_list(self, net, word_width, validation_data, favor_integer):
        """
        This method calculates the integer fractional bit distribution for weights, biases, activations for all layers
        of the neuronal network

        Args:
            net: Neuronal network which shall be quantized
            word_width: Sets the quantization word width
            validation_data: Validation data of the training dataset which was used for training.
            favor_integer: Specifies whether the integer part is to be favored

        Returns:
            List of Integer Fractional bit distribution for the Neuronal network
        """
        weights, biases, activations = self.get_variables(net, validation_data)
        bit_distribution_list = list()

        for (weights_layer, bias_data_layer, activation_layer) in zip(weights, biases, activations):

            layer_name = weights_layer["layer"]
            logging.debug("Layer: {}".format(layer_name))

            weight_data = weights_layer["variables"].flatten()
            bias_data = bias_data_layer["variables"].flatten()
            activation_data = activation_layer["variables"].flatten()

            # Calculation int frac bit distribution for weights
            int_weight_bits, frac_weight_bits = self.__calculate_int_frac_bit_dist_layer(weight_data, word_width, favor_integer)
            logging.debug("Weight Int {} Frac {}".format(int_weight_bits, frac_weight_bits))

            # Calculation int frac bit distribution for biases
            int_bias_bits, frac_bias_bits = self.__calculate_int_frac_bit_dist_layer(bias_data, word_width, favor_integer)
            logging.debug("Bias Int {} Frac {}".format(int_bias_bits, frac_bias_bits))

            # Calculation int frac bit distribution for activations
            int_act_bits, frac_act_bits = self.__calculate_int_frac_bit_dist_layer(activation_data, word_width, favor_integer)
            logging.debug("Activation Int {} Frac {}".format(int_act_bits, frac_act_bits))

            weight_filter_bit_dist = BitDistribution(int_weight_bits, frac_weight_bits)
            bias_bit_dist = BitDistribution(int_bias_bits, frac_bias_bits)
            activation_bit_dist = BitDistribution(int_act_bits, frac_act_bits)

            layer_bit_distribution = LayerBitDistribution(weight_filter_bit_dist, bias_bit_dist, activation_bit_dist, layer_name)

            bit_distribution_list.append(layer_bit_distribution)

        return bit_distribution_list

    def __calculate_int_frac_bit_dist_layer(self, data_as_array, word_width, favor_integer):
        """
        This method calculates the integer fractional bit distribution for weights, biases, activations for a single
        layer of the neuronal network

        Args:
            data_as_array: Layer data as array
            word_width: Sets the quantization word width
            favor_integer: Specifies whether the integer part is to be favored

        Returns:
            Number of Integer Bits, Number of Fractional Bits, Ratio
        """
        integer_bits = word_width
        fractional_bits = 0

        span, min_val, max_val = self.get_span(data_as_array)

        # If all values are between -1 and 1, 0 Integer-Bits are set
        if min_val > -1.0 and max_val < 1.0:
            integer_bits = 0

        logging.debug("Favor-Integer is {}".format(favor_integer))
        logging.debug("Integer-Bits Before {}".format(integer_bits))

        # Get the absolute max value of the dataset
        max_dataset_val = abs(max_val) if (abs(max_val) > abs(min_val)) else abs(min_val)

        while integer_bits > 1:

            logging.debug("Integer-Bits Loop {}".format(integer_bits))

            # Adjust the number of fractional_bits, if the integer part is preferred
            # This adjustmentcauses the upper limit to fall more slowly
            if favor_integer is True:
                fractional_bits = word_width - integer_bits

            upper_limit = tf.pow(2.0, integer_bits - 1.0) - tf.pow(0.5, fractional_bits)

            # If the max quantized value is greater than the max value of the input-data decrease
            if upper_limit > max_dataset_val:
                integer_bits = integer_bits - 1

            # If the maximum quantized value is smaller than the maximum value of the input data, the number of
            # integer bits is increased by a maximum of one bit
            else:
                integer_bits = integer_bits + 1
                break

        # If the minimum value is less than 0.0, then add a sign bit
        if min_val < 0.0:
            integer_bits = integer_bits + 1

        logging.debug("Integer-Bits after adding Sign-Bit {}".format(integer_bits))

        # If the number of integer bits is greater than the word width, set integer bits to word width
        integer_bits = word_width if integer_bits > word_width else integer_bits

        # The number of fractional bits is 0, if the number of integer bits is greater equal word width
        fractional_bits = word_width - integer_bits if (integer_bits < word_width) else 0

        logging.debug("Integer-Bits: {} Fractional-Bits: {}".format(integer_bits, fractional_bits))
        logging.debug("------------------------------------------------\n")

        return integer_bits, fractional_bits

    def __set_quantizer(self, net, bit_distribution_list):
        """
        Sets the quantizers and integer fractional bit distribution of all layers
        Args:
            net: Neuronal Network
            bit_distribution_list: List of Integer Fractional bit distribution for the Neuronal network

        Returns:
            -
        """
        for bit_dist in bit_distribution_list:
            for index in range(len(net.layers)):
                if net.layers[index].name == bit_dist.name:
                    self.__set_quant_layer_variables(net.layers[index], bit_dist, mq.FixedPointQuantizer)

    def __set_quant_layer_variables(self, layer, bit_dist, quantizer):
        """
        Sets quantizers for specific layer for weights/filters, biases and activations based on Integer/Fractional
        bit distribution

        Args:
            layer: Layer, which should be quantized
            bit_dist: Integer/Fractional bit distribution for weights/filters, biases and activations
            quantizer: Quantizer-Type

        Returns:
            -
        """
        quant_filter_weight = quantizer("_{}fw".format(layer.name), dtype=DEFAULT_DATATYPE)
        quant_filter_weight.setParams(bit_dist.weights_filters.integer_bits, bit_dist.weights_filters.fractional_bits)

        quant_bias = quantizer("_{}b".format(layer.name), dtype=DEFAULT_DATATYPE)
        quant_bias.setParams(bit_dist.bias.integer_bits, bit_dist.bias.fractional_bits)

        quant_out = quantizer("_{}out".format(layer.name), dtype=DEFAULT_DATATYPE)
        quant_out.setParams(bit_dist.activation.integer_bits, bit_dist.activation.fractional_bits)

        if isinstance(layer, mq.Conv2DLayer):
            layer.f.quant_out = quant_filter_weight
            layer.b.quant_out = quant_bias
            layer.quant_out = quant_out

        if isinstance(layer, mq.DenseLayer):
            layer.w.quant_out = quant_filter_weight
            layer.b.quant_out = quant_bias
            layer.quant_out = quant_out
