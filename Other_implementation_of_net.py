import math
import numpy as np

# The idea is that net can be realised as two arrays: array of input data and array of weights. In this case we don't
# need to make classes for all types of objects (for net, layer and single neuron). But there may be difficulties if net
# isn't consist of only dense layers.


class DenseNeuralNet:
    def __init__(self, number_of_layers=2, number_of_neurons=3):
        # self.__w = np.zeros((number_of_neurons, number_of_neurons, number_of_layers))
        self.__w = np.array([[[0.9, 0.2, 0.1],
                             [0.3, 0.8, 0.5],
                             [0.4, 0.2, 0.6]],
                             [[0.3, 0.6, 0.8],
                              [0.7, 0.5, 0.1],
                              [0.5, 0.2, 0.9]]])

    def predict(self, input_data):
        result = input_data
        for i in self.__w:
            result = np.array(list(map(lambda x: 1 / (1 + math.exp(-x)), i.T @ result)))
        return result


if __name__ == "__main__":
    input_data = np.array([0.9, 0.1, 0.8])
    net = DenseNeuralNet()
    print(net.predict(input_data))
