import numpy as np


class Perceptron:
    def __init__(self):
        self.__inputs = None
        self.__weights = None
        self.__output = None

    def __update_inputs(self, inputs):
        # reshaping inputs to add bias (adding a first fictitious value = 1)
        self.__inputs = np.eye(inputs.shape[0] + 1, inputs.shape[0], k=-1) @ inputs
        self.__inputs[0][0] = 1

    def update_weights_and_bias(self, weights, bias):
        # reshaping weights to add bias. Bias is the first weight
        self.__weights = np.eye(weights.shape[0] + 1, weights.shape[0], k=-1) @ weights
        self.__weights[0][0] = bias

    def get_result(self, inputs):
        if self.__weights is not None:
            self.__update_inputs(inputs)
            self.__output = self.__act_func(self.__weights.T @ self.__inputs)
        else:
            print("Not all required params are set")
        return self.__output

    @staticmethod
    def __act_func(value):
        if value > 0:
            return 1
        else:
            return 0


if __name__ == "__main__":
    my_perceptron = Perceptron()
    # representing simple boolean function AND
    my_perceptron.update_weights_and_bias(np.array([[1],
                                                    [1]]), -1.5)

    print(my_perceptron.get_result(np.array([[1],
                                             [1]])))
