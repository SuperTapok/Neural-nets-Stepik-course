import numpy as np


class Perceptron:
    def __init__(self):
        self.__inputs = None
        self.__weights = None
        self.__output = None

    @property
    def weights(self):
        return self.__weights

    @staticmethod
    def __expand_array(array):
        # expanding matrices from size n x m to size n + 1 x m
        return np.eye(array.shape[0] + 1, array.shape[0], k=-1) @ array

    def __update_inputs(self, inputs):
        # reshaping inputs to add bias (adding a first fictitious value = 1)
        self.__inputs = self.__expand_array(inputs)
        self.__inputs[0][0] = 1

    def __update_weights_and_bias(self, weights, bias):
        # reshaping weights to add bias. Bias is the first weight
        self.__weights = self.__expand_array(weights)
        self.__weights[0][0] = bias

    def train(self, train_x, train_y):
        self.__weights = np.zeros((train_x[0].shape[0] + 1, train_x[0].shape[1]))

        perfect = False
        while not perfect:
            perfect = True
            for index, value in enumerate(train_x):
                result = self.get_result(value)
                if result != train_y[index]:
                    perfect = False
                    if result == 1:
                        value = self.__expand_array(value)
                        value[0] = 1
                        self.__weights -= value
                    else:
                        value = self.__expand_array(value)
                        value[0] = 1
                        self.__weights += value

    @staticmethod
    def __act_func(value):
        if value > 0:
            return 1
        else:
            return 0

    def get_result(self, inputs):
        if self.__weights is not None:
            self.__update_inputs(inputs)
            self.__output = self.__act_func(self.__weights.T @ self.__inputs)
        else:
            print("Not all required params are set")
        return self.__output


if __name__ == "__main__":
    my_perceptron = Perceptron()

    train_values = np.array([[[1],
                              [0.3]],
                             [[0.4],
                              [0.5]],
                             [[0.7],
                              [0.8]]])
    train_results = np.array([[1], [1], [0]])

    my_perceptron.train(train_values, train_results)
    print(my_perceptron.weights)
