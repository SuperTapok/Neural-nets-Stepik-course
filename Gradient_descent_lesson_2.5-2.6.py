import numpy as np
# It is simple realisation of gradient descent learning method. It is presented on single neuron with linear activation
# function. Actually, at first it planned to have sigmoid function, but I have counted gradient formulas for linear
# neuron, so I decided to change activation function.
# Here I am going to predict the probability of event when I am going for a walk. This event has 3 params:
# - Do I want it?
# - Are friends calling me?
# - Is a good weather today?


class Neuron:
    def __init__(self):
        self.__w = None

    @staticmethod
    def rescale(vector):
        vector = np.eye(vector.shape[0] + 1, vector.shape[0], -1) @ vector
        vector[0][0] = 1
        return vector

    @staticmethod
    def __act_func(x):
        return x

    def get_result(self, inputs):
        return self.__act_func(self.__w.T @ self.rescale(inputs))

    def train(self, inputs_x, inputs_y):
        self.__w = self.rescale(inputs_x[0])
        gradient = np.ones(self.__w.shape[0])
        learning_rate = 0.1
        threshold = 0.001
        # if sum of all elements in gradient is greater than 0.2, so each element of gradient is greater than threshold
        while np.sum(np.abs(gradient)) > self.__w.shape[0] * threshold:
            errors = np.array([self.get_result(x)[0] for x in inputs_x]) - inputs_y
            partial_derivatives = []
            for i in range(self.__w.shape[0]):
                # general formula is ji = sum(errors.T*derivation(f_act(weights.T * inputs)).T*inputs)
                partial_derivatives.append(errors.T @ np.array(list(map(lambda x: self.rescale(x).T, inputs_x)))[..., i])
            gradient = np.array(partial_derivatives)
            self.__w -= learning_rate / self.__w.shape[0] * gradient[..., 0]


if __name__ == "__main__":
    inputs_x = np.array([[[0], [0], [0]], [[0], [0], [1]], [[0], [1], [0]], [[0], [1], [1]], [[1], [0], [0]],
                         [[1], [0], [1]], [[1], [1], [0]]])
    inputs_y = np.array([[0], [0.5], [0.5], [0.75], [0.5], [0.75], [0.75]])

    neuron = Neuron()

    neuron.train(inputs_x, inputs_y)

    test_x = np.array([[1],
                       [1],
                       [1]])
    print(neuron.get_result(test_x))
