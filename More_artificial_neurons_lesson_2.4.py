import numpy as np
import math

# --- Some extras for numpy ---
# we can implement vector in numpy using
a = np.array([1, 2, 3])
# and his shape is (3, ). So, if we want to multiply this vector on matrix, we should use
c = np.dot(a, np.array([[1],
                        [2],
                        [3]]))
print(c)
# so, shapes (m,) and (m, 1) are not equal
# ---


# types of neurons. Input x for every activation function is result of summing function w^T*x
class BaseNeuron:
    @staticmethod
    def act_func(x):
        return x


class LinearNeuron(BaseNeuron):
    @staticmethod
    def act_funct(x):
        return x


class Perceptron(LinearNeuron):
    @staticmethod
    def act_funct(x):
        return 1 if x > 0 else 0


class SigmoidalOrLogisticNeuron(BaseNeuron):
    @staticmethod
    def act_func(x):
        return 1 / (1 + math.exp(- x))


class HyperbolicTangent(BaseNeuron):
    @staticmethod
    def act_func(x):
        return (math.exp(x) - math.exp(- x)) / (math.exp(x) + math.exp(- x))


class ReLU(BaseNeuron):
    @staticmethod
    def act_func(x):
        return max(0, x)


class Softplus(BaseNeuron):
    @staticmethod
    def act_func(x):
        return math.log(1 + math.exp(x))
