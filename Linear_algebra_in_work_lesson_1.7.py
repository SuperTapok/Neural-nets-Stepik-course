import numpy as np


# Main topik of this theme is to show us how linear algebra can make our lives way-way easier.

# simple linear regression example. We must predict length of braking distance, based on initial speed.
def predict_the_braking_distance():
    # array of braking distances
    d = np.array([[10],
                  [7],
                  [12]])
    # array of initial speeds
    v = np.array([[60],
                  [50],
                  [75]])
    y = d

    # extending the matrix of predictors with multiplication on identity matrix
    x = v @ np.eye(v.shape[1], v.shape[1] + 1, 1)
    x[..., 0] = 1

    b_hat = np.linalg.inv((x.T @ x)) @ x.T @ y
    print(b_hat)


# sample input data: https://stepic.org/media/attachments/lesson/16462/boston_houses.csv
def ex_7():
    # In this exercise we should read the input matrix from the file and then calculate
    # the vector of coefficients to predict the first column.
    file_url = input()
    data = np.loadtxt(file_url, skiprows=1, delimiter=',')

    y = data[..., 0]
    x = data[..., 1:]

    # expanding the origin matrix
    x = x @ np.eye(x.shape[1], x.shape[1] + 1, k=1)
    x[..., 0] = 1

    b_hat = np.linalg.inv(x.T @ x) @ x.T @ y

    # forming the answer
    print(' '.join(map(lambda i: str(round(i, 4)), b_hat)))


# here I tried to repeat the standard prediction function with inputs and result
def predict_the_potato_cost(x, y, to_predict):
    # In this "task" we should predict the cost of bag of potato, based on some previous data.
    # transform inputs
    x = x @ np.eye(x.shape[1], x.shape[1] + 1, 1)
    # we also must transform the predicting values to use they in multiplication on vector of coefficients!!!
    to_predict = to_predict @ np.eye(to_predict.shape[1], to_predict.shape[1] + 1, k=1)

    coefficients = np.linalg.pinv(x.T @ x) @ x.T @ y

    return to_predict @ coefficients


if __name__ == "__main__":
    # predict_the_braking_distance()
    # ex_7()
    potato_bags_weight = np.array([[1],
                                   [2],
                                   [3]])

    potato_bags_cost = np.array([[50],
                                 [100],
                                 [150]])

    weight_to_predict_cost = np.array([[4]])
    print(predict_the_potato_cost(potato_bags_weight, potato_bags_cost, weight_to_predict_cost))
