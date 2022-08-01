import numpy as np

# Array assignment

# from sequence
a = np.array([[1, 2, 3], [4, 5, 6]])
# print(a)

# basic class implementation
b = np.ndarray((1, 3))
# print(b)

# ---
# Array types

# basic, from sequence
basic = np.array([[1, 2, 3], [4, 5, 6]])
# print(basic)

# identity matrix
eye = np.eye(3, 3, k=0)
# print(eye)

# singular matrix of zeros
zeros = np.zeros((3, 3))
# print(zeros)

# singular matrix of ones
ones = np.ones((3, 3))
# print(ones)

# singular matrix filled with value
with_value = np.full((3, 3), 10)
# print(with_value)

# ---
# Indexing and slices
# NumPy uses basic indexing construction "start:stop:step". Stop is excluded
# print(basic[0, 0:2 + 1:1])

# We can use "..." or ":" to choose all elements from that dimension
# print(basic[:, 0])
# print(basic[..., 0])

# also, NumPy is supporting negative indexes
# print(basic[-1])


def ex_2():
    # Create and print array:
    # 2 1 0 0
    # 0 2 1 0
    # 0 0 2 1

    # stupid solution
    array = np.array([[2, 1, 0, 0],
                      [0, 2, 1, 0],
                      [0, 0, 2, 1]])
    print(array)

    # more complex, uses sum of matrix and constant multiplications
    first_array = 2 * np.eye(3, 4, 0, int)
    second_array = np.eye(3, 4, 1, int)
    print(first_array + second_array)


# ex_2()

# ---
# Operations

# !!! all most common arithmetic operations (+ - * / // % ** < <= == >= > !=) are implemented for np.arrays
# They transform the number to solid matrix and count the result elementwise.
c = basic ** 2
# print(c)

# Shape
# transform array to one-dimensional
flatten_basic = basic.flatten()
# print(flatten_basic)

# changes the order of axes
transposed_basic = basic.transpose((1, 0))
# print(transposed_basic)

# reshaping array. It straightens array and then fill value to new shape
reshaped_basic = basic.reshape((1, 6))
# print(reshaped_basic)

# Basic statistic functions
max = basic.max(axis=None)
# print(max)

min = basic.min(axis=None)
# print(min)

mean = basic.mean(axis=None)
# print(mean)

standard_deviation = basic.std(axis=None)
# print(standard_deviation)

sum = basic.sum(axis=None)
# print(sum)

prod = basic.prod(axis=None)
# print(prod)

# particular sum (like a1, a1 + a2 and so on)
cumsum = basic.cumsum(axis=None)
# print(cumsum)

# particular multiplication
cumprod = basic.cumprod(axis=None)
# print(cumprod)

multiplied_matrix = basic.dot([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
# print(multiplied_matrix)

inverted_matrix = np.linalg.inv(eye)
# print(inverted_matrix)


def ex_4():
    # Sample data:
    # 2 3
    # 8 7 7 14 4 6
    # 4 3
    # 5 5 1 5 2 6 3 3 9 1 4 6
    # ---
    # 2 3
    # 5 9 9 10 8 9
    # 3 4
    # 6 11 3 5 4 5 3 2 5 8 2 2

    # Read shapes and values of matrices and print value of XY^T or print "matrix shapes do not match"
    x_shape = tuple(map(int, input().split()))
    X = np.fromiter(map(int, input().split()), int).reshape(x_shape)
    y_shape = tuple(map(int, input().split()))
    Y = np.fromiter(map(int, input().split()), int).reshape(y_shape)

    if x_shape[1] != Y.shape[1]:
        print("matrix shapes do not match")
    else:
        # we can use infix operator "@" to present matrix multiplication instead of .dot
        print(X @ Y.T)
        # T is field. Not a method


# ex_4()

# ---
# Reading from file with NumPy
# f = np.loadtxt("filename", usecols=(0, 1, 2), skiprows=1, delimiter=',', dtype={'names': ('date', 'open', 'close'),
# 'formats': ('datetime64[D]', 'f4', 'f4')})

def ex_5():
    # Sample data: https://stepic.org/media/attachments/lesson/16462/boston_houses.csv
    # You must read from input string with url address of dataset and then find the mean value of every column
    file_url = input()

    data = np.loadtxt(file_url, delimiter=',', skiprows=1, dtype=float)
    print(data.mean(axis=0))


ex_5()
