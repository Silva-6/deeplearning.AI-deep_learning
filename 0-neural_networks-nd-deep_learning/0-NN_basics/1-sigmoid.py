# GRADED FUNCTION: sigmoid

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """

    # (≈ 1 line of code)
    # s =
    s = 1 / (1 + np.exp(-x))

    return s

t_x = np.array([1, 2, 3])
print("sigmoid(t_x) = " + str(sigmoid(t_x)))

sigmoid_test(sigmoid)