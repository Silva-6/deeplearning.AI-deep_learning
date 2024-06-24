# GRADED FUNCTION: linear_function

def linear_function():
    """
    Implements a linear function:
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- Y = WX + b
    """

    np.random.seed(1)

    """
    Note, to ensure that the "random" numbers generated match the expected results,
    please create the variables in the order given in the starting code below.
    (Do not re-arrange the order).
    """
    # (approx. 4 lines)
    # X = ...
    # W = ...
    # b = ...
    # Y = ...
    # YOUR CODE STARTS HERE
    X = tf.constant(np.random.randn(3, 1), name="X")
    W = tf.Variable(np.random.randn(4, 3), name="X")
    b = tf.Variable(np.random.randn(4, 1), name="X")
    Y = tf.add(tf.matmul(W, X), b)  # Y = W@X + b
    # YOUR CODE ENDS HERE
    return Y
