def pool_backward(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    # Retrieve information from cache (≈1 line)
    # (A_prev, hparameters) = None

    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    # stride = None
    # f = None

    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    # m, n_H_prev, n_W_prev, n_C_prev = None
    # m, n_H, n_W, n_C = None

    # Initialize dA_prev with zeros (≈1 line)
    # dA_prev = None

    # for i in range(None): # loop over the training examples

    # select training example from A_prev (≈1 line)
    # a_prev = None

    # for h in range(n_H):                   # loop on the vertical axis
    # for w in range(n_W):               # loop on the horizontal axis
    # for c in range(n_C):           # loop over the channels (depth)

    # Find the corners of the current "slice" (≈4 lines)
    # vert_start = None
    # vert_end = None
    # horiz_start = None
    # horiz_end = None

    # Compute the backward propagation in both modes.
    # if mode == "max":

    # Use the corners and "c" to define the current slice from a_prev (≈1 line)
    # a_prev_slice = None

    # Create the mask from a_prev_slice (≈1 line)
    # mask = None

    # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
    # dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += None

    # elif mode == "average":

    # Get the value da from dA (≈1 line)
    # da = None

    # Define the shape of the filter as fxf (≈1 line)
    # shape = None

    # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
    # dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += None
    # YOUR CODE STARTS HERE
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache

    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    f = hparameters['f']

    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):  # loop over the training examples

        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i, :, :, :]

        for h in range(n_H):  # loop on the vertical axis
            for w in range(n_W):  # loop on the horizontal axis
                for c in range(n_C):  # loop over the channels (depth)

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    # Compute the backward propagation in both modes.
                    if mode == "max":

                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += dA[i, h, w, c] * mask

                    elif mode == "average":

                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    # YOUR CODE ENDS HERE

    # Making sure your output shape is correct
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev


np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
print(A.shape)
print(cache[0].shape)
dA = np.random.randn(5, 4, 2, 2)

dA_prev1 = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev1[1,1] = ', dA_prev1[1, 1])
print()
dA_prev2 = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev2[1,1] = ', dA_prev2[1, 1])

assert type(dA_prev1) == np.ndarray, "Wrong type"
assert dA_prev1.shape == (5, 5, 3, 2), f"Wrong shape {dA_prev1.shape} != (5, 5, 3, 2)"
assert np.allclose(dA_prev1[1, 1], [[0, 0],
                                    [ 5.05844394, -1.68282702],
                                    [ 0, 0]]), "Wrong values for mode max"
assert np.allclose(dA_prev2[1, 1], [[0.08485462,  0.2787552],
                                    [1.26461098, -0.25749373],
                                    [1.17975636, -0.53624893]]), "Wrong values for mode average"
print("\033[92m All tests passed.")
