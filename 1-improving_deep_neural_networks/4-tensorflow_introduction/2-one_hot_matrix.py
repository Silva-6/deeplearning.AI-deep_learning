# GRADED FUNCTION: one_hot_matrix
def one_hot_matrix(label, C=6):
    """
    Computes the one hot encoding for a single label

    Arguments:
        label --  (int) Categorical labels
        C --  (int) Number of different classes that label can take

    Returns:
         one_hot -- tf.Tensor A one-dimensional tensor (array) with the one hot encoding.
    """
    # (approx. 1 line)
    # one_hot = None(None(None, None, None), shape=[C, ])
    # YOUR CODE STARTS HERE
    one_hot = tf.reshape(tf.one_hot(label, depth, axis=0), [depth,1])
    # YOUR CODE ENDS HERE
    return one_hot


def one_hot_matrix_test(target):
    label = tf.constant(1)
    C = 4
    result = target(label, C)
    print("Test 1:", result)
    assert result.shape[0] == C, "Use the parameter C"
    assert np.allclose(result, [0., 1., 0., 0.]), "Wrong output. Use tf.one_hot"
    label_2 = [2]
    C = 5
    result = target(label_2, C)
    print("Test 2:", result)
    assert result.shape[0] == C, "Use the parameter C"
    assert np.allclose(result, [0., 0., 1., 0., 0.]), "Wrong output. Use tf.reshape as instructed"

    print("\033[92mAll test passed")


one_hot_matrix_test(one_hot_matrix)
