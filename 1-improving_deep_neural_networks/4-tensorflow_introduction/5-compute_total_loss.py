# GRADED FUNCTION: compute_total_loss

def compute_total_loss(logits, labels):
    """
    Computes the total loss

    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
    labels -- "true" labels vector, same shape as Z3

    Returns:
    total_loss - Tensor of the total loss value
    """

    # (1 line of code)
    # remember to set `from_logits=True`
    # total_loss = ...
    # YOUR CODE STARTS HERE
    tf.reduce_sum(tf.keras.metrics.categorical_crossentropy(tf.transpose(labels),tf.transpose(logits),from_logits=True))
    # YOUR CODE ENDS HERE
    return total_loss


def compute_total_loss_test(target, Y):
    pred = tf.constant([[2.4048107, 5.0334096],
                        [-0.7921977, -4.1523376],
                        [0.9447198, -0.46802214],
                        [1.158121, 3.9810789],
                        [4.768706, 2.3220146],
                        [6.1481323, 3.909829]])
    minibatches = Y.batch(2)
    for minibatch in minibatches:
        result = target(pred, tf.transpose(minibatch))
        break

    print("Test 1: ", result)
    assert (type(result) == EagerTensor), "Use the TensorFlow API"
    assert (np.abs(result - (
                0.50722074 + 1.1133534) / 2.0) < 1e-7), "Test 1 does not match. Did you get the reduce sum of your loss functions?"

    ### Test 2
    labels = tf.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    logits = tf.constant([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])

    result = compute_total_loss(logits, labels)
    print("Test 2: ", result)
    assert np.allclose(result, 3.295837), "Test 2 does not match."

    print("\033[92mAll test passed")


compute_total_loss_test(compute_total_loss, new_y_train)
