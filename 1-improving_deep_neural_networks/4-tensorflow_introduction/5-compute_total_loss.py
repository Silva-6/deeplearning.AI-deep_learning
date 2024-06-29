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
    total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true=labels, y_pred=logits, from_logits=True))
    # YOUR CODE ENDS HERE
    return total_loss
