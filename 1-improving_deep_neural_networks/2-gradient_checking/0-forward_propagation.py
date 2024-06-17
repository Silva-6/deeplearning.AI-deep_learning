# GRADED FUNCTION: forward_propagation

def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)

    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well

    Returns:
    J -- the value of function J, computed using the formula J(theta) = theta * x
    """

    # (approx. 1 line)
    # J =
    # YOUR CODE STARTS HERE
    J = np.dot(theta, x)
    # YOUR CODE ENDS HERE

    return J
