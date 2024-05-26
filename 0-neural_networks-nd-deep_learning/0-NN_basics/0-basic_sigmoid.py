import math
from public_tests import *


# GRADED FUNCTION: basic_sigmoid

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    # (≈ 1 line of code)
    # s =
    s = (1 + math.exp(-x)) ** (-1)

    return s