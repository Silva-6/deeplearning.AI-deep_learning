# Gradient Checking

In this assignment you'll be implementing gradient checking.

By the end of this notebook, you'll be able to:

Implement gradient checking to verify the accuracy of your backprop implementation.

## Packages
```
import numpy as np
from testCases import *
from public_tests import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

%load_ext autoreload
%autoreload 2
```
## 2 - Problem Statement

You are part of a team working to make mobile payments available globally, and are asked to build a deep learning model
to detect fraud--whenever someone makes a payment, you want to see if the payment might be fraudulent, such as if the 
user's account has been taken over by a hacker.

You already know that backpropagation is quite challenging to implement, and sometimes has bugs. Because this is a 
mission-critical application, your company's CEO wants to be really certain that your implementation of backpropagation 
is correct. Your CEO says, "Give me proof that your backpropagation is actually working!" To give this reassurance, you 
are going to use "gradient checking."

Let's do it!

