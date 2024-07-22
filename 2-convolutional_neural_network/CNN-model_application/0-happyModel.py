# GRADED FUNCTION: happyModel

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """
    model = tf.keras.Sequential([
        ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3

        ## Conv2D with 32 7x7 filters and stride of 1

        ## BatchNormalization for axis 3

        ## ReLU

        ## Max Pooling 2D with default parameters

        ## Flatten layer

        ## Dense layer with 1 unit for output & 'sigmoid' activation

        # YOUR CODE STARTS HERE
        tfl.Input((64, 64, 3)),
        ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
        tfl.ZeroPadding2D(padding=(3, 3)),
        ## Conv2D with 32 7x7 filters and stride of 1
        tfl.Conv2D(32, (7, 7), strides=(1, 1)),
        ## BatchNormalization for axis 3
        tfl.BatchNormalization(axis=3),
        ## ReLU
        tfl.ReLU(),
        ## Max Pooling 2D with default parameters
        tfl.MaxPooling2D((2, 2)),
        ## Flatten layer
        tfl.Flatten(),
        ## Dense layer with 1 unit for output & 'sigmoid' activation
        tfl.Dense(1, activation='sigmoid'),

        # YOUR CODE ENDS HERE
    ])

    return model


happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)

output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
          ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
          ['BatchNormalization', (None, 64, 64, 32), 128],
          ['ReLU', (None, 64, 64, 32), 0],
          ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
          ['Flatten', (None, 32768), 0],
          ['Dense', (None, 1), 32769, 'sigmoid']]

comparator(summary(happy_model), output)
