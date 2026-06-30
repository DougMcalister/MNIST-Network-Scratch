import numpy as np

def relu (x: float):
    return np.maximum(0.0, x)

def relu_der (x):
    return np.where(x > 0.0, 1.0, 0.0)

def softmax (z):
    '''Return the softmax output of a vector.'''
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def cce (y_true, y_pred, batch = True):
    """
    Calculate categorical cross entropy loss.

    y_true: one-hot encoded true labels
    y_pred: predicted probabilities from softmax
    """
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    if batch:
        return np.mean(-np.sum(y_true * np.log(y_pred), axis=0))
    else:
        return -np.sum(y_true * np.log(y_pred))

if __name__ == "__main__":
   test_array = np.array([0.352, 1.201, 0.884, 0.948, 0.426, 0.021])
   out = softmax(test_array)
   print(out, "\n", round(out.sum()))
