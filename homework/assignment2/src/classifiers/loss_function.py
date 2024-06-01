import numpy as np
# from random import shuffle

def softmax_loss_native(W, X, y, reg=0.0):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W   
    """
    # Initialize the loss and gradient
    loss = 0.0
    dW = np.zeros_like(W)

    N, C = X.shape[0], W.shape[1]
    for i in range(N):
        z_i = np.dot(X[i], W)
        z_i -= np.max(z_i) # z.shape = C
        y_i = int(y[i])

        loss = loss + np.log(np.sum(np.exp(z_i))) - z_i[y_i]
        dW[:, y_i] -= X[i]
        _sum_j = np.sum(np.exp(z_i))
        for j in range(C):
            dW[:, j] += np.exp(z_i[j]) / _sum_j * X[i]
        loss = loss / N + 0.5 * reg * np.sum(W * W)
        dW = dW / N + reg * W

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg=0.0):
    """
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    """
    loss = 0.0
    dW = np.zeros_like(W)

    pass
    N = X.shape[0]
    z = np.dot(X, W) # z.shape = (N, C)
    z -= z.max(axis=1).reshape(N, 1)
    y = y.astype(int)

    _sum_j = np.exp(z).sum(axis=1)
    loss = np.log(_sum_j).sum() - z[range(N), y].sum()
    counts = np.exp(z) / _sum_j.reshape(N,1)
    counts[range(N), y] -= 1
    dW = np.dot(X.T, counts)

    loss = loss / N + 0.5 * reg * np.sum(W * W)
    dW = dW / N + reg * W

    return loss, dW

def hard_svm_loss(W, X, y, reg):
    pass

def soft_svm_loss(W, X, y, reg, C_para):
    pass

