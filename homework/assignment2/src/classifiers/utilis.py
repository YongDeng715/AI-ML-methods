import numpy as np
import matplotlib.pyplot as plt
# from classifiers.linear_classifier import Softmax
# from classifiers.softmax import softmax_loss_native

def visualize_softmax(W, X, y, fig):
    # X.shape: (N, D), W.shape: (D, C), y.shape: (N, 1)
    fig.scatter(X[:, 0], X[:, 1], c=y, cmap='flag')

    # Generate a grid of points for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    flat_X = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1),
                          np.ones((len(xx.ravel()), 1))], axis=1)
    # print(inp.shape)  # (num of pixels, 2)
    
    # Predict output for flat_X, using argmax to get label of maximum probability
    flat_y = np.argmax(_softmax(W, X))
    # flat_y = np.argmax(np.dot(flat_X, W), axis=1) 
    flat_y = flat_y.reshape(xx.shape)
    # Plot decision boundary
    fig.contourf(xx, yy, flat_y, alpha=0.2, cmap=plt.cm.get_cmap('prism'))


def _softmax(theta, X):
    assert theta.shape[0] == X.shape[1]
    # print(theta.shape)
    # print(theta)
    # print(X.shape)
    # print(X)
    tmp = np.exp(np.dot(X, theta))
    probs = tmp / np.sum(tmp, axis=1, keepdims=True)
    return probs