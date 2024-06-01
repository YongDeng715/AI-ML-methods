import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def hard_svm_loss(W, X, y, reg=0.0):
    # X: N x D, W: D x C, y: N x 1
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    pass
    N = X.shape[0]
    scores = np.dot(X, W) # N x C
    margin = scores - scores[range(0,N), y].reshape(-1, 1) + 1 # N x C
    margin[range(N), y] = 0
    margin = (margin > 0) * margin # max(0, s_j - s_yi + delta)
    loss += margin.sum() / N
    loss += 0.5 * reg * np.sum(W * W)
    #############################################################################

    pass
    counts = (margin > 0).astype(int)
    counts[range(N), y] = - np.sum(counts, axis = 1)
    dW += np.dot(X.T, counts) / N + reg * W
    #############################################################################
    return loss, dW

def soft_svm_loss(W, X, y, C_para, reg=0.0):
    pass



class LinearSVM(object):
    def __init__(self, C_param):
        self.W = None
        self.b = None
        self.C_param = C_param
    
    def train(self, X, y, learning_rate=1e-3, reg=0.0, num_iters=1000):
        N, D = X.shape
        y_tmp = np.where(y <= 0, -1, 1).reshape(-1, 1)
        self.W = np.zeros((D , 1))
        self.b = 0.0

        # gradient descent
        for _ in range(num_iters):
            for idx, sample in enumerate(X):
                condition = y_tmp[idx] * (np.dot(sample.reshape(1, -1), self.W) - 1) >= 1

                if condition:
                    self.W -= learning_rate * (2 * reg * self.W)
                else:
                    self.W -= learning_rate * (2 * reg * self.W - \
                                               np.dot(sample.reshape(-1, 1), y_tmp[idx].reshape(1, -1)))
                    self.b -= learning_rate * y_tmp[idx]
    
    def predict(self, X_test):
        linear_output = np.dot(X_test, self.W) - self.b
        return np.sign(linear_output)
    
    def loss(self, X_batch, y_batch, reg, soft=True):
        if soft:
            return soft_svm_loss(self.W, X_batch, y_batch, self.C_param, reg)
        else:
            return hard_svm_loss(self.W, X_batch, y_batch, reg)


def svm_visualization(X_test, y_test, y_pred):
     # Create scatter plots of the test data with colored points representing the true and predicted labels
    fig, ax = plt.subplots()

    # Plot true labels
    scatter1 = ax.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], \
                           c='b', label='True Label -1', cmap='viridis')
    scatter2 = ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], \
                           c='g', label='True Label 1', cmap='viridis')

    # Plot predicted labels
    scatter4 = ax.scatter(X_test[y_pred[:, 0] == -1, 0], X_test[y_pred[:, 0] == -1, 1], \
                           c='r', marker='x', s=100, label='Predicted Label -1')
    scatter5 = ax.scatter(X_test[y_pred[:, 0] == 1, 0], X_test[y_pred[:, 0] == 1, 1], \
                          c='orange', marker='x', s=100, label='Predicted Label 1')

    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100),
                        np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 100))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('SVM Results')
    handles = [scatter1, scatter2, scatter4, scatter5]
    labels = [h.get_label() for h in handles]
    ax.legend(handles=handles, labels=labels)

    # Set the limits of the axes to include all points
    ax.set_xlim(X_test[:, 0].min() - 0.01, X_test[:, 0].max() + 0.01)
    ax.set_ylim(X_test[:, 1].min() - 0.01, X_test[:, 1].max() + 0.01)

    plt.show()  

if __name__ == '__main__':
    X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=42)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, cmap=plt.cm.get_cmap('flag'))
    ax[1].scatter(X_test[:, 0], X_test[:, 1], marker='x', c=y_test, cmap=plt.cm.get_cmap('flag'))
    ax[0].set_title('Training Data')
    ax[1].set_title('Testing Data')
    plt.show()

    svm = LinearSVM(C_param = 0.1)
    svm.train(X_train, y_train, learning_rate=0.001, num_iters=1000)
    svm_visualization(X_train, y_train, svm.predict(X_train))
    y_pred = svm.predict(X_test)
    svm_visualization(X_test, y_test, y_pred)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

