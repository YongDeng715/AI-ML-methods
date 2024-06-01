import numpy as np
import matplotlib.pyplot as plt

import os
import argparse

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

class GaussianNBC(object):
    def __init__(self, eps=1e-6):
        self.classes = None # (n_classes,)
        self.n_classes = None # num of classes
        self.parameters = {
            "mean": None,  # shape: (K, M)
            "sigma": None,  # shape: (K, M)
            "prior": None,  # shape: (K,)
        }
        self.hyperparameters = {"eps": eps}

    def fit(self, X, y):
        """
        Fit the model parameters via maximum likelihood.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`
        y: :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The class label for each of the `N` examples in `X`  
        """
        self.classes = np.unique(y) # get unique classes
        self.n_classes = len(self.classes) # num of classes
        K = self.n_classes
        N, M = X.shape

        P = self.parameters
        H = self.hyperparameters
        P["mean"] = np.zeros((K, M))
        P["sigma"] = np.zeros((K, M))
        P["prior"] = np.zeros((K,))

        for i, cls_ in enumerate(self.classes):
            cls_indices = np.where(y == cls_)[0] # get indices of class cls_
            X_cls = X[cls_indices]

            P["mean"][i, :] = np.mean(X_cls, axis=0)
            P["sigma"][i, :] = np.var(X_cls, axis=0) + H["eps"]
            P["prior"][i] = len(cls_indices) / N
        return self
    
    def predict(self, X):
        """
        predict the class label for each example in **X** via trained Gaussion NBC.
        """
        return self.classes[np.argmax(self._log_posterior(X), axis=1)]

    def _log_posterior(self, X):
        
        K = self.n_classes
        P = self.parameters
        log_posterior = np.zeros((X.shape[0], K))
        for i in range(K):
            mu = P["mean"][i]
            sigsq = P["sigma"][i]
            prior = P["prior"][i]

            # log likelihood = log X | N(mu, sigsq)
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigsq))
            log_likelihood -= 0.5 * np.sum(((X - mu) ** 2) / sigsq, axis=1)
            log_posterior[:, i] = log_likelihood + np.log(prior)
        return log_posterior
    

def load_file_data(file_path):
    X = []
    y = []
    text = np.loadtxt(file_path, skiprows=1)
    X.append(text[:, 1:])
    y.append(text[:, 0])
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

def visualization(classifier, X, y):
    plt.figure()
    # X.shape: (N, D), W.shape: (D, C), y.shape: (N, 1)
    n_classes = int(np.max(np.unique(y))) + 1
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o',s=5, cmap=plt.cm.get_cmap('Set1', n_classes))

    # Generate a grid of points for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    flat_X = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)
    # print(inp.shape)  # (num of pixels, 2)
    flat_y = classifier.predict(flat_X)
    flat_y = flat_y.reshape(xx.shape)
    # Plot decision boundary
    plt.contourf(xx, yy, flat_y, alpha=0.3, cmap=plt.cm.get_cmap('gray', n_classes))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../gmm/GMM6.txt")
    parser.add_argument("--num_cross", type=int, default=5)
    parser.add_argument("--if_display", type=bool, default=True)

    opt = parser.parse_args()
    
    # load data file
    X, y = load_file_data(opt.data_path)

    print(f"shape of data: {X.shape}, shape of labels: {y.shape}")

    n_classes = int(np.max(np.unique(y))) + 1
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o',s=5, cmap=plt.cm.get_cmap('Set1', n_classes))
    plt.colorbar()
    plt.show()

    classifier = GaussianNBC()
    # 初始化交叉验证分割器
    kf = KFold(n_splits=opt.num_cross, shuffle=True, random_state=42)

    # 用于存储每个交叉验证模型的评分
    cv_scores = []

    # 执行 N 倍交叉验证
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        score = np.mean(y_pred == y_test)
        cv_scores.append(score)

    # 输出每个交叉验证模型的评分
    for i, score in enumerate(cv_scores):
        print("Fold {}: {}".format(i+1, score))

    # 输出交叉验证评分的平均值
    print("Average CV Score:", np.mean(cv_scores))

    if opt.if_display:
        visualization(classifier, X, y)
    

    