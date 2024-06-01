import numpy as np
from classifiers.loss_function import *

class LinearClassifier(object):
    def __init__(self):
        self.W = None
    
    def loss(self, X_batch, y_batch, reg):
       """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """     
       pass 

    def train(self, X, y, learning_rate=1e-3, reg=0.0, decay=1.0,\
              num_iters=500, batch_size=32, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        - W: shape (D, C)

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; 
            there are N training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; 
            y[i] = c means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.       
        """
        num_train, dim = X.shape
        num_class = int(np.max(y)) + 1 # assume that y takes values 0,...,K-1
        if self.W is None:
            # lazily initialize W
            self.W = np.random.randn(dim, num_class) * 0.001
        
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # Sample batch_size elements from the training data and their labels
            pass
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]
            ####################################################################
            
            # evaluate the loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            # update the weights using the learing rate and the gradient
            learning_rate *= decay
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('itearnation %d / %d : loss %f' % (it, num_iters, loss))
                
        return loss_history
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[1])
        """
        Predict labels for data points via the trained weights of this classifier

        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels, a 1-dimensional array of length N, 
        and each element is an integer giving the predicted class.
        """
        pass
        y_pred = np.argmax(np.dot(X, self.W), axis = 1)
        return y_pred
    

class Softmax(LinearClassifier):

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
    

class LinearSVM(LinearClassifier):
    pass

