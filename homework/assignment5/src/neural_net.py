# neural_net.py
import numpy as np  
import matplotlib.pyplot as plt
from typing import List 

def softmax(x):
    epsilon = 1e-8
    exp_vals = np.exp(x - np.max(x)) 
    return exp_vals / (np.sum(exp_vals, axis=0) + epsilon)

def get_activation_func(name: str):
    """
    Args:
        x: A 1-D numpy array.
    
    Returns:
        A 1-D numpy array containing the softmax values.
    """   
    if name == 'sigmoid':
        return lambda x: 1 / (1 + np.exp(-x))
    elif name == 'relu':
        return lambda x: np.maximum(0, x)
    elif name == 'tanh':
        return lambda x: np.tanh(x)
    else:
        raise KeyError(f'No such activation function: {name}')

def get_activation_de_func(name: str):
    if name == 'sigmoid':
        return lambda x: np.exp(x) / (1 + np.exp(x))**2
    elif name == 'relu':
        return lambda x: np.where(x > 0, 1, 0)
    elif name == 'tanh':
        return lambda x: 1 - np.tanh(x)**2
    else: 
        raise KeyError(f'No such activation function: {name}')


class NeuralNet(object): 

    def __init__(self, neuron_cnt: List[int], activation_func: List[str],
                 regularizaion='None'):
        assert len(neuron_cnt) - 2 == len(activation_func) 
        assert regularizaion in ['None', 'weight_decay', 'dropout']

        self.num_layer = len(neuron_cnt) - 1
        self.neuron_cnt = neuron_cnt 
        self.activation_func = activation_func 
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        self.best_W: List[np.ndarray] = []
        self.best_b: List[np.ndarray] = []

        if regularizaion == 'weight_decay':
            self.weight_decay = True
        else:
            self.weight_decay = False
        if regularizaion == 'dropout':
            self.dropout = True
        else: 
            self.dropout = False


        for i in range(self.num_layer): 
            self.W.append( # He initialization method
                np.random.randn(neuron_cnt[i+1], neuron_cnt[i]) * np.sqrt(2 / neuron_cnt[i]))
            self.b.append(np.zeros((neuron_cnt[i+1], 1)))
        
        self.Z_cache = [None] * self.num_layer 
        self.A_cache = [None] * (self.num_layer + 1)
        self.dW_cache = [None] * self.num_layer 
        self.db_cache = [None] * self.num_layer 

    def forward(self, X: np.ndarray, 
                train_mode=True) -> np.ndarray:
        """ 
        X: [n_feas, n_samples], is a 2D array, each colum is a sample, each row is a feature
        A_0 = X
        A_1 <- g_1(Z_1) = g_1(W_1*A_0 + b_1) 
        cache A_1, Z_1, W_1, b_1
        ...
        A_l <- g_l(Z_l) = g_l(W_l*A_{l-1} + b_l)
        cache A_l, Z_l, W_l, b_l
        """
        if train_mode: 
            self.m = X.shape[1]
        A = X 
        self.A_cache[0] = A 
        for i in range(self.num_layer):
            Z = np.dot(self.W[i], A) + self.b[i] 
             
            if i == self.num_layer - 1:
                A = softmax(Z)
            else:
                A = get_activation_func(self.activation_func[i])(Z)
            
            if train_mode and self.dropout and i < self.num_layer - 1:
                keep_prob = 0.9     # 保留概率keep_prob
                d = np.random.rand(*A.shape) < keep_prob 
                A = A * d / keep_prob
            
            if train_mode:
                self.Z_cache[i] = Z 
                self.A_cache[i+1] = A    
        return A 
    
    def backward(self, Y: np.ndarray) -> np.ndarray:   
        """ Y is a 2D array, each colum is a sample, each row is a feature
        dA_l <- dA_l * g_l'(Z_l) = dA_l * g_l'(W_l*A_{l-1} + b_l)
        dW_l <- dA_l * A_{l-1}.T / m
        db_l <- np.mean(dZ_l, axis=1, keepdims=True)
        update dW_l, db_l
        dA_{l-1} <- W_l.T * dZ 
        """
        epsilon = 1e-8
        # dA = -Y / (self.A_cache[-1] + epsilon)+ (1 - Y) / (1 - self.A_cache[-1] + epsilon)
        assert self.m == Y.shape[1] 

        for i in range(self.num_layer - 1, -1, -1):
            if i == self.num_layer - 1:
                dZ = self.A_cache[-1] - Y
            else:
                dZ = dA * get_activation_de_func(self.activation_func[i])(
                    self.Z_cache[i])
            dW = np.dot(dZ, self.A_cache[i].T) / self.m 
            db = np.mean(dZ, axis=1, keepdims=True)
            dA = np.dot(self.W[i].T, dZ)
            self.dW_cache[i] = dW 
            self.db_cache[i] = db 

    def _gradient_descent(self, lr: float):
        for i in range(self.num_layer):
            if self.weight_decay:
                LAMBDA = 4 
                self.W[i] -= lr * (self.dW_cache[i] + LAMBDA * self.W[i] / self.m) 
                self.b[i] -= lr * self.db_cache[i]
            else:
                self.W[i] -= lr * self.dW_cache[i]
                self.b[i] -= lr * self.db_cache[i]

    def _loss_function(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """cross entropy loss"""
        epsilon = 1e-8  # Small epsilon value to prevent taking the logarithm of zero
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)  # Clip probabilities to avoid extreme values
        
        if self.weight_decay:
            LAMBDA = 4
            tot = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            for i in range(self.num_layer):
                tot += np.sum(self.W[i] * self.W[i]) * LAMBDA / 2 / self.m
            return tot
        else:
            return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))  
    
    def fit(self, X, y, n_iters=200, lr=0.001, 
            verbose=False, display=True, patience=10):
        cur_loss = np.inf
        best_loss = np.inf
        count = 0
        loss_history = []
        train_acc_list = []

        for i in range(n_iters):
            y_hat = self.forward(X)
            self.backward(y)
            self._gradient_descent(lr)
            cur_loss = self._loss_function(y_hat, y)
            loss_history.append(cur_loss)

            if cur_loss < best_loss or i == 0:
                best_loss = cur_loss
                self.best_W = [w.copy() for w in self.W]
                self.best_b = [b.copy() for b in self.b]
                count = 0
            else:
                count += 1 
                if count >= patience:
                    print(f"Early stopping afer {i} iterations, loss: {cur_loss}")
                    break
            
            if verbose and i % 20 == 0:
                print(f'Epoch {i}/{n_iters}, loss: {cur_loss}')

            train_acc = (y_hat.argmax(axis=0)==y).sum() / y.shape[1]
            train_acc_list.append(train_acc)
        print(f'Finished training, best loss: {best_loss}')
        if display:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].plot(loss_history)
            axs[0].set_title('Loss history')
            axs[1].plot(train_acc_list)
            axs[1].set_title('Train accuracy')
            plt.show()
        return loss_history
       
    def predict(self, X, y, return_loss=False):
        y_hat = self.forward(X, train_mode=False)
        y_pred = np.argmax(y_hat, axis=0)
        accuracy = np.mean(y_pred == y)
        if return_loss:
            loss = self._loss_function(y_hat, y)
            return y_pred, accuracy, loss 
        else:
            return y_pred, accuracy

    def save(self, filepath: str):
        """filepath: str, best_mode.npz"""
        save_dict = {} 
        for i in range(len(self.best_W)):
            save_dict['W' + str(i+1)] = self.best_W[i]
        for i in range(len(self.best_b)):    
            save_dict['b' + str(i+1)] = self.best_b[i]
        np.savez(filepath, **save_dict)
    
    def load(self, filepath: str):
        """filepath: str, load_mode.npz"""
        params = np.load(filepath) 
        for i in range(len(self.W)): 
            self.W[i] = params['W' + str(i+1)]
        for i in range(len(self.b)):
            self.b[i] = params['b' + str(i+1)]

def data_normalization(x, axis=0):
    return (x - x.mean(axis=axis)) / x.std(axis=axis)

def one_hot_encoding(y, c: int=3):
    """y: [1, n_samples]"""
    label = np.zeros((c, y.shape[1]))
    for i in range(y.shape[1]):
        label[y[0, i], i] = 1
    return label 

if __name__ == '__main__':

    X_train = np.loadtxt('../Exam/train/x.txt')
    y_train = np.loadtxt('../Exam/train/y.txt')
    X_test = np.loadtxt('../Exam/test/x.txt')
    y_test = np.loadtxt('../Exam/test/y.txt')
    # X_train = np.loadtxt('../Iris/train/x.txt')
    # y_train = np.loadtxt('../Iris/train/y.txt')
    # X_test = np.loadtxt('../Iris/test/x.txt')
    # y_test = np.loadtxt('../Iris/test/y.txt')

    # print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
    # print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')
    n_samples, input_size = X_train.shape
    label_cnt = int(np.max(y_train)) + 1

  
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o',
                cmap=plt.get_cmap('Set1', label_cnt))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', 
                cmap=plt.get_cmap('Set1', label_cnt))
    plt.colorbar()
    plt.legend(['Train', 'Test'])
    plt.title('Data for training and testing')
    plt.show()
    
    hidden_size = 10
    output_size = label_cnt
    n_epochs = 200
    learning_rate = 0.001
    batch_size = 40

    # x -> neu1 -> relu -> neu2 -> relu -> neu3 -> output -> softmax
    neuron_list = [input_size, hidden_size, hidden_size, output_size]
    activation_list = ['relu', 'relu']

    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1) 

    X_train = data_normalization(X_train, axis=0).reshape(input_size, -1)
    # y_train = one_hot_encoding(y_train, c=label_cnt)
    X_test = data_normalization(X_test, axis=0).reshape(input_size, -1)
    # y_test = one_hot_encoding(y_test, c=label_cnt)

    plt.figure(figsize=(10, 5))
    plt.scatter(X_train[0, :], X_train[1, :], c=y_train[:], marker='o',
                cmap=plt.get_cmap('Set1', label_cnt))
    plt.scatter(X_test[0, :], X_test[1, :], c=y_test[:], marker='x', 
                cmap=plt.get_cmap('Set1', label_cnt))
    plt.colorbar()
    plt.legend(['Train', 'Test'])
    plt.title('Data for training and testing after normalizion')
    plt.show()

    model = NeuralNet(neuron_list, activation_list)
    loss_history = model.fit(X_train, y_train, n_iters=n_epochs, lr=learning_rate, verbose=True, display=True)
    y_pred, accuracy = model.predict(X_test, y_test, return_loss=False)
    print(f'Test accuracy: {accuracy}')
    print(y_pred)
