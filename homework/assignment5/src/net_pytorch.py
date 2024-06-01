# net_pytorch.py 
"""A Forward Neural Network with PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class NeualNet(nn.Module):
    """A simple neural network."""
    def __init__(self, input_size, hidden_size, output_size):
        super(NeualNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x

if __name__ == '__main__':

    X_train = np.loadtxt('../Exam/train/x.txt')
    y_train = np.loadtxt('../Exam/train/y.txt')
    X_test = np.loadtxt('../Exam/test/x.txt')
    y_test = np.loadtxt('../Exam/test/y.txt')
    # X_train = np.loadtxt('../Iris/train/x.txt')
    # y_train = np.loadtxt('../Iris/train/y.txt')
    # X_test = np.loadtxt('../Iris/test/x.txt')
    # y_test = np.loadtxt('../Iris/test/y.txt')

    n_samples, input_size = X_train.shape

    # Hyperparameters
    hidden_size = 10
    output_size = int(np.max(y_train)) + 1
    n_epochs = 500
    learning_rate = 0.001
    batch_size = 40

    def init_display():
        plt.figure(figsize=(10, 5))
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o',
                    cmap=plt.get_cmap('Set1', output_size))
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', 
                    cmap=plt.get_cmap('Set1', output_size))
        plt.colorbar()
        plt.legend(['Train', 'Test'])
        plt.title('Data for training and testing')
        plt.show()
    init_display()


    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    train_acc_list = []
    loss_history = []

    print(X_train.shape, y_train.shape)
    # device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NeualNet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i in range(n_epochs):
        X_train = X_train.reshape(-1, input_size).to(device)
        y_train = y_train.to(device)
        # forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{i+1}/{n_epochs}], Loss: {loss.item():.4f}')
        
        train_acc = (outputs.argmax(dim=1) == y_train).sum().item() / n_samples
        train_acc_list.append(train_acc) 
        loss_history.append(loss.item())
           
    print('Finished Training')

    # Test the model
    with torch.no_grad():
        X_test = X_test.to(device)
        y_pred = model(X_test)
        _, y_pred = torch.max(y_pred, 1)
        y_pred = y_pred.cpu().numpy()
        acc = np.mean(y_pred == y_test)
        # print(f'Predicted: {y_pred}')
        print(f'Accuracy: {acc:.4f}')
    
    def res_display():
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(loss_history)
        axs[0].set_title('Loss History')
        axs[1].plot(train_acc_list)
        axs[1].set_title('Train Accuracy')
        plt.show()
    res_display()


