import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms 
import torchvision.datasets as datasets
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, Dataset 

from torch.utils.tensorboard import SummaryWriter
import sys

class ConvNetwork(nn.Module):
    # original shape of images [bz, 3, 28, 28]
    # input_layer: 3 input channels, 6 output channels, 5 kernel size 
    def __init__(self):
        super(ConvNetwork, self).__init__() 
        self.conv1 = nn.Conv2d(1, 6, 5) # [bz, 3, 28, 28] -> [bz, 6, 24, 24]
        self.pool = nn.MaxPool2d(2, 2)  # [bz, 6, 24, 24] -> [bz, 6, 12, 12]
        self.conv2 = nn.Conv2d(6, 16, 3)# [bz, 6, 12, 12] -> [bz, 16, 10, 10] 
        # pool: [bz, 16, 10, 10] -> [bz, 16, 5, 5]
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # [bz, 16, 5, 5] -> [bz, 120]
        self.fc2 = nn.Linear(120, 84) # [bz, 120] -> [bz, 84]
        self.fc3 = nn.Linear(84, 10) # [bz, 84] -> [bz, 10]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # flatten the output of conv2 to (batch_size, 16*5*5)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

writer = SummaryWriter("runs/mnist")

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters 
input_size = 784 # 28x28
hidden_size = 50
num_classes = 10
num_epochs = 5
batch_size = 32
learning_rate = 0.001

# MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize((0.5, ), (0.5, ))]
)
 
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True) 
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) 

examples = iter(train_loader) 
images, labels = examples.__next__()
print(f'Size of data, labels: {images.shape}, {labels.shape}')

img_grid = torchvision.utils.make_grid(images)
writer.add_image('MNIST Images', img_grid)

model = ConvNetwork()
# loss and optimizer 
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

# training loop 
n_total_steps = len(train_loader)
running_loss = 0.0 
running_correct = 0 

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device) 
        labels = labels.to(device)

        # forward pass 
        outputs = model(images) 
        loss = criterion(outputs, labels) 
        # backward pass and update weights 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # tensorbard 
        running_loss += loss.item() 
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{len(train_loader)}, loss = {running_loss/100:.3f}')
            writer.add_scalar('training loss', running_loss/100, epoch*n_total_steps + 1)
            writer.add_scalar('accuracy', running_correct/100, epoch*n_total_steps + 1)
            running_loss = 0.0
            running_correct = 0 

# Test 
preds = [] 
pred_labels = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device) 
        labels = labels.to(device) 
        outputs = model(images) 

        # max returns (value, index) 
        _, predicted = torch.max(outputs.data, 1) 
        n_samples += labels.shape[0] 
        n_correct += (predicted == labels).sum().item() 

        # classification results for tensorboard 
        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        # print(predicted, labels)
        preds.append(class_predictions) 
        pred_labels.append(predicted) 

    preds = torch.cat([torch.stack(batch) for batch in preds])
    pred_labels = torch.cat(pred_labels, dim=0) 
    acc = 100.0 * n_correct / n_samples 
    print(f'Accuracy on the testing images = {acc}%')

    for i in range(num_classes):
        labels_i = pred_labels == i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0) 
        
    writer.close()