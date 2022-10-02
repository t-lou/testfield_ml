#!/bin/env python3
import numpy
import torch
import torchvision


class ANN(torch.nn.Module):

    def __init__(self):
        super(ANN, self).__init__()
        n_hidden = 128
        self.fc1 = torch.nn.Linear(28 * 28, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

    def __str__(self):
        return 'ANN'


class DNN(torch.nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

    def __str__(self):
        return 'DNN'


def norm(data, mean=None, std=None):
    data = data.double() / 255.0
    if mean is None:
        mean = data.sum() / numpy.array(data.shape).prod()
    if std is None:
        std = data.std()
    data = (data - mean) / std
    return data, mean, std


def train(model, device, data_train, optimizer):
    model.train()
    for data, label in data_train:
        optimizer.zero_grad()

        data = data.to(device)
        label = label.to(device)

        pred = model(data)
        loss = torch.nn.functional.nll_loss(pred, label)
        loss.backward()

        optimizer.step()


def test(model, device, data_test):
    model.eval()
    total_loss = 0.0
    num_correct = 0
    with torch.no_grad():
        for data, label in data_test:
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = torch.nn.functional.nll_loss(pred, label).sum().item()
            total_loss += loss

            if label.item() == pred.argmax(keepdim=True).item():
                num_correct += 1

    print(
        f'loss {total_loss / len(data_test)} accuracy {num_correct / len(data_test)}'
    )


# create device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

# create datasets
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
])
data_train = torchvision.datasets.MNIST('data',
                                        train=True,
                                        download=True,
                                        transform=transform)
data_test = torchvision.datasets.MNIST('data',
                                       train=False,
                                       download=True,
                                       transform=transform)

# create data loader
data_valid = torch.utils.data.DataLoader(data_train)
data_train = torch.utils.data.DataLoader(data_train, 16)
data_test = torch.utils.data.DataLoader(data_test)

# create model and optimizer
model = ANN().to(device)
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

# train
test(model, device, data_test)
for epoch in range(10):
    train(model, device, data_train, optimizer)
    test(model, device, data_test)
    scheduler.step()

    torch.save(model.state_dict(), f'mnist_{model}_{epoch}.pt')
