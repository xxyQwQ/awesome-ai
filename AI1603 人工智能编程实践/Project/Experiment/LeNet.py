# Import packages
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Enable GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))


# Read dataset
class MNISTReader(datasets.VisionDataset):
    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.data_label = torch.load(root)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return 20000

    def __getitem__(self, index):
        image, target = self.data_label[index]
        return self.transform(image), target


# LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, 2),
        )

    def forward(self, x):
        x = self.model(x)
        return x


def main():
    # Dataset train1
    train1_dataset = MNISTReader('./ColoredMNIST/train1.pt')
    train1_loader = DataLoader(train1_dataset, batch_size=32, shuffle=True)
    train1_size = len(train1_dataset)
    print("Dataset train1 loaded, size = {}".format(train1_size))

    # Dataset train2
    train2_dataset = MNISTReader('./ColoredMNIST/train2.pt')
    train2_loader = DataLoader(train2_dataset, batch_size=32, shuffle=True)
    train2_size = len(train2_dataset)
    print("Dataset train2 loaded, size = {}".format(train2_size))

    # Dataset test
    test_dataset = MNISTReader('./ColoredMNIST/test.pt')
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    test_size = len(test_dataset)
    print("Dataset test loaded, size = {}".format(test_size))

    # Model parameters
    model = LeNet().to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    learning_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_accuracy = []

    # Training parameters
    total_epoch = 10
    current_progress = 0

    for epoch in range(total_epoch):
        print("---------- Training round {} started ----------".format(epoch + 1))

        # Train
        model.train()
        total_correct = 0
        # train1
        for inputs, targets in train1_loader:
            # Enable GPU
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward propagation
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            # Calculate accuracy
            correct = (outputs.argmax(1) == targets).sum()
            total_correct += correct
            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Record progress
            current_progress += 1
            if current_progress % 100 == 0:
                print("Training progress: {}, Loss: {}".format(current_progress, loss.item()))
        # train2
        for inputs, targets in train2_loader:
            # Enable GPU
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward propagation
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            # Calculate accuracy
            correct = (outputs.argmax(1) == targets).sum()
            total_correct += correct
            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Record progress
            current_progress += 1
            if current_progress % 100 == 0:
                print("Training progress: {}, Loss: {}".format(current_progress, loss.item()))
        # Record accuracy
        accuracy_train = total_correct.item() / 40000

        # Test
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Enable GPU
                inputs, targets = inputs.to(device), targets.to(device)
                # Make prediction
                outputs = model(inputs)
                # Calculate accuracy
                correct = (outputs.argmax(1) == targets).sum()
                total_correct += correct
        # Record accuracy
        accuracy_test = total_correct.item() / 20000
        total_accuracy.append([accuracy_train, accuracy_test])
        print("Train Accuracy: {}, Test Accuracy: {}".format(accuracy_train, accuracy_test))

        print("---------- Training round {} finished ----------".format(epoch + 1))

    # Generate data points
    x = np.linspace(1, 10, 10, dtype=np.int_)
    y = np.array(total_accuracy)

    # Plot learning curve
    plt.figure(figsize=(5, 5))
    plt.plot(x, y[:, 0], c='purple', label='Train')
    plt.plot(x, y[:, 1], c='yellow', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
