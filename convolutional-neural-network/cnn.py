"""
Problem Tanimi: CIFAR10 veri seti ile siniflandirma problemi
CIFAR 10
Convolutional Neural Networks - CNNs - Evrisimli Sinir Aglari
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

device = torch.device("mps" if torch.mps.is_available() else "cpu")

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(train=True, download=True, transform=transform, root='./data')
    test_set = torchvision.datasets.CIFAR10(train=False, download=True, transform=transform, root='./data')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

train_loader, test_loader = get_data_loaders(batch_size=64)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def get_samples_data(train_loader):
    images, labels = next(iter(train_loader))
    return images, labels

def visualize(n):
    plt.figure(figsize=(10, 10))
    images, labels = get_samples_data(train_loader)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        imshow(images[i])
        plt.axis('off')
        plt.title(f"Labels: {labels[i]}")
    plt.tight_layout()
    plt.show()

visualize(10)

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))  # pool2 varsa
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)  # sınıflandırma ise genelde en sonda aktivasyon yok
        return x

model = ConvolutionalNeuralNetwork().to(device)

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
)
criterion, optimizer = define_loss_and_optimizer(model)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epochs: {epoch+1}/{epochs}, Loss: {avg_loss:.3f}")

    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker="o", linestyle="-", label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.legend()
    plt.show()


train_model(model, train_loader, criterion=criterion, optimizer=optimizer, epochs=10)

def test_model(model, loader, dataset):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"{dataset} Accuracy: {(100 * correct / total):.5f}%")

test_model(model, loader = test_loader, dataset = "Test")
test_model(model, loader = train_loader, dataset = "Train")

