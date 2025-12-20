from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

DATA_DIR = Path(__file__).resolve().parent / "data"

# device = torch.device("mps" if torch.mps.is_available() else "cpu")


def get_data_loaders(
    batch_size=64,
):  # batch_size : her itersasyonda islenecek veri sayisi.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]  # 3 kanalli RGB bir gorselle calisiyoruz.
    )

    train_set = torchvision.datasets.CIFAR10(
        download=True, train=True, transform=transform, root="./data"
    )
    test_set = torchvision.datasets.CIFAR10(
        download=True, train=False, transform=transform, root="./data"
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True
    )

    return train_loader, test_loader


# train_loader, test_loader = get_data_loaders(batch_size=64)

# visualize - gorsellestirme
# gorsellestirme icin bir fonksiyon
# veri setinden ornek gorselleri alamk icin bir fonksiyon
# daha sonrasinda veri setinin icerinden ratgele 3-5 tane gorsel alarak bunlari gorsellestirecegiz.

"""
verilerimiz su anda normalize edilmis durumda bir verileri yuklerken bu sekilde aldik.
normalize edilmis bir goruntuyu gorsellestirmek yerine
biz "TERS NORMALIZASYON" islemi yapacagiz.
"""


def imshow(image):
    image = image / 2 + 0.5
    numpy_image = image.numpy()
    plt.imshow(np.transpose(numpy_image, (1, 2, 0)))


def get_sample_data(train_loader):
    images, labels = next(iter(train_loader))
    return images, labels


def visualize(n):
    train_loader, test_loader = get_data_loaders(batch_size=64)
    images, labels = get_sample_data(train_loader)
    plt.figure()
    for i in range(n):
        plt.subplot(1, n, i + 1)
        imshow(images[i])
        plt.title(f"Labels: {labels[i].item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# visualize(5)


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # Convolution da Feature Extraction asamasiydi.
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()  # Aktivasyon fonksiyonu
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # 2x2 boyutunda pooling katmani
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )  # 64 filtreli ikinci convolutional layer.
        self.dropout = nn.Dropout(0.20)

        # Tam bagli katmanlari insaa edecegiz.
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# model = ConvolutionalNeuralNetwork().to(device)

define_loss_and_optimizer = lambda model: (
    [nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001)]
)

# criterion, optimizer = define_loss_and_optimizer(model)


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
            total_loss = total_loss + loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epochs: {epoch + 1}/{epochs}, Loss: {avg_loss}")

    plt.figure()
    plt.plot(
        range(1, epochs + 1),
        train_losses,
        marker="o",
        linestyle="-",
        label="Train Loss",
    )
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# train_model(model, train_loader, criterion=criterion, optimizer=optimizer, epochs=10)


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total}%")


# test_model(model, test_loader)


# main program
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(batch_size=64)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    visualize(5)
    model = ConvolutionalNeuralNetwork().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(
        model, train_loader, criterion=criterion, optimizer=optimizer, epochs=10
    )
    test_model(model, test_loader)
