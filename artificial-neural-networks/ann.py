#import libraries
import torch #pytorch
import torch.nn as nn #Neural Network
import torch.optim as optim #optimizer
import torchvision #dataset and pre-trained model
import torchvision.transforms as transforms #transforms, tensor
import matplotlib.pyplot as plt #visualization
from pathlib import Path

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#Load Data
def get_data_loaders(batch_size=64):
    train_set = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=False, transform=transform)
    test_set  = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#train_loader, test_loader = get_data_loaders(batch_size=64)

#Visualize Samples
def visualize_samples(loader, n):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(1, n, figsize=(n*2, 2))
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

#visualize_samples(train_loader, 10)

#define model and compile
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

#model = NeuralNetwork().to(device)

define_loss_and_optimizer = lambda model :(
    nn.CrossEntropyLoss(),
    optim.Adam(model.parameters(), lr=0.001)
)

#criterion, optimizer = define_loss_and_optimizer(model)

#train model
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    total_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        total_losses.append(avg_loss)
        print(f"Epochs: {epoch + 1}/{epochs}, Loss: {avg_loss}")

    plt.figure()
    plt.plot(range(1, epochs + 1), total_losses, marker="o", linestyle="-", label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.legend(loc="upper right")
    plt.show()

#train_model(model,train_loader, criterion=criterion, optimizer=optimizer, epochs=10)

#test model
def test_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}")

#test_model(model, test_loader)

#main program
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(batch_size=64)
    visualize_samples(train_loader, 10)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = NeuralNetwork().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion=criterion, optimizer=optimizer, epochs=10)
    test_model(model, test_loader)


