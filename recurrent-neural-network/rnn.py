"""
Recurrent Neural Networks - RNNs - Yinelemeli/Tekrarlayan Sinir Aglari
RNNs
Kendi veri setimizi olsuturacagiz.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Veri setimizi olusturduk.
def generate_data(SequentialLength=50, NumberOfSamples=1000):
    X = np.linspace(0, 100, NumberOfSamples)
    y = np.sin(X)
    sequence = []
    targets = []

    for i in range(len(X) - SequentialLength):
        sequence.append(y[i : i + SequentialLength])
        targets.append(y[i + SequentialLength])

    # veriyi gorsellestirdik.
    plt.plot(X, y, label="sin(t)")
    plt.title("Sinus Zaman Dalgasi")
    plt.xlabel("Zaman(radyan)")
    plt.ylabel("Sinus Degeri")
    plt.legend()
    plt.show()

    # cikan dizilerimizi return ettik.
    return np.array(sequence), np.array(targets)


# sequence, targets = generate_data(SequentialLength=50, NumberOfSamples=1000)


# modeli olusturduk.
class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RecurrentNeuralNetwork, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(
            out[:, -1, :]
        )  # son zaman adimindaki ciktiyi al ve fc layera bagla.
        return out


# Hiperparametreler - Hyperparameter
SequentialLength = 50  # input dizisinin boyutu
input_size = 1  # input dizisinini boyutu
hidden_size = 16  # RNN in gizli katmandaki dugum, node sayisi
output_size = 1  # output boyutu, tahmin edilen deger
num_layers = 1  # rnn katman sayisi
epochs = 20  # modeli kac kez tum veri seti uzerinde egitilcegi.
batch_size = 32  # her bir egitim adiminda kac ornegin kullanilacagi
learning_rate = 0.001  # oprimizasyon algoritmasi icin ogrenme hizi

# Veriyi Hazirlama
# Veriyi olusturduk ve tensore cevirdik.
# Tensore cevirip donusum yapmayi unutma.
X, y = generate_data(
    SequentialLength, NumberOfSamples=1000
)  # bu numpy array olarka doner.
X = torch.tensor(X, dtype=torch.float32).unsqueeze(
    -1
)  # pytorvh tensorune cevir ve boyut ekle.
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# Pytorch veri setini olusturacagiz.
dataset = torch.utils.data.TensorDataset(X, y)  # pytorch dataset olsutuma.
# Veri yukleyici olustur.
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# modeli tanimla.
model = RecurrentNeuralNetwork(input_size, hidden_size, output_size, num_layers)

# optimizer ve criterion --> loss hesaplamak icin

define_loss_and_optimizer = lambda model: (
    [nn.MSELoss(), optim.Adam(model.parameters(), lr=learning_rate)]
)

criterion, optimizer = define_loss_and_optimizer(model)


import matplotlib.pyplot as plt


def train_model(
    model,
    dataLoader,
    criterion,
    optimizer,
    epochs=20,
):
    model.train()  # Modeli egitim moduna al

    # Düzeltme: Liste burada olmalı ki her epoch'un verisi üst üste eklensin
    train_losses = []

    for epoch in range(epochs):
        for batch_x, batch_y in dataLoader:
            optimizer.zero_grad()  # gradyanlari sifirla
            pred_y = model(batch_x)  # modelden tahmini al
            loss = criterion(pred_y, batch_y)  # loss hesapla
            loss.backward()  # geri yayilim ile gradyanlari hesapla
            optimizer.step()  # agirliklari guncelle

        # Her epoch bittikten sonra, o epoch'un son batch loss değerini ekle
        train_losses.append(loss.item())
        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Grafik Çizimi
    plt.figure()
    plt.plot(
        range(1, epochs + 1),
        train_losses,
        marker="o",
        linestyle="-",
        label="Train Loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.show()


# Egitimi baslat
train_model(model, dataLoader, criterion, optimizer, epochs=epochs)


###simdi elde edttigmiz sonuclarimizi degerlendirecegiz.
# simdi iki adet test verisi hesaplayacagiz.

X_test = np.linspace(100, 110, SequentialLength).reshape(1, -1)  # ilk test verisi
y_test = np.sin(X_test)  # test degerimizin gercek sonucu.

X_test2 = np.linspace(120, 130, SequentialLength).reshape(1, -1)
y_test2 = np.sin(X_test2)

# pytorch tensorlerine cevirme ve boyut ekleme.
# from numpy to tensor

X_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
X_test2 = torch.tensor(y_test2, dtype=torch.float32).unsqueeze(-1)

# model.eval() #modeli degerlendirme moduna al.
model.eval()
prediction1 = model(X_test).detach().numpy()
prediction2 = model(X_test2).detach().numpy()

# gorsellestirme
plt.figure()
plt.plot(np.linspace(0, 100, len(y)), y, marker="o", label="Training Data Set")
plt.plot(X_test.numpy().flatten(), marker="o", label="Test 1")
plt.plot(X_test2.numpy().flatten(), marker="o", label="Test2")
plt.plot(
    np.arange(SequentialLength, SequentialLength + 1),
    prediction1.flatten(),
    "ro",
    label="Prediction1",
)
plt.plot(
    np.arange(SequentialLength, SequentialLength + 1),
    prediction2.flatten(),
    "ro",
    label="Prediction2",
)
plt.legend()
plt.show()
