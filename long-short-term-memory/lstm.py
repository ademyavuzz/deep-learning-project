from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch._decomp.decompositions import embedding

# verimizi bu sekilde olusturduk.
text = """
Bu ürün beklentimi fazlasıyla karşıladı. İlk gün biraz tereddüt ettim. Sonra iyi ki almışım dedim. Malzeme kalitesi ilk dokunuşta bile kendini belli ediyor. Kalite kalite diye boşuna dememişler. Kullanımı pratik. Pratik olması gerçekten önemli. Günlük hayatta işimi kolaylaştırdı. Bazen hızlıca kullanıyorum. Bazen uzun uzun deniyorum. Her seferinde aynı sonuç. Sonuç güzel. Güzel güzel ilerliyor. Tasarım sade. Sade ama şık. Şık duruyor. Evde de uyuyor. İşte de uyuyor. Kargo hızlıydı. Paketleme özenliydi. Özenli özenli sarılmış. Kutuyu açınca içi düzenliydi. Düzenli olması hoşuma gitti. Parçalar sağlam. Sağlam duruyor. Duruyor duruyor bozulmuyor. Bir iki kere yanlış kullandım. Yine de sorun çıkarmadı. Ses seviyesi düşük. Düşük olunca rahat ediyorsun. Rahat rahat kullanıyorsun. Kullanırken elimi yormadı. Yormaması iyi. İyi iyi dedirtti. Bazı yerleri biraz karışık geldi. Karışık gibi ama alışınca kolay. Kolay olunca keyifli. Keyifli keyifli devam. Temizliği kolay. Kolay olduğu için sürekli ertelemiyorum. Ertelemiyorum çünkü uğraştırmıyor. Uğraştırmıyor ve iş görüyor. İş görüyor iş görüyor. Fiyatı da fena değil. Fena değil demek az bile. Fiyatına göre kesinlikle tavsiye ederim! Bazı detaylar beklediğimden iyi çıktı. İyi çıkınca şaşırdım. Şaşırdım ama memnun kaldım. Memnun memnun kullanıyorum. Bir arkadaşım da aldı. O da beğendi. Beğendi beğendi. Bazen ufak şeyler can sıkabilir. Ama genel tablo güzel. Güzel olduğu için tekrar alırım. Tekrar alırım!"""

words = text.replace(".", "").replace("!", "").lower().split()
word_counts = Counter(words)  # kelime sayilari

# vocab aslinda bizim kac tane kelimemiz b=ve her bir kelimeden ne kadar var bunu verir.
vocab = sorted(word_counts, key=word_counts.get, reverse=True)

word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}

# egitim verisini olusturduk.
data = [(words[i], words[i + 1]) for i in range(len(words) - 1)]


# LSTM model tanimlama
# #embeddig, lstm, fc katmanlari
class LongShortTermMemory(nn.Module):  # bir ust sinifin constractirini cagirma.
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LongShortTermMemory, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # embedding katmani
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)  # LSTM katmani
        self.fc = nn.Linear(
            hidden_dim, vocab_size
        )  # Fully Connected Layer --> Embedding, LSTM ve Fully Connected katmanlarindan olusur.

    def forward(self, x):  # ileri besleme fonksiyonu
        x = self.embedding(
            x
        )  # input embedding katmanindna geciyor. input --> embdedding
        lstm_out, _ = self.lstm(x.view(1, 1, -1))  # embedding --> lstm
        output = self.fc(lstm_out.view(1, -1))  # lstm --> output
        return output


model = LongShortTermMemory(len(vocab), 8, 32)


# tensore cevirme islemi
# #burada verileri alip hepsini bir tensore cevirecegiz.
def prepare_sequence(sequence, to_index):
    return torch.tensor([to_index[word] for word in sequence], dtype=torch.long)


# Hyperparameter Tuning Kombinasyonlarinin Belirle
embedding_sizes = [8, 16]  # giristeki token vektoru boyutu
hidden_size = [16, 32, 64]  # gizli katmanin boyutu--> denenecek gizli katman boyutlari
learning_rate = [0.001, 0.01, 0.005]  # ogrenme orani

best_loss = float(
    "inf"
)  # en dusuk kayip degerini saklamak icin bir degisken. --> en dusuk loss degerni saklyacagim.
best_params = {}  # en iyi paramaetreleri saklamak icin bos bir dictionary , en iyi parametreleri saklayacagim.

print("Hyperparameter Tuning Basliyor...")
# grid search kismina geceis yapabiliriz.

for embedding_size, hidden_size, learning_rate in product(
    embedding_sizes, hidden_size, learning_rate
):
    print(
        f"Deneme: Embedding: {embedding_size}, Hidden: {hidden_size}, Learning_Rate: {learning_rate}"
    )

    model = LongShortTermMemory(
        vocab_size=len(vocab), hidden_dim=hidden_size, embedding_dim=embedding_size
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 50
    total_loss = 0

    for epoch in range(epochs):
        epoch_loss = 0  # epoch baslangicinda kaybi sifirlayalim.
        for word, next_word in data:
            model.zero_grad()  # gradyanlari sifirla.
            input_tensor = prepare_sequence(
                [word], word_to_index
            )  # girdiyi tensore cevir.
            target_tensor = prepare_sequence(
                [next_word], word_to_index
            )  # girdiyi tensore cevir. Hedef kelimeyi tensore cevirecegiz.
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()  # geri yayilim islmei uygula
            optimizer.step()  # parametreleri guncelle.
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Epochs Loss: {epoch_loss}")
        total_loss = epoch_loss

    # en iyi modeli kaydet.
    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {
            "Embedding Dim": embedding_size,
            "Hidden": hidden_size,
            "Learning_Rate": learning_rate,
        }
    print()

print(f"Best Params: {best_params}")

final_model = LongShortTermMemory(
    vocab_size=len(vocab), embedding_dim=16, hidden_dim=32
)
optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()  # kayip fonksiyonu

print("Final Model Training")

epochs = 100
train_losses = []

for epoch in range(epochs):
    epoch_loss = 0.0
    n_steps = 0

    for word, next_word in data:
        final_model.zero_grad()

        input_tensor = prepare_sequence([word], word_to_index)
        target_tensor = prepare_sequence([next_word], word_to_index)

        output = final_model(input_tensor)
        loss = criterion(output, target_tensor)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_steps += 1

    avg_epoch_loss = epoch_loss / max(n_steps, 1)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(
            f"Final Model Epoch: {epoch + 1}/{epochs}, Avg Loss: {avg_epoch_loss:.5f}"
        )

# visualize
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Avg Loss")
plt.title("Training Loss per Epoch")
plt.show()

# Evaluation - baslangic kelimesi girilecek ve n adet kelime verecek.


def predict_sequence(start_word, num_words):
    current_word = start_word  # su anki kelime baslangic kelimesi olarak ayarlanir.
    output_sequence = [current_word]  # cikti dizisi olarak adlandirabiliriz.

    for _ in range(num_words):  # belirtilen sayida kelime tahmini
        with torch.no_grad():  # gradyan hesaplamasi yapmadan
            input_tensor = prepare_sequence([current_word], word_to_index)
            output = final_model(input_tensor)

            # .item() yerine doğrudan tensor üzerinden argmax alıyoruz
            # Genellikle çıktı (1, vocab_size) şeklindedir, son boyuttan en büyüğü seçeriz.
            predicted_index = torch.argmax(output, dim=-1).item()

            predicted_word = index_to_word[predicted_index]
            output_sequence.append(predicted_word)
            current_word = predicted_word

    return output_sequence


start_word = "iş"
num_words = 10
print(" ".join(predict_sequence(start_word, num_words)))
