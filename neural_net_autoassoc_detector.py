from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import roc_curve

# Veri dosyasını yükleme
data = pd.read_csv("data/DSL-StrongPasswordData.csv")

# Kullanıcılar
subjects = data["subject"].unique()

# EER hesaplama fonksiyonu
def evaluateEER(user_scores, imposter_scores):
    labels = [0] * len(user_scores) + [1] * len(imposter_scores)
    fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
    missrates = 1 - tpr
    farates = fpr
    dists = missrates - farates
    idx1 = np.argmin(dists[dists >= 0])
    idx2 = np.argmax(dists[dists < 0])
    x = [missrates[idx1], farates[idx1]]
    y = [missrates[idx2], farates[idx2]]
    a = (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0])
    eer = x[0] + a * (y[0] - x[0])
    return eer

# Neural Network Autoencoder tabanlı algılama sınıfı
class NeuralNetAutoAssocDetector:
    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.subjects = subjects
        self.learning_rate = 0.001  # Öğrenme oranını artırdık
        self.training_epochs = 500  # Eğitim için toplam iterasyon sayısı
        self.n_hidden = 50  # Gizli katmandaki birim sayısını artırdık
        self.learning_momentum = 0.9  # Öğrenme momentumu

    # Eğitim işlemi
    def training(self):
        # Eğitim verilerini normalize et
        self.train = (self.train - self.train.mean()) / self.train.std()

        # MLPRegressor tanımlama
        self.nn = MLPRegressor(
            hidden_layer_sizes=(self.n_hidden,),  # Gizli katman boyutu
            learning_rate_init=self.learning_rate,  # Başlangıç öğrenme oranı
            max_iter=1000,  # İterasyon sayısını artırdık
            momentum=self.learning_momentum,  # Öğrenme momentumu
            random_state=42
        )
        self.nn.fit(np.array(self.train), np.array(self.train))

    # Test işlemi
    def testing(self):
        # Test verilerini normalize et
        self.test_genuine = (self.test_genuine - self.test_genuine.mean()) / self.test_genuine.std()
        self.test_imposter = (self.test_imposter - self.test_imposter.mean()) / self.test_imposter.std()

        # Gerçek kullanıcı verileri için test
        preds = self.nn.predict(np.array(self.test_genuine))
        for i in range(self.test_genuine.shape[0]):
            self.user_scores.append(np.linalg.norm(self.test_genuine.iloc[i].values - preds[i]))

        # Sahtekar verileri için test
        preds = self.nn.predict(np.array(self.test_imposter))
        for i in range(self.test_imposter.shape[0]):
            self.imposter_scores.append(np.linalg.norm(self.test_imposter.iloc[i].values - preds[i]))

    # Model değerlendirmesi
    def evaluate(self):
        eers = []

        for subject in subjects:
            self.user_scores = []
            self.imposter_scores = []

            # Kullanıcıya ait veriler
            genuine_user_data = data.loc[data.subject == subject, "H.period":"H.Return"]
            imposter_data = data.loc[data.subject != subject, :]

            # Eğitim ve test setleri
            self.train = genuine_user_data[:200]
            self.test_genuine = genuine_user_data[200:]
            self.test_imposter = imposter_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]

            # Eğitim ve test işlemleri
            self.training()
            self.testing()

            # EER hesapla
            eers.append(evaluateEER(self.user_scores, self.imposter_scores))

        return np.mean(eers), np.std(eers)

# Çalıştırma
if __name__ == "__main__":
    print("NeuralNet AutoAssoc Detector Değerlendirmesi:")
    nn_detector = NeuralNetAutoAssocDetector(subjects)
    mean_eer, std_eer = nn_detector.evaluate()
    print(f"Ortalama EER: {mean_eer}")
    print(f"Standart Sapma: {std_eer}")

    with open("../results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["NeuralNet AutoAssoc Detector", mean_eer, std_eer])


