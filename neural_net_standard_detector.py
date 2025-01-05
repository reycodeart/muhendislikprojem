# Uyarıları devre dışı bırakma
import csv
from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPRegressor  # Güncel kütüphane ile

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

# Neural Network Standard Detector sınıfı
class NeuralNetStandardDetector:
    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.subjects = subjects
        self.learning_rate = 0.0001  # Öğrenme oranı
        self.training_epochs = 500  # Eğitim iterasyon sayısı
        self.n_hidden = 21  # Gizli katmandaki birim sayısı

    # Eğitim işlemi
    def training(self):
        # Model tanımlama
        self.nn = MLPRegressor(
            hidden_layer_sizes=(self.n_hidden,),
            learning_rate_init=self.learning_rate,
            max_iter=self.training_epochs,
            random_state=42
        )
        # Eğitim verilerini modele öğret
        self.nn.fit(
            np.vstack([self.train, self.test_imposter]),
            np.concatenate([np.ones(self.train.shape[0]), np.zeros(self.test_imposter.shape[0])])
        )


    # Test işlemi
    def testing(self):
        # Gerçek kullanıcı verileri için test
        preds = 1 - self.nn.predict(np.array(self.test_genuine))
        for tmp in preds:
            self.user_scores.append(tmp)

        # Sahte kullanıcı verileri için test
        preds = 1 - self.nn.predict(np.array(self.test_imposter))
        for tmp in preds:
            self.imposter_scores.append(tmp)

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
    print("NeuralNet Standard Detector Değerlendirmesi:")
    nn_standard_detector = NeuralNetStandardDetector(subjects)
    mean_eer, std_eer = nn_standard_detector.evaluate()
    print(f"Ortalama EER: {mean_eer}")
    print(f"Standart Sapma: {std_eer}")

    with open("../results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["NeuralNet Standard Detector", mean_eer, std_eer])
