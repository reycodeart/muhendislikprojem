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

# Mahalanobis Detector sınıfı
class MahalanobisDetector:
    def __init__(self, subjects):
        self.subjects = subjects

    # Eğitim işlemi
    def training(self):
        self.mean_vector = self.train.mean().values
        cov_matrix = np.cov(self.train.T)
        # Küçük bir epsilon değeri eklenerek singular hatasını önleyin
        epsilon = 1e-5
        cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
        self.covinv = np.linalg.inv(cov_matrix)


    # Test işlemi
    def testing(self):
        # Kullanıcı verileri üzerinde test yap
        for i in range(self.test_genuine.shape[0]):
            diff = self.test_genuine.iloc[i].values - self.mean_vector
            cur_score = np.dot(np.dot(diff.T, self.covinv), diff)
            self.user_scores.append(np.sqrt(abs(cur_score)))

        # Sahtekar verileri üzerinde test yap
        for i in range(self.test_imposter.shape[0]):
            diff = self.test_imposter.iloc[i].values - self.mean_vector
            cur_score = np.dot(np.dot(diff.T, self.covinv), diff)
            self.imposter_scores.append(np.sqrt(abs(cur_score)))

    # Model değerlendirmesi
    def evaluate(self):
        eers = []

        for subject in self.subjects:
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
    print("Mahalanobis Detector Değerlendirmesi:")
    mahalanobis_detector = MahalanobisDetector(subjects)
    mean_eer, std_eer = mahalanobis_detector.evaluate()
    print(f"Ortalama EER: {mean_eer}")
    print(f"Standart Sapma: {std_eer}")

    with open("../results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mahalanobis Detector", mean_eer, std_eer])

