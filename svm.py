import numpy as np
import csv
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.svm import OneClassSVM

# Veri dosyasını yükleme
data = pd.read_csv("data/DSL-StrongPasswordData.csv")

# Kullanıcılar
subjects = data["subject"].unique()

# EER hesaplama fonksiyonu
def evaluateEER(user_scores, imposter_scores):
    labels = [0] * len(user_scores) + [1] * len(imposter_scores)
    scores = user_scores + imposter_scores
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Eğer ROC eğrisi oluşturulamıyorsa, `nan` döndür
    if len(fpr) < 2 or len(tpr) < 2:
        return float('nan')

    missrates = 1 - tpr
    farates = fpr
    dists = missrates - farates

    if len(dists[dists >= 0]) == 0 or len(dists[dists < 0]) == 0:
        return float('nan')  # Geçersiz durumda `nan` döndür

    idx1 = np.argmin(dists[dists >= 0])
    idx2 = np.argmax(dists[dists < 0])
    x = [missrates[idx1], farates[idx1]]
    y = [missrates[idx2], farates[idx2]]

    if (y[1] - x[1] - y[0] + x[0]) == 0:
        return float('nan')  # Bölme hatasını engellemek için

    a = (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0])
    eer = x[0] + a * (y[0] - x[0])
    return eer


# SVM tabanlı algılama değerlendirme fonksiyonu
def evaluate():
    eers = []

    for subject in subjects:
        user_scores = []
        imposter_scores = []

        # Kullanıcıya ait veriler
        genuine_user_data = data.loc[data.subject == subject, "H.period":"H.Return"]
        imposter_data = data.loc[data.subject != subject, :]

        # Eğitim ve test verilerini oluşturma
        train = genuine_user_data[:200].values
        test_genuine = genuine_user_data[200:].values
        test_imposter = imposter_data.groupby("subject").head(10).loc[:, "H.period":"H.Return"].values

        # Normalizasyon işlemi
        train_mean = train.mean(axis=0)
        train_std = train.std(axis=0) + 1e-6  # Bölme hatalarını önlemek için epsilon ekledik
        train = (train - train_mean) / train_std
        test_genuine = (test_genuine - train_mean) / train_std
        test_imposter = (test_imposter - train_mean) / train_std

        # SVM modelini tanımlama ve eğitme
        clf = OneClassSVM(kernel='rbf', gamma=0.1)  # Gamma değeri optimize edilmiş
        clf.fit(train)

        # Karar fonksiyonundan skorları hesaplama
        user_scores = -clf.decision_function(test_genuine)
        imposter_scores = -clf.decision_function(test_imposter)

        # Eğitim ve test setlerinin boyutlarını kontrol etme
        print(f"Train set shape: {train.shape}")
        print(f"Test genuine shape: {test_genuine.shape}")
        print(f"Test imposter shape: {test_imposter.shape}")

        # EER hesaplama
        eers.append(evaluateEER(list(user_scores), list(imposter_scores)))

    return np.mean(eers), np.std(eers)


# Çalıştırma
if __name__ == "__main__":
    print("SVM Tabanlı Algılama Değerlendirmesi:")
    mean_eer, std_eer = evaluate()
    print(f"Ortalama EER: {mean_eer}")
    print(f"Standart Sapma: {std_eer}")

    # Sonuçları CSV dosyasına kaydet
    with open("../results.csv", "a", newline="") as f:  # Ana dizindeki CSV dosyasına ekleme yap
        writer = csv.writer(f)
        writer.writerow(["SVM", mean_eer, std_eer])
    



