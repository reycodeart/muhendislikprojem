import numpy as np
import pandas as pd

from pynput import keyboard
import joblib  # Modeli yüklemek için
import time    # Zaman damgaları için

from sklearn.metrics import roc_curve
from scipy.spatial.distance import cityblock, mahalanobis, euclidean

# Veri dosyasını yükleme
data = pd.read_csv("data/DSL-StrongPasswordData.csv")

# Toplam 51 kullanıcı
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

# **Algoritma Sınıfları**

# 1. Euclidean Detector
class EuclideanDetector:
    def __init__(self, subjects):
        self.train = None
        self.test_genuine = None
        self.test_imposter = None
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
        
    def training(self):
        self.mean_vector = self.train.mean().values
        
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = np.linalg.norm(self.test_genuine.iloc[i].values - self.mean_vector)
            self.user_scores.append(cur_score)
            
        for i in range(self.test_imposter.shape[0]):
            cur_score = np.linalg.norm(self.test_imposter.iloc[i].values - self.mean_vector)
            self.imposter_scores.append(cur_score)
    
    def evaluate(self):
        eers = []
        for subject in self.subjects:
            self.user_scores = []
            self.imposter_scores = []

            # Kullanıcı verileri
            genuine_user_data = data.loc[data.subject == subject, "H.period":"H.Return"]
            imposter_data = data.loc[data.subject != subject, :]

            # Eğitim ve test setleri
            self.train = genuine_user_data[:200]
            self.test_genuine = genuine_user_data[200:]
            self.test_imposter = imposter_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
            
            self.training()
            self.testing()

            eers.append(evaluateEER(self.user_scores, self.imposter_scores))
        
        return np.mean(eers), np.std(eers)

# 2. Manhattan Detector
class ManhattanDetector:
    def __init__(self, subjects):
        self.train = None
        self.test_genuine = None
        self.test_imposter = None
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
        
    def training(self):
        self.mean_vector = self.train.mean().values
        
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = cityblock(self.test_genuine.iloc[i].values, self.mean_vector)
            self.user_scores.append(cur_score)
            
        for i in range(self.test_imposter.shape[0]):
            cur_score = cityblock(self.test_imposter.iloc[i].values, self.mean_vector)
            self.imposter_scores.append(cur_score)
    
    def evaluate(self):
        eers = []
        for subject in self.subjects:
            self.user_scores = []
            self.imposter_scores = []

            # Kullanıcı verileri
            genuine_user_data = data.loc[data.subject == subject, "H.period":"H.Return"]
            imposter_data = data.loc[data.subject != subject, :]

            # Eğitim ve test setleri
            self.train = genuine_user_data[:200]
            self.test_genuine = genuine_user_data[200:]
            self.test_imposter = imposter_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
            
            self.training()
            self.testing()

            eers.append(evaluateEER(self.user_scores, self.imposter_scores))
        
        return np.mean(eers), np.std(eers)
    


class RealTimeKeystroke:
    def __init__(self, model_path):
        """
        Gerçek zamanlı keystroke dinleyici.
        model_path: Eğitilmiş modelin yolu
        """
        self.model = joblib.load(model_path)  # Modeli yükle
        self.start_time = {}  # Tuşa basma zamanlarını saklar
        self.features = []    # Toplanan özellikler

    def on_press(self, key):
        """
        Bir tuşa basıldığında zaman kaydet.
        """
        try:
            self.start_time[key] = time.time()
        except Exception as e:
            print(f"Hata (basma): {e}")

    def on_release(self, key):
        """
        Tuş bırakıldığında özellikleri hesapla ve tahmin yap.
        """
        try:
            press_time = self.start_time[key]
            release_time = time.time()
            dwell_time = release_time - press_time
            self.features.append(dwell_time)
            print(f"Tuş: {key}, Basma Süresi: {dwell_time}")
        except KeyError:
            print(f"Tuş {key} başlangıç zamanı bulunamadı.")

        # 5 özellik toplandıysa tahmin yap
        if len(self.features) == 5:
            self.predict()
            self.features = []  # Özellikleri sıfırla

    def predict(self):
        """
        Gerçek zamanlı tahmin yap.
        """
        try:
            data = np.array([self.features])
            prediction = self.model.predict(data)
            print(f"Tahmin Sonucu: {prediction[0]}")
        except Exception as e:
            print(f"Tahmin hatası: {e}")

    def start(self):
        """
        Keystroke dinleyiciyi başlat.
        """
        print("Gerçek zamanlı keystroke dinleyici başlatıldı...")
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()


# Diğer sınıflar için yukarıdaki şablonu kullanarak (ör. MahalanobisDetector vb.) devam edebilirsiniz.

# **Ana Çalışma**
if __name__ == "__main__":
    print("1. Euclidean Detector Değerlendir")
    print("2. Manhattan Detector Değerlendir")
    print("3. Gerçek Zamanlı Tahmin Başlat")
    choice = input("Seçiminizi yapın: ")

    if choice == "1":
        euclidean_detector = EuclideanDetector(subjects)
        print(euclidean_detector.evaluate())
    elif choice == "2":
        manhattan_detector = ManhattanDetector(subjects)
        print(manhattan_detector.evaluate())
    elif choice == "3":
        rt_keystroke = RealTimeKeystroke("random_forest_model.pkl")  # Model yolu
        rt_keystroke.start()
    else:
        print("Geçersiz seçim.")






