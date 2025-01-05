import csv
import math
import random
import numpy as np
import pandas as pd
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

# Nokta sınıfı (n boyutlu uzayda bir nokta)
class Point:
    def __init__(self, coords):
        self.coords = coords
        self.n = len(coords)

    def __repr__(self):
        return str(self.coords)

# Küme sınıfı (bir küme noktaları ve onların merkezi)
class Cluster:
    def __init__(self, points):
        if len(points) == 0:
            raise Exception("HATALI: Boş küme")
        self.points = points
        self.n = points[0].n
        for p in points:
            if p.n != self.n:
                raise Exception("HATALI: Farklı boyutlarda noktalar")
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        return str(self.points)

    def update(self, points):
        # Kümenin merkezini günceller
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        # Kümenin yeni merkezini hesaplar
        numPoints = len(self.points)
        coords = [p.coords for p in self.points]
        unzipped = zip(*coords)
        centroid_coords = [math.fsum(dList) / numPoints for dList in unzipped]
        return Point(centroid_coords)

# İki nokta arasındaki Öklid uzaklığı
def getDistance(a, b):
    ret = sum(pow((a.coords[i] - b.coords[i]), 2) for i in range(a.n))
    return math.sqrt(ret)

# KMeans algoritması
def kmeans(points, k, cutoff):
    # İlk k merkezi rastgele seç
    initial = random.sample(points, k)
    clusters = [Cluster([p]) for p in initial]
    loopCounter = 0

    while True:
        lists = [[] for _ in clusters]
        loopCounter += 1

        for p in points:
            smallest_distance = getDistance(p, clusters[0].centroid)
            clusterIndex = 0
            for i in range(len(clusters) - 1):
                distance = getDistance(p, clusters[i + 1].centroid)
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i + 1
            lists[clusterIndex].append(p)

        biggest_shift = 0.0
        for i in range(len(clusters)):
            shift = clusters[i].update(lists[i])
            biggest_shift = max(biggest_shift, shift)

        if biggest_shift < cutoff:
            print(f"{loopCounter} iterasyondan sonra yakınsadı.")
            break
    return clusters

# Test işlemleri
def testing(clusters, test_genuine, test_imposter):
    user_scores = []
    imposter_scores = []

    for test in test_genuine:
        min_distance = min(getDistance(test, cluster.centroid) for cluster in clusters)
        user_scores.append(min_distance)

    for test in test_imposter:
        min_distance = min(getDistance(test, cluster.centroid) for cluster in clusters)
        imposter_scores.append(min_distance)

    return user_scores, imposter_scores

# KMeans değerlendirme fonksiyonu
def evaluate():
    eers = []
    k = 3
    cut_off = 0.5

    for subject in subjects:
        # Kullanıcı ve sahtekar verilerini ayır
        genuine_user_data = data.loc[data.subject == subject, "H.period":"H.Return"]
        imposter_data = data.loc[data.subject != subject, :]

        # Eğitim seti
        train = genuine_user_data[:200].values
        points = [Point(list(map(float, p))) for p in train]

        # Test seti
        test_genuine = genuine_user_data[200:].values
        test_imposter = imposter_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"].values
        points_test_genuine = [Point(list(map(float, p))) for p in test_genuine]
        points_test_imposter = [Point(list(map(float, p))) for p in test_imposter]

        # KMeans kümeleme
        clusters = kmeans(points, k, cut_off)

        # Test işlemleri
        user_scores, imposter_scores = testing(clusters, points_test_genuine, points_test_imposter)
        eers.append(evaluateEER(user_scores, imposter_scores))

    return np.mean(eers), np.std(eers)

# Çalıştır
if __name__ == "__main__":
    print("KMeans Algoritması Değerlendirmesi:")
    mean_eer, std_eer = evaluate()
    print(f"Ortalama EER: {mean_eer}")
    print(f"Standart Sapma: {std_eer}")

    with open("../results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["KMeans", mean_eer, std_eer])



