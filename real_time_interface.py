import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import euclidean, mahalanobis
from sklearn.cluster import KMeans
from sklearn.svm import SVC

# Sabit Metinler
TEST_TEXT_1 = "Reyhan kesin Muğla Kütahya"
TEST_TEXT_2 = "Klavye dinamikleri projesi"

# Veri tabanı dosyası yolu
DATA_FILE = "data/DSL-StrongPasswordData.csv"

# Dinamik veri toplama fonksiyonu
def collect_dynamics(input_text):
    """
    Kullanıcının klavye dinamiklerini toplar (simüle edilmiş veri).
    """
    dwell_time = np.random.normal(0.2, 0.05, len(input_text))
    flight_time = np.random.normal(0.1, 0.02, len(input_text) - 1)
    return {"H": dwell_time, "DD": flight_time, "UD": flight_time * 0.8}

# Tüm algoritmaları çalıştırma
def run_all_algorithms(dynamics1, dynamics2):
    """
    Tüm algoritmaları çalıştırır ve sonuçları döner.
    """
    results = {}
    min_len = min(len(dynamics1["H"]), len(dynamics2["H"]))  # Minimum uzunluğu bul
    
    # Dizileri minimum uzunluğa göre kırp
    h1 = dynamics1["H"][:min_len]
    h2 = dynamics2["H"][:min_len]
    dd1 = dynamics1["DD"][:min_len - 1]
    dd2 = dynamics2["DD"][:min_len - 1]

    # SVM - Euclidean Mesafesi
    svm_distance = np.linalg.norm(h1 - h2)
    results["SVM"] = svm_distance

    # KMeans - Ortalama Mutlak Fark
    kmeans_distance = np.mean(np.abs(h1 - h2))
    results["KMeans"] = kmeans_distance

    # Mahalanobis
    try:
        cov_matrix = np.cov(np.vstack([h1, h2]))
        inv_cov = np.linalg.pinv(cov_matrix)
        delta = h1 - h2
        mahal_distance = np.sqrt(np.dot(np.dot(delta, inv_cov), delta.T))
        results["Mahalanobis"] = mahal_distance
    except Exception as e:
        results["Mahalanobis"] = float('inf')  # Hata durumunda sonsuz değer
        print(f"Mahalanobis hesaplama hatası: {e}")

    return results
#----------------------------------------------------------------

def clean_and_fix_data(file_path):
    # Veriyi yükle
    df = pd.read_csv(file_path)

    # Eksik değerleri kontrol et ve hatalı satırları sil
    df_clean = df.dropna()

    # Tüm satırların aynı sütun sayısına sahip olduğundan emin ol
    num_columns = df_clean.shape[1]
    df_clean = df_clean[df_clean.apply(lambda x: len(x.dropna()) == num_columns, axis=1)]

    # Tekrar eden satırları kaldır (subject, sessionIndex ve rep bazında)
    df_clean = df_clean.drop_duplicates(subset=["subject", "sessionIndex", "rep"])

    # Temizlenmiş veriyi kaydet
    df_clean.to_csv("cleaned_data.csv", index=False)
    print("Veri temizlendi ve 'cleaned_data.csv' dosyasına kaydedildi.")

# Kullanım
file_path = "data/DSL-StrongPasswordData.csv"  # Dosya yolu
clean_and_fix_data(file_path)

#----------------------------------------------------------------

# Kullanıcı verisini kaydetme
def save_user_data(username, input_text):
    dynamics = collect_dynamics(input_text)
    new_data = {"subject": username, "sessionIndex": 1, "rep": 1}
    for idx, key in enumerate(["H.period", "DD.period.t", "UD.period.t"]):
        new_data[key] = dynamics["H"][idx] if "H" in key else dynamics["DD"][idx]

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=new_data.keys())
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    messagebox.showinfo("Başarılı", "Kullanıcı verisi kaydedildi!")

# Profili güncelleme
def update_existing_profile(username, input_text):
    df = pd.read_csv(DATA_FILE)
    user_data = df[df["subject"] == username]
    new_dynamics = collect_dynamics(input_text)
    if user_data.empty:
        messagebox.showerror("Hata", "Kullanıcı bulunamadı.")
        return
    mean_profile = user_data.select_dtypes(include=[np.number]).mean()
    updated_dynamics = (mean_profile + new_dynamics["H"][:len(mean_profile)]) / 2
    updated_profile = pd.DataFrame(updated_dynamics).T
    updated_profile["subject"] = username
    updated_profile.to_csv(DATA_FILE, mode='a', index=False, header=False)
    messagebox.showinfo("Başarılı", "Kullanıcı profili güncellendi!")

# En iyi algoritmayı bulma
def find_best_algorithm(results, threshold=0.3):
    best_algorithm = min(results, key=results.get)
    min_distance = results[best_algorithm]
    same_user = min_distance < threshold
    return best_algorithm, min_distance, same_user

# Tüm algoritmaları çalıştırma ve karşılaştırma
def run_all_tests():
    username = username_entry.get()
    input_text_1 = entry_text_1.get()
    input_text_2 = entry_text_2.get()

    if not username or not input_text_1 or not input_text_2:
        messagebox.showerror("Hata", "Lütfen tüm alanları doldurun.")
        return

    dynamics1 = collect_dynamics(input_text_1)
    dynamics2 = collect_dynamics(input_text_2)

    results = run_all_algorithms(dynamics1, dynamics2)
    best_algorithm, min_distance, same_user = find_best_algorithm(results)

    result_text = f"Tüm Algoritma Sonuçları:\n"
    for alg, dist in results.items():
        result_text += f"{alg}: {dist:.3f}\n"
    result_text += f"\nEn iyi algoritma: {best_algorithm} (Mesafe: {min_distance:.3f})\n"
    result_text += "Sonuç: Aynı kullanıcı" if same_user else "Sonuç: Farklı kullanıcı"

    result_label.config(text=result_text, fg="green" if same_user else "red")

# Ana pencere oluştur
root = tk.Tk()
root.title("Klavye Dinamikleri Karşılaştırma ve Profil Güncelleme")

# Kullanıcı adı
tk.Label(root, text="Kullanıcı Adı:").grid(row=0, column=0, padx=10, pady=5)
username_entry = tk.Entry(root)
username_entry.grid(row=0, column=1)

# Test Metni 1
tk.Label(root, text="Test Edilecek Metin 1:").grid(row=1, column=0)
tk.Label(root, text=TEST_TEXT_1, fg="blue").grid(row=1, column=1)
tk.Label(root, text="Giriş Metni 1:").grid(row=2, column=0)
entry_text_1 = tk.Entry(root, width=40)
entry_text_1.grid(row=2, column=1)

# Test Metni 2
tk.Label(root, text="Test Edilecek Metin 2:").grid(row=3, column=0)
tk.Label(root, text=TEST_TEXT_2, fg="blue").grid(row=3, column=1)
tk.Label(root, text="Giriş Metni 2:").grid(row=4, column=0)
entry_text_2 = tk.Entry(root, width=40)
entry_text_2.grid(row=4, column=1)

# Butonlar
tk.Button(root, text="Tüm Algoritmaları Çalıştır", command=run_all_tests).grid(row=5, column=0, columnspan=2, pady=5)
tk.Button(root, text="Yeni Kullanıcı Ekle", command=lambda: save_user_data(username_entry.get(), entry_text_1.get())).grid(row=6, column=0, pady=5)
tk.Button(root, text="Profili Güncelle", command=lambda: update_existing_profile(username_entry.get(), entry_text_1.get())).grid(row=6, column=1, pady=5)

# Sonuç
result_label = tk.Label(root, text="Sonuç burada görünecek.", fg="green")
result_label.grid(row=7, column=0, columnspan=2, pady=10)

root.mainloop()










