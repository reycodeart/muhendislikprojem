import pandas as pd
import matplotlib.pyplot as plt

# CSV dosyasını yükleyin
results = pd.read_csv("notebooks/results.csv", names=["Model", "Mean EER", "Standard Deviation"])

# Verileri sayısal türe dönüştür
results["Mean EER"] = pd.to_numeric(results["Mean EER"], errors='coerce')
results["Standard Deviation"] = pd.to_numeric(results["Standard Deviation"], errors='coerce')

# Hatalı değerleri kontrol edin
if results.isnull().values.any():
    print("Hatalı veriler bulundu:")
    print(results[results.isnull().any(axis=1)])
    results = results.dropna()  # Gerekirse hatalı satırları kaldırın

# Bar grafiği oluşturma
plt.figure(figsize=(10, 6))
colors = ['skyblue', 'orange', 'green', 'red', 'purple']
plt.bar(results["Model"], results["Mean EER"], yerr=results["Standard Deviation"], capsize=5, color=colors)
plt.title("Model Performans Karşılaştırması", fontsize=16)
plt.xlabel("Model", fontsize=14)
plt.ylabel("Ortalama EER", fontsize=14)
plt.ylim(0, 0.3)  # Y eksenini genişletin
plt.xticks(rotation=45)

# Çubukların üstüne değerleri yazdırma
for i, value in enumerate(results["Mean EER"]):
    plt.text(i, value + 0.01, f'{value:.4f}', ha='center', fontsize=12)

plt.tight_layout()
plt.show()





