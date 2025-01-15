# Import perpustakaan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Membaca dataset
data = pd.read_excel('Dataset Update pasien Gagal Jantung.xlsx')

# Menampilkan informasi dataset
print("Informasi Dataset:")
print(data.info())
print("\n5 Data Teratas:")
print(data.head())

# 2. Preprocessing Data
# Memisahkan fitur (X) dan target (y)
X = data.drop(columns=['DEATH_EVENT'])  # Semua kolom kecuali target
y = data['DEATH_EVENT']  # Kolom target

# Membagi dataset menjadi data training (80%) dan data testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nJumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))

# 3. Membuat Model Decision Tree (ID3)
# Menggunakan kriteria 'entropy' untuk ID3
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# 4. Evaluasi Model
# Melakukan prediksi pada data testing
y_pred = model.predict(X_test)

# Menampilkan hasil evaluasi
print("\nAkurasi Model:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# 5. Visualisasi Pohon Keputusan
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['Tidak Meninggal', 'Meninggal'], filled=True)
plt.title("Pohon Keputusan (ID3)")
plt.savefig("pohon_keputusan.pdf")
plt.show()