# 📘 Judul Proyek
*ANALISIS FAKTOR PENENTU KEAMANAN JAMUR DAN KLASIFIKASI JAMUR BERACUN MENGGUNAKAN MACHINE LEARNING DAN DEEP LEARNING PADA DATASET MUSHROOM UCI*

## 👤 Informasi
- **Nama:** Elin Akma Pratama
- **Repo:** https://github.com/elinakma/UAS---Mushroom-UCI.git   
- **Video:** https://drive.google.com/drive/folders/1wbc-086cQup9Y6eLXty6s8sS_iOgHRsh?usp=sharing 

---

# 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk mengklasifikasi jamur apakah **edible** atau **poisonous** menggunakan dataset *Agaricus-Lepiota*.  
Dalam proyek ini dilakukan tahapan berikut:
- Melakukan data cleaning & preparation  
- Melakukan encoding terhadap fitur kategorikal  
- Membangun tiga model:
  - **Baseline Model:** Logistic Regression  
  - **Advanced ML Model:** Random Forest  
  - **Deep Learning Model:** Multilayer Perceptron (MLP)
- Melakukan evaluasi performa dan membandingkan hasil antar model  
- Menentukan model terbaik   

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- Identifikasi manual terhadap jamur beracun dan tidak beracun sering mengalami kesalahan karena kemiripan morfologi, sehingga diperlukan model yang mampu melakukan klasifikasi secara akurat dan konsisten.
- Dataset Mushroom UCI memiliki fitur kategori yang kompleks dan membutuhkan proses preprocessing yang tepat untuk memastikan model dapat mempelajari pola dengan baik.
- Belum diketahui fitur mana yang paling berpengaruh dalam menentukan keamanan jamur, sehingga dibutuhkan analisis feature importance untuk memahami faktor penentu toksisitas.
- Diperlukan perbandingan performa antara model baseline, model machine learning lanjutan, dan deep learning untuk menentukan pendekatan terbaik dalam klasifikasi jamur beracun.
 

**Goals:**  
- Mengembangkan tiga model berbeda (baseline, machine learning advanced, dan deep learning) untuk melakukan klasifikasi jamur beracun dan tidak beracun menggunakan dataset Mushroom UCI.
- Mencapai akurasi klasifikasi minimal ≥ 80% pada model terbaik sebagai indikator keberhasilan prediksi keamanan jamur.
- Melakukan preprocessing data secara benar, termasuk encoding fitur kategori serta pembagian data train–test, agar model dapat mempelajari pola secara optimal.
- Mengevaluasi dan membandingkan performa ketiga model menggunakan metrik seperti accuracy, precision, recall, dan f1-score untuk menentukan model paling efektif.
- Mengidentifikasi fitur yang paling berpengaruh terhadap klasifikasi jamur beracun melalui analisis feature importance pada model advanced.


---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── mushroom.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
│   └── model_mlp.h5
│
├── images/                 # Visualizations
│   └── Confusion_Matrix_LogisticRegression.png
│   └── Confusion_Matrix_RandomForest.png
│   └── Confusion_Matrix_MLP.png
│   └── EDA_Distribusi_Kelas.png
│   └── EDA_Korelasi_Fitur.png
│   └── EDA_Top10.png
│   └── Perbandingan_Metrik_Model.png
│   └── Visualitation_Loss_Accuracy.png
│
├── requirements.txt        # Dependencies
├── Checklist_Submit.md
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset

- **Dataset:** Mushrooom  
- **Sumber:** UCI Machine Learning Repository  
- **Jumlah data:** 8124 baris  
- **Tipe data:** Seluruh fitur bertipe kategorikal  
- **Target:**  
  - `e` = edible  
  - `p` = poisonous  

### 📌 Fitur Utama Dataset

| **Fitur** | **Deskripsi** |
|----------|---------------|
| **class** | Kelas jamur: edible (bisa dimakan) atau poisonous (beracun) |
| **cap-shape** | Bentuk tudung jamur |
| **cap-surface** | Tekstur permukaan tudung |
| **cap-color** | Warna tudung |
| **bruises** | Apakah jamur berubah warna saat memar |
| **odor** | Bau jamur |
| **gill-attachment** | Cara bilah menempel pada batang |
| **gill-spacing** | Jarak antar bilah |
| **gill-size** | Ukuran bilah |
| **gill-color** | Warna bilah |
| **stalk-shape** | Bentuk batang |
| **stalk-root** | Jenis akar batang |
| **stalk-surface-above-ring** | Tekstur batang di atas cincin |
| **stalk-surface-below-ring** | Tekstur batang di bawah cincin |
| **stalk-color-above-ring** | Warna batang di atas cincin |
| **stalk-color-below-ring** | Warna batang di bawah cincin |
| **veil-type** | Jenis selubung (umumnya satu nilai saja) |
| **veil-color** | Warna selubung |
| **ring-number** | Jumlah cincin pada batang |
| **ring-type** | Jenis cincin |
| **spore-print-color** | Warna cetakan spora |
| **population** | Kelimpahan populasi jamur |
| **habitat** | Lokasi jamur ditemukan |

---

# 4. 🔧 Data Preparation
Tahapan yang dilakukan:
- Mengganti missing value (`?`) menggunakan nilai modus  
- Encoding target (`e → 0`, `p → 1`)  
- One-Hot Encoding seluruh fitur kategorikal  
- Train-test split (80:20, stratify target)  

---

# 5. 🤖 Modeling

### **Model 1 – Baseline**
**Logistic Regression**  
- Parameter default  
- Digunakan sebagai acuan sederhana untuk performa awal  

### **Model 2 – Advanced ML**
**Random Forest Classifier**  
- n_estimators: 200  
- max_depth: None  
- Model lebih kuat dibanding baseline  

### **Model 3 – Deep Learning**
**Multilayer Perceptron (MLP)**  
- Optimizer: Adam  
- Learning rate: 0.001  
- Batch size: 32  
- Epochs: 50  
- Validation split: 0.2  
- Callbacks: EarlyStopping, ReduceLROnPlateau  

---

# 6. 🧪 Evaluation

### **Metrik:** Accuracy, Precision, Recall

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Logistic Regression | 0.9988 | 1.0 | 1.0 |
| Random Forest | 1.0000 | 1.0 | 1.0 |
| Deep Learning | 0.89 | 0.88 | 0.90 |

### Visualisasi Perbandingan
(Gambar bar chart disimpan di folder `images/Perbandingan_Metrik_Model.png`)

---

# 7. 🏁 Kesimpulan
- **Model terbaik:** Random Forest  
- **Performa:**
- Accuracy = 1.00
- Precision = 1.00
- Recall = 1.00

- **Alasan:**  
  - Mampu menangkap pola non-linear dengan sangat baik.
  - Robust terhadap noise dan variabilitas fitur.
  - Kinerjanya stabil pada dataset tabular seperti klasifikasi jamur.
  - Menghasilkan performa sempurna tanpa kehilangan generalisasi.

- **Insight:**  
  - Model sederhana seperti Logistic Regression saja sudah memberikan akurasi sangat tinggi (~99.88%), menandakan dataset ini mudah dipelajari.
  - Model Random Forest mampu mencapai performa sempurna karena dapat menangkap interaksi fitur yang tidak dapat ditangani dengan baik oleh model linear.
  - Model Deep Learning tidak unggul pada dataset tabular kecil seperti ini, sehingga performanya lebih rendah meskipun arsitektur dan hyperparameter sudah dioptimalkan


---

# 8. 🔮 Future Work
- [✅] Tambah variasi data  
- [✅] Hyperparameter tuning lebih lanjut  
- [✅] Mencoba arsitektur Deep Learning yang lebih kompleks  
- [✅] Deployment model (API / Web App)   

---

# 9. 🔁 Reproducibility
Gunakan Environment:
python -m venv environment
environment\Script\activate

Install Dependencies:
pip install -r requirements.txt

Install Library:
pip install pandas seaborn matplotlib scikit-learn
pip install tensorflow