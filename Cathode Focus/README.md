# AMARUS: CGCNN Multi-Property Screening Platform for Li-S Battery Cathode Hosts 🔋⚡

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-ff4b4b.svg)](https://streamlit.io/)
[![Pymatgen](https://img.shields.io/badge/Pymatgen-Crystal%20Structure-green.svg)](https://pymatgen.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AMARUS** (*Advanced Material Architectures for Rechargeable Ultra-Storage*) adalah platform riset komputasional berbasis **Crystal Graph Convolutional Neural Network (CGCNN)** yang dirancang untuk melakukan penapisan (*screening*) cepat dan evaluasi multi-properti pada material katoda kandidat untuk **Baterai Lithium-Sulfur (Li-S)** generasi mendatang.

---

## 🌟 Fitur Utama

- **Multi-Property Predictive Modeling**: Prediksi simultan 5 properti fisik & termodinamika kritis:
  1. **Band Gap ($E_g$)** [eV] — Penilaian sifat konduktivitas elektronik.
  2. **Formation Energy ($E_f$)** [eV/atom] — Evaluasi stabilitas termodinamika kristal.
  3. **Bulk Modulus ($K$)** [GPa] — Ketahanan terhadap deformasi volume.
  4. **Shear Modulus ($G$)** [GPa] — Ketahanan terhadap deformasi geser.
  5. **Polysulfide Adsorption Energy ($E_{ads}$)** [eV] — Kemampuan penjangkaran molekul polisulfida ($\text{Li}_2\text{S}_x$) untuk memitigasi *shuttle effect*.
- **Interactive 3D Visualizer**: Rendering struktur kristal 3D interaktif berbasis `py3Dmol` dan `stmol` dengan kontrol supercell dinamis ($n_a, n_b, n_c$).
- **Atomic Graph Representation**: Visualisasi graf kristal CGCNN (atom = node, ikatan tetangga = edge) sesuai radius cutoff 8.0 Å.
- **Exploratory Data Analysis (EDA)**: Dashboard analitik komprehensif dengan breakdown kuantitatif material berbasis 5 pilar properti fisika.
- **Publication-Ready Figures**: Integrasi diagram skematik dan mekanisme reaksi elektrokimia berstandar Wiley & Chemistry Europe.

---

## 📁 Struktur Repositori

```text
AMARUS/
├── app.py                             # Wrapper entry point utama Streamlit
├── requirements.txt                   # Daftar dependensi Python
├── README.md                          # Dokumentasi resmi repositori
├── .gitignore                         # Konfigurasi pengabaian berkas sensitif/cache
│
├── models/                            # Arsitektur & Checkpoint Deep Learning
│   ├── cgcnn_model.py                 # Implementasi PyTorch CGCNN (Graph Builder & Network)
│   └── cgcnn_model.pt                 # Checkpoint bobot trained multi-property model
│
├── data/                              # Dataset & Cache Profiling
│   └── dataset_jarvis_dft3d_matched.pkl # Dataset multi-properti JARVIS-DFT3D
│
├── notebooks/                         # Pipeline Riset & Eksperimen Jupyter Notebook
│   ├── LiS_Material_Screening_Pipeline.ipynb # Pipeline screening material Li-S utama
│   ├── JARVIS_DFT3D_Data_Extraction.ipynb    # Ekstraksi & preprocessing data DFT3D
│   └── DFT_GPAW_TPMS_Calculation.ipynb       # Perhitungan Ab-initio GPAW DFT
│
├── scripts/                           # Skrip Eksekusi Modul & Dashboard UI
│   ├── app.py                         # Logika utama dasbor antarmuka Streamlit
│   ├── build_new_notebook.py          # Skrip penyusun notebook riset
│   └── build_and_run_notebook.py      # Eksekutor otomatisasi pipeline
│
└── structures/                        # Berkas Struktur Kristal CIF
    └── Graphene_TPMS_Sheet/           # Topologi Graphene TPMS (P, G, D, I-WP, Neovius)
```

---

## 🚀 Panduan Instalasi & Eksekusi Lokal

### 1. Kloning Repositori
```bash
git clone https://github.com/amar-codes02/CGCNN-MULTI-PROPERTY.git
cd CGCNN-MULTI-PROPERTY
```

### 2. Buat Lingkungan Virtual (Virtual Environment)
```bash
python -m venv venv
source venv/bin/activate  # Untuk Linux/macOS
# Pada Windows gunakan: venv\Scripts\activate
```

### 3. Instal Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Jalankan Dashboard Streamlit
```bash
streamlit run app.py
```
Aplikasi akan secara otomatis terbuka di peramban web Anda pada alamat `http://localhost:8501`.

---

## 💻 Modul Dasbor Interactive UI

Platform AMARUS memiliki 4 Tab Navigasi Utama:

1. **🧬 Tab 1: Fondasi Sains & Reaksi Elektrokimia Li-S**
   - Menjelaskan mekanisme reaksi reduksi 4-tahap sulfur ($S_8 \to \text{Li}_2\text{S}$).
   - Menyajikan analisis fenomena korosi anoda parasitik dan *shuttle effect*.
   - Menyematkan grafik publikasi jurnal (Figure 1: Komparasi Baterai Li-S vs Li-ion & Figure 2: Tantangan Degradasi Baterai).

2. **🏆 Tab 2: Hasil Pengujian TPMS & Ranking Multi-CIF**
   - Evaluasi performa penapisan pada 5 struktur Triply Periodic Minimal Surface (TPMS) Graphene.
   - Fitur upload multi-file CIF kustom untuk prediksi properti serentak.

3. **🧊 Tab 3: Visualisasi Kristal & Graph 3D**
   - Visualisasi kristal 3D interaktif dengan pengubah ukuran *supercell* dinamis.
   - Representasi graf atomik CGCNN yang memperlihatkan konektivitas ikatan antar atom.

4. **📊 Tab 4: Dashboard Analisis Exploratory Data Analytics (EDA)**
   - Distribusi statistik dan klasifikasi material berdasarkan 5 pilar properti fisika (Band Gap, Formation Energy, Bulk Modulus, Shear Modulus, dan Adsorption Energy).

---

## 🧠 Arsitektur Model CGCNN

Model **Crystal Graph Convolutional Neural Network (CGCNN)** mengonversi struktur kristal padat menjadi graf atomik di mana:
- **Node Feature ($v_i$)**: Vektor fitur atomik 114-dimensi (nomor atom, elektronegativitas, jari-jari atom, valensi, dll.).
- **Edge Feature ($u_{ij}$)**: Vektor ekspansi fungsi Gaussian dari jarak antar-atom $r_{ij}$ dengan batas radius cutoff $R_c = 8.0\text{ \AA}$.
- **Convolutional Layer**: Memperbarui representasi atomik melalui *neighboring message passing* dengan fungsi aktivasi non-linear dan *gated mechanism*:

$$v_i^{(t+1)} = v_i^{(t)} + \sum_{j \in N(i)} \sigma(z_{ij}^{(t)} W_f + b_f) \odot g(z_{ij}^{(t)} W_g + b_g)$$

di mana $z_{ij}^{(t)} = v_i^{(t)} \oplus v_j^{(t)} \oplus u_{ij}$.

---

## 📜 Lisensi & Kontribusi

Proyek ini dilisensikan di bawah **MIT License**. Hak cipta penuh atas data riset dan pengembangan materi publikasi dikelola oleh tim peneliti AMARUS.