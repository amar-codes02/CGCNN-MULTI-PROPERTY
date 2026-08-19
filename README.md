# CGCNN Material Property Predictor — Streamlit App

Aplikasi ini dibuat dari notebook `Final_jarvis_EN.ipynb`. Ada 3 hal yang bisa
diupload lewat browser:

1. **Dataset** (di tab EDA) — file `.pkl` seperti `dataset_jarvis_dft3d.pkl`,
   atau `.csv` dengan kolom `band_gap`, `formation_energy`, `bulk_modulus`,
   `shear_modulus`.
2. **Model checkpoint** (`.pt`, di sidebar) — hasil training dari notebook.
3. **File CIF** (di bagian atas halaman) — material yang mau diprediksi/divisualisasikan.

Kalau dataset atau model belum diupload, aplikasi tetap jalan: tab EDA
menampilkan data demo/sintetis, dan aplikasi otomatis mencoba memakai file di
`models/cgcnn_model.pt` kalau ada.

## 1. Struktur folder

```
streamlit_app/
├── app.py                 # aplikasi utama
├── cgcnn_model.py          # arsitektur CGCNN + graph builder (di-port dari notebook)
├── eda_precompute.py       # opsional: bikin cache EDA lokal (lihat bagian 4)
├── requirements.txt
├── models/
│   └── cgcnn_model.pt      # opsional: kalau ditaruh di sini, tidak perlu upload model tiap buka app
└── data/
    └── eda_cache.csv       # opsional: kalau ditaruh di sini, dipakai sebagai default sebelum upload
```

## 2. Instalasi

```bash
cd streamlit_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Catatan GPU: `pip install torch` di atas menginstall versi CPU secara default.
> Kalau Anda punya GPU dan CUDA terpasang, install torch versi CUDA sesuai
> instruksi di https://pytorch.org/get-started/locally/ sebelum menjalankan app.

## 3. Jalankan aplikasi

```bash
streamlit run app.py
```

## 4. Cara pakai di browser

1. **Tab EDA Dataset** — kosong sampai Anda upload dataset di tab ini:
   - `.pkl` seperti `dataset_jarvis_dft3d.pkl` (dict dengan key `names`,
     `band_gap`, `formation_energy`, `bulk_modulus`, `shear_modulus`), atau
   - `.csv` dengan kolom `band_gap`, `formation_energy`, `bulk_modulus`,
     `shear_modulus` (kolom `name` opsional).
   Begitu file terupload, statistik, histogram, korelasi, scatter plot, dan
   klasifikasi metal/semimetal/semikonduktor/insulator langsung muncul.

2. **Sidebar** — upload file checkpoint model (`cgcnn_model.pt`, hasil
   `MODEL_CHECKPOINT_FILE` dari notebook). Tanpa ini, tab **Prediksi** tidak
   aktif, tapi tab Struktur Kristal & Graph Model tetap bisa dipakai (tidak
   butuh model).

3. Di bagian atas halaman, **upload 1 file `.cif`** milik Anda.

4. **Tab Prediksi Sifat Material** — otomatis menampilkan prediksi 4 sifat
   (band gap, formation energy, bulk modulus, shear modulus).

5. **Tab Struktur Kristal** — visualisasi 3D interaktif (bisa diputar/di-zoom),
   dengan 3 slider (`na`, `nb`, `nc`) untuk mengatur ukuran supercell.

6. **Tab Graph Model** — visualisasi graph (node = atom, edge = ikatan/tetangga)
   yang menjadi input CGCNN, dengan slider radius cutoff & jumlah tetangga
   maksimum untuk eksplorasi (tidak mengubah hasil prediksi, yang selalu
   memakai radius=8.0 Å dan maks. 12 tetangga sesuai training).

## 5. (Opsional) Skip upload berulang

Kalau Anda malas upload dataset/model tiap kali buka app, taruh saja:
- checkpoint di `models/cgcnn_model.pt`
- cache EDA (`.csv`, buat dengan `python eda_precompute.py`) di `data/eda_cache.csv`

Aplikasi akan otomatis memakainya sebagai default bila belum ada file yang
diupload lewat UI pada sesi tersebut.

## 6. Troubleshooting: "Paket py3Dmol dan stmol belum terinstall"

Kalau tab **Struktur Kristal** menampilkan pesan ini padahal sudah `pip install`,
biasanya penyebabnya salah satu dari ini:

1. **Install di environment yang salah.** Kalau pakai virtualenv/conda, pastikan
   `pip install` dan `streamlit run` dijalankan di environment (venv) yang sama.
   Cek dengan:
   ```bash
   which python
   which streamlit
   python -c "import py3Dmol, stmol; print('OK')"
   ```
   Kalau baris terakhir error, berarti env-nya beda.

2. **Versi tidak cocok.** `stmol` cukup sensitif terhadap versi `py3Dmol`.
   Install versi yang sudah pasti kompatibel (sudah dipin di `requirements.txt`):
   ```bash
   pip uninstall -y py3Dmol stmol
   pip install py3Dmol==2.0.4 stmol==0.0.9 ipython_genutils
   ```

3. **Streamlit Cloud / server tanpa restart.** Kalau deploy di Streamlit Cloud,
   pastikan `py3Dmol`, `stmol`, `ipython_genutils` ada di `requirements.txt`
   lalu **reboot app** (Manage app → Reboot), bukan cuma refresh browser --
   package baru butuh rebuild environment.

4. Setelah install, **restart proses `streamlit run`** (bukan cuma refresh
   tab browser) supaya modul barunya ke-load.



- Prediksi hanya seakurat model yang di-training (lihat evaluasi R²/MAE di
  notebook asli). Untuk material yang jauh berbeda dari domain JARVIS-DFT,
  hasil bisa kurang akurat.
- Skor kesesuaian seperti "host katoda Li-S" di notebook adalah proksi kasar,
  bukan pengganti perhitungan DFT adsorpsi/eksperimen.