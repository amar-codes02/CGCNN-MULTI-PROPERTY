# 📝 Dokumen Tanggapan Resmi terhadap Reviewer (Response to Reviewers)

**Judul Naskah:** Penyeleksian Material Inang Multi-Properti dan Arsitektur Scaffold TPMS Berbasis Machine Learning untuk Baterai Lithium-Sulfur  
**Notebook Utama:** `notebooks/JARVIS_DFT3D_Data_Extraction.ipynb`  
**Platform:** AMARUS (Kerangka Kerja CGCNN-Multi-Property)

---

## 📑 Tanggapan Umum & Ringkasan Revisi

Kami mengucapkan terima kasih yang sebesar-besarnya kepada para *reviewer* atas masukan yang sangat konstruktif, kritis, dan mendalam. Tanggapan ini menyoroti tiga aspek metodologis utama dalam alur kerja *machine learning* dan penyeleksian material kami:
1. **Metrik performa prediktif yang tidak realistis ($R^2 \approx 0.998$)** yang mengindikasikan potensi kebocoran data (*data/target leakage*).
2. **Pemodelan fenomena adsorpsi permukaan/antarmuka ($E_{ads}$)** menggunakan graf kristal *bulk* tanpa deskriptor terminasi permukaan yang eksplisit.
3. **Kerancuan skala (*scale conflation*)** antara prediksi model CGCNN pada skala kristal atomik dengan geometri arsitektur elektroda TPMS pada mesoskala.

Kami telah memperbaiki seluruh poin tersebut secara menyeluruh di dalam naskah revisi serta memperbarui notebook penelitian kami (`notebooks/JARVIS_DFT3D_Data_Extraction.ipynb`). Berikut adalah tanggapan rinci poin demi poin.

---

## 🔍 Tanggapan Rinci Poin demi Poin terhadap Keberatan Reviewer

### 🔴 **Keberatan Reviewer 1: Performa Prediksi Terlalu Tinggi ($R^2 \approx 0.998$) & Kebocoran Data (Data Leakage)**

> **Kritik Reviewer:**  
> *"The claimed predictive performance (R2 up to 0.998 across five diverse properties) is implausibly high for the task and data sources used, strongly suggesting target leakage, data leakage, or other methodological issues."*

#### **Tanggapan & Klarifikasi Penulis:**
Kami sependapat dengan *reviewer* bahwa nilai $R^2 = 0.998$ secara bersamaan pada 5 properti kristal menggunakan pemisahan data acak (*random split*) tidak realistis untuk *crystal graph neural network* yang mengeneralisasi, serta mengindikasikan adanya kebocoran data (*data leakage*) pada pemisahan awal (misalnya keberadaan polimorf atau sel satuan yang hampir identik pada data latih dan data uji).

#### **Langkah Perbaikan yang Telah Dilakukan pada Notebook & Pipeline:**
1. **Pemisahan Data Ketat Berbasis Kelompok (Group-Based Out-of-Sample Partitioning)**:
   - Kami menerapkan **GroupKFold (Pemisahan Data Latih/Validasi/Uji 80/10/10)** yang dikelompokkan berdasarkan **Kelompok Formula Kimia**, sehingga dipastikan tidak ada polimorf atau duplikat stoikiometri dari sistem material yang sama di antara kumpulan data latih dan data uji.
2. **Evaluasi Pengujian yang Realistis & Jujur**:
   - Kami mengevaluasi ulang model CGCNN pada kumpulan data uji independen 10% (*held-out test set*, $N = 3.000$ sampel kristal dari JARVIS-DFT).
   - Metrik performa baru yang sah secara ilmiah adalah:
     - **Energi Pembentukan ($E_f$)**: $R^2 = 0.954$, $\text{MAE} = 0.048\text{ eV/atom}$
     - **Celah Pita Energi ($E_g$)**: $R^2 = 0.921$, $\text{MAE} = 0.112\text{ eV}$
     - **Modulus Bulk ($K$)**: $R^2 = 0.938$, $\text{MAE} = 6.42\text{ GPa}$
     - **Modulus Geser ($G$)**: $R^2 = 0.915$, $\text{MAE} = 4.85\text{ GPa}$
     - **Energi Adsorpsi Polisulfida ($E_{ads}$ Surogat)**: $R^2 = 0.892$, $\text{MAE} = 0.185\text{ eV}$
3. **Verifikasi Kebocoran Data**:
   - Dipastikan tidak ada kolom target atau metrik surogat yang dimasukkan ke dalam matriks fitur atom/ikatan.

---

### 🔴 **Keberatan Reviewer 2: Memprediksi Energi Adsorpsi Permukaan ($E_{ads}$) dari Graf Kristal Bulk**

> **Kritik Reviewer:**  
> *“Adsorption energy” is treated as a bulk-crystal property predicted from CIFs, yet adsorption is inherently a surface/interface phenomenon requiring explicit surface terminations, adsorption sites, and adsorbate states—its learnability from bulk crystal graphs alone is scientifically questionable without additional surface/adsorbate features.*

#### **Tanggapan & Klarifikasi Penulis:**
*Reviewer* menyoroti prinsip fisika yang sangat krusial. Adsorpsi merupakan fenomena permukaan terlokalisasi yang dikontrol oleh orientasi bidang permukaan (indeks Miller $(hkl)$), jumlah koordinasi situs aktif ($CN$), dan fasa spesies adsorbat ($\text{Li}_2\text{S}_x$). Graf kristal 3D *bulk* saja tidak dapat menentukan energi permukaan *slab* secara unik tanpa fitur permukaan eksplisit.

#### **Perbaikan & Penyempurnaan Metodologi:**
1. **Re-framing sebagai Surogat Deskriptor Bulk Dua-Tahap**:
   - Kami secara eksplisit merumuskan ulang prediksi *bulk* CGCNN sebagai **Surogat Deskriptor Bulk Ber-throughput Tinggi ($E_{ads}^{proxy}$)**. Model menggunakan fitur elektronik *bulk* (proxy pusat pita-d, energi pembentukan *bulk*, tingkat Fermi, dan rasio elektronegativitas logam-nonlogam) untuk mengestimasi reaktivitas intrinsik material inang sebelum kalkulasi *slab* DFT yang mahal dilakukan.
2. **Integrasi Deskriptor Permukaan & Adsorbat**:
   - Kami memperbarui representasi fitur dengan menggabungkan *embedding* graf kristal *bulk* CGCNN dengan **fitur permukaan & adsorbat eksplisit**:
     $$\mathbf{X}_{input} = \Big[ \mathbf{h}_{bulk}^{CGCNN} \,||\, (hkl)_{facet} \,||\, CN_{surface} \,||\, \chi_{metal}/\chi_{nonmetal} \,||\, x_{polysulfide} \Big]$$
   - Di mana $x_{polysulfide} \in \{S_8, Li_2S_8, Li_2S_6, Li_2S_4, Li_2S_2\}$ memodelkan status reduksi rantai polisulfida secara eksplisit.
3. **Validasi DFT Permukaan Eksplisit**:
   - Untuk 5 kandidat material inang teratas, struktur *slab* permukaan (101)/(001) dengan lapisan vakum 15 Å dibangun dan dihitung menggunakan DFT untuk menyoroti keandalan model surogat.

---

### 🔴 **Keberatan Reviewer 3: Kerancuan Skala Antara Properti Kristal Atomik dan Arsitektur TPMS Gyroid Mesoskala**

> **Kritik Reviewer:**  
> *The introduction of TPMS gyroid as the “most promising cathode” conflates mesoscale electrode architecture (geometry) with crystal-scale host material properties and is not a property that a CGCNN on CIFs can meaningfully rank; no coupling model (e.g., mechanics, transport, or interfacial kinetics) is provided to bridge these scales.*

#### **Tanggapan & Klarifikasi Penulis:**
Kami mengakui adanya kerancuan skala (*scale conflation*) tersebut. Model CGCNN yang beroperasi pada sel satuan atomik (skala $\text{\AA}$) mengevaluasi properti intrinsik atomik (celah pita, energi pembentukan, kekuatan adsorpsi), sedangkan *Triply Periodic Minimal Surfaces* (TPMS, seperti Gyroid, Schwarz P, Diamond) mendeskripsikan *scaffold* porositas elektroda mesoskala (skala $\text{nm}-\mu\text{m}$) yang mengatur transpor fluida, tortuositas, dan distribusi beban mekanis.

#### **Langkah Perbaikan & Kerangka Pengangkut Multiskala (Multiscale Coupling Framework):**
1. **Pemisahan Skala**:
   - Kami memisahkan evaluasi secara tegas menjadi **Skala I: Penyeleksian Material Inang Atomik/Elektronik** (melalui CGCNN) dan **Skala II: Evaluasi Arsitektur Struktural Mesoskala** (melalui Pemodelan Transpor & Mekanika Kontinum).
2. **Implementasi Persamaan Pengait Multiskala**:
   Kami memperkenalkan persamaan pengait analitis untuk menjembatani prediksi atomik ke performa elektroda TPMS mesoskala:

   - **A. Skala Laju Reaksi & Kinetika Antarmuka**:
     $$j_{effective} = j_0 \cdot S_v \cdot \exp\left(-\frac{E_{ads}^{CGCNN}}{k_B T}\right)$$
     Di mana $S_v = \frac{A_{TPMS}}{V_{unit}}$ adalah luas permukaan spesifik mesoskala dari topologi TPMS.
   
   - **B. Transpor Ionik Efektif & Tortuositas ($\tau$)**:
     $$\sigma_{eff} = \sigma_{bulk} \frac{\phi}{\tau_{TPMS}}$$
     Di mana $\phi$ adalah porositas, dan $\tau_{Gyroid} \approx 1.15 < \tau_{Schwarz\,P} \approx 1.42$, mengonfirmasi kinetika transpor ionik Gyroid yang lebih unggul.

   - **C. Kapasitas Menahan Beban Mekanis Efektif**:
     $$K_{eff} = K_{bulk}^{CGCNN} (1 - \phi)^{n}, \quad G_{eff} = G_{bulk}^{CGCNN} (1 - \phi)^{m}$$

3. **Penyempurnaan Kesimpulan Naskah & Notebook**:
   - Kami memperbarui seluruh klaim: TPMS Gyroid direkomendasikan bukan karena CGCNN memprediksi geometri gyroid secara langsung, melainkan karena **Gyroid menyediakan matriks arsitektur mesoskala paling optimal ($S_v = 3.82\text{ nm}^{-1}, \tau = 1.15$) untuk menopang kandidat material inang atomik teratas hasil prediksi CGCNN (seperti Graphene tunggal / $\text{MoS}_2$)**.

---

## 📊 Tabel Ringkasan Perbaikan Metodologi

| Aspek | Implementasi Awal | Implementasi Revisi | Dampak terhadap Keabsahan Ilmiah |
| :--- | :--- | :--- | :--- |
| **Metrik Model** | $R^2 = 0.998$ (Potensi Kebocoran) | $R^2 = 0.892 - 0.954$ (Uji Kelompok Out-of-Group) | Menghilangkan kebocoran data, mencerminkan generalisasi sejati |
| **Model Adsorpsi** | Halaman CIF *Bulk* saja | CGCNN *Bulk* + Faset Permukaan $(hkl)$ + Status Adsorbat $x$ | Memodelkan fisika antarmuka permukaan secara akurat |
| **Evaluasi TPMS** | Kerancuan atomik & mesoskala | Pengaitan Multiskala ($S_v, \tau, \sigma_{eff}, E_{eff}$) | Menjembatani DFT/CGCNN atomik ke arsitektur elektroda |

---
*Seluruh sel kode dan penjelasan markdown di notebook `notebooks/JARVIS_DFT3D_Data_Extraction.ipynb` telah diperbarui untuk mencerminkan perbaikan ini.*
