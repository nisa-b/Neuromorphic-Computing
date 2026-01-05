# Neuromorphic-Computing
# Evet-Hayır Ses Sınıflandırma: ANN vs SNN Analizi

Bu proje, Türkçe "Evet" ve "Hayır" komutlarını ayırt etmek için geliştirilmiş uçtan uca bir ses işleme ve makine öğrenmesi hattıdır. Geleneksel Yapay Sinir Ağları (ANN) ile nöromorfik Sıçramalı Sinir Ağları (SNN) arasındaki performans ve verimlilik farklarını analiz etmeyi amaçlar.

## 🚀 Proje Aşamaları

### 1. Veri Önişleme (`01_preprocess_and_split.py`)
Ham ses verileri standartlaştırılır:
* **Filtreleme:** 300-3500 Hz aralığında Butterworth bandpass filtresi uygulanır.
* **Trim & Crop:** Sessiz kısımlar atılır ve RMS enerjisi kullanılarak en yoğun 1 saniyelik kesit alınır.
* **Bölme:** Veri seti %80 eğitim, %20 test olacak şekilde sınıflara göre dengeli (stratified) bölünür.

### 2. Özellik Çıkarımı (`02_feature_extraction.py`)
Üç farklı temsil yöntemi kullanılarak özellikler çıkarılır:
* **Zaman Serisi (T):** Doğrudan genlik değerleri.
* **Frekans Analizi (Fourier):** FFT tabanlı log-magnitude spektrumu.
* **Dalgacık Analizi (Wavelet):** `db4` dalgacığı ile 5. seviye ayrıştırma üzerinden istatistiksel özellikler (ortalama, enerji, entropi vb.).

### 3. Model Eğitimi (`03_train.py`)
* **ANN:** Çok katmanlı algılayıcı (MLP) mimarisi.
* **SNN:** `snntorch` kullanılarak oluşturulan, sızıntılı entegre et ve ateşle (LIF) nöron modeli.
* **Kodlama:** Girdi verileri Poisson kodlama yöntemiyle sıçrama (spike) dizilerine dönüştürülür.

### 4. Sonuç Analizi (`04_results_table_visualization.py`)
Modellerin doğruluğu, karmaşıklık matrisleri ve SNN'lerin ortalama sıçrama sayıları (enerji verimliliği) görselleştirilir.

## 🛠️ Kurulum

1. Depoyu klonlayın.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
