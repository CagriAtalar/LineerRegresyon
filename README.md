# Lineer Regresyon Analiz Programi - Teknik Dokumantasyon

## Genel Bakis

Bu program, klasik/basit lineer regresyon analizini En Kucuk Kareler (OLS) yontemiyle gerceklestiren, sonuclari istatistiksel metriklerle birlikte gorsellestiren bir Python uygulamasidir.

**Temel Denklem:** Y = a*X + b
- a: Egim (slope)
- b: Kesisim noktasi (intercept)
- Y: Bagimli degisken
- X: Bagimsiz degisken

---

## Program Yapisi

### 3 Ana Sekme:
1. **OLS Analizi** - Sentetik veri ile En Kucuk Kareler yontemi
2. **Interaktif Alan** - Kullanici tarafindan eklenen noktalarla analiz
3. **CSV Analizi** - Harici veri dosyalarindan analiz

---

## 1. TEMEL ALGORITMA FONKSIYONLARI

### 1.1 calculate_linear_regression_ols(X_data, y_data)

**Konum:** Satir 63-90

**Matematiksel Temel:**
En Kucuk Kareler (Ordinary Least Squares) yontemi hata karelerinin toplamini minimize eder.

**Formulasyon:**
```
Hata Kareler Toplami: SSE = Σ(yi - (a*xi + b))²

Normal Denklemler:
β = (X^T * X)^(-1) * X^T * y

Burada:
- β = [b, a]^T (parametre vektoru)
- X = [1, x1, x2, ..., xn]^T (tasarim matrisi, sabit terim icin 1'ler eklenmis)
```

**Algoritma Adimlari:**
1. X verisine sabit terim sutunu ekle (intercept icin)
2. X^T * X matris carpimini hesapla
3. (X^T * X)^(-1) tersini al
4. Beta = (X^T * X)^(-1) * X^T * y islemini yap
5. Parametreleri dondur: b (intercept), a (slope)

**Kullanilan Kutuphaneler:**
- `statsmodels.api.OLS()` - OLS model sinifi
- `sm.add_constant()` - Sabit terim ekleme

**Cagrildigri Yerler:**
- `run_ols_analysis()` - Satir 360
- `calculate_interactive_regression()` - Satir 791
- `calculate_csv_regression()` - Satir 1346

---

### 1.2 calculate_performance_metrics(y_true, y_predicted)

**Konum:** Satir 93-129

**Hesaplanan Metrikler:**

#### R² (R-squared) - Belirlilik Katsayisi

**Formul:**
```
SSres = Σ(yi - ŷi)²  (Residual Sum of Squares)
SStot = Σ(yi - ȳ)²  (Total Sum of Squares)

R² = 1 - (SSres / SStot)
```

**Yorumlama:**
- R² = 1: Mukemmel fit (tum varyans aciklaniyor)
- R² = 0: Model hic aciklama gucune sahip degil
- Negatif R²: Model ortalamadan bile kotu

**Kod Adimlari:**
1. Residual kareler toplamini hesapla: ss_residual
2. Total kareler toplamini hesapla: ss_total
3. R² = 1 - (ss_residual / ss_total)

#### RMSE (Root Mean Square Error)

**Formul:**
```
MSE = (1/n) * Σ(yi - ŷi)²
RMSE = √(MSE)
```

**Yorumlama:**
- Y ile ayni birimde hata olcusu
- 0'a yakin = dusuk hata = iyi model

**Cagrildigri Yerler:**
- `run_ols_analysis()` - Satir 372
- `calculate_interactive_regression()` - Satir 807
- `calculate_csv_regression()` - Satir 1361

---

## 2. GUI FONKSIYONLARI ve VERI AKISI

### 2.1 TAB 1: OLS Analizi

#### create_tab1_ols_analysis(self)
**Konum:** Satir 251-335
- Grafik alanlari olusturur (scatter plot, residual histogram, residual scatter)
- Kontrol paneli ve metrik gosterim alanlari hazirlar

#### run_ols_analysis(self)
**Konum:** Satir 337-401

**Veri Akisi:**
```
1. Sentetik veri olustur: y = 2.5*X + 5 + gurultu
2. calculate_linear_regression_ols() cagir → a, b parametreleri
3. Tahminleri hesapla: y_pred = a*X + b
4. calculate_performance_metrics() → R², RMSE
5. Residuals hesapla: residuals = y - y_pred
6. Grafikleri guncelle
```

**Residual Analizi:**
- **Histogram:** Artiklarin dagilimini gosterir (ideal: normal dagilim)
- **Scatter:** Artiklar vs tahminler (ideal: rastgele dagilim, pattern yok)

#### show_statistics_window(self)
**Konum:** Satir 403-562

**Istatistiksel Testler:**

**Model ANOVA Testi:**
```
H0: Model anlamsiz (tum parametreler sifir)
H1: Model anlamli (en az bir parametre sifirdan farkli)

F-istatistigi = (SSreg / p) / (SSres / (n-p-1))
p = parametre sayisi
n = gozlem sayisi

P-value < 0.05 ise H0 reddedilir (model anlamli)
```

**Parametre t-Testi:**
```
H0: Parametre = 0 (anlamsiz)
H1: Parametre ≠ 0 (anlamli)

t-degeri = (β̂ - 0) / SE(β̂)
SE = Standart hata

P-value < 0.05 ise parametre anlamli
```

**Tablolarda Gosterilen Degerler:**
- R² (rsquared): Model aciklama gucu
- Adjusted R²: Parametre sayisina gore duzeltilmis R²
- F-statistic: Model anlamliligi testi
- Prob(F-statistic): ANOVA p-value
- Katsayi (params): Tahmin edilen parametreler
- Std. Error (bse): Katsayi standart hatalari
- t-degeri (tvalues): t-istatistigi
- P-value (pvalues): Parametre anlamliligi
- Guven Araligi: %95 guven araligi

---

### 2.2 TAB 2: Interaktif Alan

#### create_tab2_interactive(self)
**Konum:** Satir 564-665
- Click event dinleyicisi baglar
- Kullanicinin grafige noktalar eklemesini saglar

#### on_click_add_point(self, event)
**Konum:** Satir 667-678
- Mouse koordinatlarini yakalar
- X_inter ve y_inter listelerine ekler

#### calculate_interactive_regression(self)
**Konum:** Satir 680-730

**Veri Akisi:**
```
1. Kullanici noktalari topla (X_inter, y_inter)
2. NumPy array'e cevir
3. calculate_linear_regression_ols() cagir
4. Tahminleri hesapla
5. interactive_model olarak sakla
6. Grafigi guncelle
```

#### show_interactive_residuals(self)
**Konum:** Satir 805-889
- Artik histogrami ve scatter plot olusturur
- Scipy ile normal dagilim egrisi ekler
- Renkli goruntuler icin colormap kullanir

#### show_interactive_statistics(self)
**Konum:** Satir 891-1025
- interactive_model uzerinden istatistik tablosu olusturur
- P-value ve ANOVA sonuclarini gosterir

---

### 2.3 TAB 3: CSV Dosyasi Analizi

#### create_tab3_csv_analysis(self)
**Konum:** Satir 1069-1178
- Dosya secici dialog
- CSV okuma ve parse etme
- Grafik alanlari

#### load_csv_file(self)
**Konum:** Satir 1180-1241

**CSV Format Destegi:**

**2 Sutun (orn: data.csv):**
```
Sutun1,Sutun2
52,56.9
93,76.2
```
- 1. sutun → X degiskeni
- 2. sutun → Y degiskeni

**3 Sutun (orn: Salary_dataset.csv):**
```
PersonID,Sutun1,Sutun2
0,1.2,39344
1,1.4,46206
```
- 1. sutun → Atlanir (ID)
- 2. sutun → X degiskeni
- 3. sutun → Y degiskeni

**Sutun Isim Kullanimi:**
- Grafik eksenlerinde sutun isimleri gosterilir
- Denklem sutun isimleriyle yazilir
- Ornek: "Sicaklik = 0.821*CPU_Yuku + 13.456"

#### calculate_csv_regression(self)
**Konum:** Satir 1243-1278
- CSV verilerine OLS uygular
- csv_model olarak saklar
- Tum butonlari aktif eder

#### show_csv_statistics(self)
**Konum:** Satir 1280-1414
- CSV model istatistiklerini gosterir
- P-value, ANOVA analizi

#### show_csv_residuals(self)
**Konum:** Satir 1416-1500
- Residual histogram ve scatter plot
- CSV verileri icin hata analizi

---

## 3. MATEMATIKSEL ISLEMLER ve KONUMLARI

### 3.1 OLS Hesaplamalari

**Matris Islemi (Statsmodels icinde):**
```python
X_with_const = sm.add_constant(X)  # [1, X] matrisi
model = sm.OLS(y, X_with_const).fit()
```

**Hesaplanan:** β = (X^T*X)^(-1)*X^T*y

**Sonuclar:**
- model.params[0] → b (intercept)
- model.params[1] → a (slope)

**Konum:** calculate_linear_regression_ols() fonksiyonu, Satir 75-89

---

### 3.2 R² Hesaplama

**Konum:** calculate_performance_metrics(), Satir 110-123

**Adimlar:**
```python
# 1. Residual Sum of Squares
ss_residual = np.sum((y_true - y_predicted) ** 2)

# 2. Total Sum of Squares
y_mean = np.mean(y_true)
ss_total = np.sum((y_true - y_mean) ** 2)

# 3. R²
r_squared = 1 - (ss_residual / ss_total)
```

**Formul:** R² = 1 - (SSres/SStot)

---

### 3.3 RMSE Hesaplama

**Konum:** calculate_performance_metrics(), Satir 125-129

**Adimlar:**
```python
# 1. Mean Squared Error
mse = np.mean((y_true - y_predicted) ** 2)

# 2. Root MSE
rmse = np.sqrt(mse)
```

**Formul:** RMSE = √(MSE) = √((1/n)Σ(yi - ŷi)²)

---

## 4. ISTATISTIKSEL TESTLER

### 4.1 ANOVA F-Testi

**Konum:** model.fvalue, model.f_pvalue (Statsmodels sonucu)

**Hipotez:**
```
H0: β1 = 0 (model anlamsiz)
H1: β1 ≠ 0 (model anlamli)
```

**F-istatistigi:**
```
F = MSreg / MSres

MSreg = SSreg / dfr
MSres = SSres / dfe

dfr = regresyon serbestlik derecesi (1)
dfe = hata serbestlik derecesi (n-2)
```

**Karar Kurali:**
- P-value < 0.05 → Model anlamli
- P-value > 0.05 → Model anlamsiz

**Kullanim:** Tum istatistik tablolarinda gosterilir (Satir 466, 1002, 1318)

---

### 4.2 Parametre t-Testi

**Konum:** model.tvalues, model.pvalues

**Her parametre icin:**
```
t = (β̂ - 0) / SE(β̂)

SE(β̂) = standart hata (model.bse)
```

**Hipotez:**
```
H0: βi = 0 (parametre anlamsiz)
H1: βi ≠ 0 (parametre anlamli)
```

**Karar Kurali:**
- P-value < 0.05 → Parametre anlamli
- P-value > 0.05 → Parametre anlamsiz (tesadufi olabilir)

**Kullanim:** Parametre tablolarinda (Satir 499-502, 1035-1038, 1351-1354)

---

### 4.3 Guven Araligi

**Konum:** model.conf_int()

**Formul:**
```
CI = β̂ ± t(α/2, n-2) * SE(β̂)

t(α/2, n-2) = Student-t dagilimi kritik degeri
α = 0.05 (% 95 guven duzeyi icin)
```

**Yorumlama:**
Parametrenin gercek degerinin %95 olasilikla bu aralikta olduguna inaniyoruz.

---

## 5. RESIDUAL (ARTIK) ANALIZI

### 5.1 Residual Hesaplama

**Formuler Her Yerde Ayni:**
```python
residuals = y_true - y_predicted
```

**Konum:**
- Tab 1: Satir 374
- Interaktif: Satir 818 (show_interactive_residuals)
- CSV: Satir 1425 (show_csv_residuals)

---

### 5.2 Residual Diagnostics

**Ideal Ozellikler:**
1. Sifir ortalama: E[ε] = 0
2. Sabit varyans (homoskedasticity): Var(εi) = σ²
3. Normal dagilim: εi ~ N(0, σ²)
4. Bagimsizlik: Cov(εi, εj) = 0

**Kontrol Yontemleri:**
- **Histogram:** Normallik kontrolu
- **Scatter Plot:** Heteroskedasticity ve pattern kontrolu
- **Sifir cizgisi:** Sistematik hata kontrolu

---

## 6. ONEMLI NOTLAR

### Model Varsayimlari:
1. **Dogrusallik:** X ve Y arasi iliski dogrusal olmali
2. **Bagimsizlik:** Gozlemler birbirinden bagimsiz
3. **Homoskedasticity:** Hata varyansi sabit
4. **Normallik:** Hatalar normal dagilmali

### P-value Yorumlama:
- **< 0.001:** Cok guclu kanit
- **< 0.01:** Guclu kanit
- **< 0.05:** Yeterli kanit (genelde threshold)
- **> 0.05:** Yetersiz kanit (parametre anlamsiz olabilir)

### R² Yorumlama:
- **0.9-1.0:** Mukemmel fit
- **0.7-0.9:** Iyi fit
- **0.5-0.7:** Orta fit
- **< 0.5:** Zayif fit

---

## 7. KURULUM

### Gerekli Kutuphaneler:
```bash
pip install numpy pandas matplotlib statsmodels scikit-learn scipy
```


## 8. REFERANSLAR

### Matematiksel Temel:
- Ordinary Least Squares (OLS) yontemi
- ANOVA (Analysis of Variance)
- Hipotez testleri (t-test, F-test)
- Residual analizi

