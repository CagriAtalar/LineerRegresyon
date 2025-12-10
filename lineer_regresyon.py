"""
==============================================
LİNEER REGRESYON ANALİZ PROGRAMI
==============================================

Bu program klasik/basit lineer regresyon analizini içerir:
1. En Küçük Kareler Yöntemi (Ordinary Least Squares - OLS)
2. İstatistiksel Analiz (P-value, ANOVA, R², RMSE)
3. Hata Analizi (Residuals)

Matematiksel Temel:
Lineer Regresyon Denklemi: Y = a*X + b
- a: Eğim (slope)
- b: Kesişim noktası (intercept)
- Y: Bağımlı değişken
- X: Bağımsız değişken
"""

import tkinter as tk
from tkinter import ttk, Toplevel, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Matplotlib tema ayarları - Sade ve net görünüm
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.family'] = 'sans-serif'


# ============================================================================
# ALGORİTMA 1: EN KÜÇÜK KARELER YÖNTEMİ (OLS)
# ============================================================================
def calculate_linear_regression_ols(X_data, y_data):
    """
    En Küçük Kareler Yöntemi ile Lineer Regresyon
    
    Algoritma:
    1. Veriyi al (X ve y)
    2. X'e sabit terim ekle (intercept için)
    3. OLS formülü uygula: β = (X'X)⁻¹X'y
    4. Parametreleri döndür
    
    Parametreler:
        X_data: Bağımsız değişken verileri
        y_data: Bağımlı değişken verileri
    
    Döndürür:
        model: Statsmodels OLS model objesi
        a (eğim), b (kesişim)
    """
    # Adım 1: Sabit terim ekleme (intercept için)
    X_with_const = sm.add_constant(X_data)
    
    # Adım 2: OLS modeli oluştur ve fit et
    # Bu adımda matematiksel olarak: β = (X'X)⁻¹X'y hesaplanır
    model = sm.OLS(y_data, X_with_const).fit()
    
    # Adım 3: Parametreleri çıkar
    b_intercept = model.params[0]  # Kesişim noktası (b)
    a_slope = model.params[1]      # Eğim (a)
    
    return model, a_slope, b_intercept


# ============================================================================
# ALGORİTMA 2: PERFORMANS METRİKLERİ
# ============================================================================
def calculate_performance_metrics(y_true, y_predicted):
    """
    Model Performans Metriklerini Hesapla
    
    1. R² (R-squared) - Belirlilik Katsayısı
       Formül: R² = 1 - (SSres / SStot)
       SSres = Σ(y_true - y_pred)²
       SStot = Σ(y_true - y_mean)²
       
    2. RMSE (Root Mean Square Error)
       Formül: RMSE = √(Σ(y_true - y_pred)² / n)
    
    Parametreler:
        y_true: Gerçek değerler
        y_predicted: Tahmin edilen değerler
    
    Döndürür:
        r2_score: R² değeri (0-1 arası, 1'e yakın = iyi)
        rmse: RMSE değeri (0'a yakın = iyi)
    """
    # R² Hesaplama
    # 1. Residual Sum of Squares (Artıkların kareler toplamı)
    ss_residual = np.sum((y_true - y_predicted) ** 2)
    
    # 2. Total Sum of Squares (Toplam kareler)
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    
    # 3. R² = 1 - (SSres / SStot)
    r_squared = 1 - (ss_residual / ss_total)
    
    # RMSE Hesaplama
    # 1. Mean Squared Error
    mse = np.mean((y_true - y_predicted) ** 2)
    
    # 2. Root MSE
    rmse = np.sqrt(mse)
    
    return r_squared, rmse




# ============================================================================
# ANA UYGULAMA SINIFI
# ============================================================================
class LinearRegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lineer Regresyon Analizi")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Notebook (Tab sistemi) oluştur
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 3 sekme oluştur
        self.create_tab1_ols_analysis()      # Tab 1: OLS Analizi
        self.create_tab2_interactive()       # Tab 2: İnteraktif Alan
        self.create_tab3_csv_analysis()      # Tab 3: CSV Analizi
        
    # ========================================================================
    # TAB 1: EN KÜÇÜK KARELER YÖNTEMİ (LS) ANALİZİ
    # ========================================================================
    def create_tab1_ols_analysis(self):
        """
        Sekme 1: OLS (Ordinary Least Squares) Analizi
        
        Bu sekmede:
        - En Küçük Kareler yöntemi ile regresyon
        - Model performans metrikleri (R², RMSE)
        - Hata analizi (Residuals)
        - İstatistiksel testler (P-value, ANOVA)
        """
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text=" LS Analizi")
        
        # Üst Kontrol Paneli
        control_frame = tk.Frame(tab1, bg='white', relief='raised', bd=2)
        control_frame.pack(side='top', fill='x', padx=10, pady=10)
        
        tk.Label(control_frame, text="Veri Sayısı (N):", 
                font=('Arial', 12, 'bold'), bg='white').pack(side='left', padx=10)
        
        self.n_slider = tk.Scale(control_frame, from_=20, to=200, orient='horizontal', 
                                 length=300, font=('Arial', 10))
        self.n_slider.set(50)
        self.n_slider.pack(side='left', padx=10)
        
        analyze_btn = tk.Button(control_frame, text="▶ ANALİZ BAŞLAT (OLS)", 
                               command=self.run_ols_analysis,
                               font=('Arial', 12, 'bold'), bg='#4CAF50', fg='white',
                               padx=20, pady=10, cursor='hand2')
        analyze_btn.pack(side='left', padx=20)
        
        # İstatistik tablosu butonu
        self.stats_btn = tk.Button(control_frame, text="📊 P-value & ANOVA Tablosu", 
                                   command=self.show_statistics_window,
                                   font=('Arial', 12, 'bold'), bg='white', fg='black',
                                   padx=20, pady=10, cursor='hand2', relief='solid', bd=2,
                                   state='disabled')
        self.stats_btn.pack(side='left', padx=20)
        
        # Model sakla
        self.current_model = None
        
        # Ana içerik alanı
        content_frame = tk.Frame(tab1, bg='#f0f0f0')
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # ÜST KISIM: Scatter plot ve metrikler
        top_frame = tk.Frame(content_frame, bg='#f0f0f0')
        top_frame.pack(fill='both', expand=True)
        
        # Sol: Ana scatter plot
        self.fig1 = Figure(figsize=(7, 5), facecolor='white')
        self.ax1_main = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, top_frame)
        self.canvas1.get_tk_widget().pack(side='left', fill='both', expand=True, padx=5)
        
        # Sağ: Metrik paneli
        metrics_frame = tk.Frame(top_frame, bg='white', relief='solid', bd=2, width=300)
        metrics_frame.pack(side='left', fill='y', padx=5, pady=5)
        metrics_frame.pack_propagate(False)
        
        tk.Label(metrics_frame, text="MODEL PARAMETRELERİ", 
                font=('Arial', 14, 'bold'), bg='white', fg='#2196F3').pack(pady=15)
        
        self.equation_label = tk.Label(metrics_frame, text="Y = a*X + b", 
                                      font=('Arial', 13, 'bold'), bg='white', fg='#d32f2f')
        self.equation_label.pack(pady=10, padx=10)
        
        tk.Frame(metrics_frame, height=2, bg='#e0e0e0').pack(fill='x', padx=20, pady=10)
        
        tk.Label(metrics_frame, text="PERFORMANS METRİKLERİ", 
                font=('Arial', 12, 'bold'), bg='white', fg='#666').pack(pady=5)
        
        self.r2_label = tk.Label(metrics_frame, text="R² = -", 
                                font=('Arial', 12), bg='white')
        self.r2_label.pack(pady=5)
        
        self.rmse_label = tk.Label(metrics_frame, text="RMSE = -", 
                                   font=('Arial', 12), bg='white')
        self.rmse_label.pack(pady=5)
        
        # ALT KISIM: Hata analizi grafikleri (Tam genişlik)
        bottom_frame = tk.Frame(content_frame, bg='#f0f0f0')
        bottom_frame.pack(fill='both', expand=True)
        
        # Hata analizi grafikleri (2 grafik yan yana, daha geniş)
        self.fig2 = Figure(figsize=(12, 5), facecolor='white')
        self.ax2_hist = self.fig2.add_subplot(121)
        self.ax2_resid = self.fig2.add_subplot(122)
        self.fig2.tight_layout(pad=3.0)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, bottom_frame)
        self.canvas2.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
    
    def run_ols_analysis(self):
        """
        OLS Analizi Çalıştır
        
        ADIMLAR:
        1. Veri oluştur
        2. OLS algoritması ile regresyon yap
        3. Tahminleri hesapla
        4. Performans metriklerini hesapla
        5. Sonuçları görselleştir
        """
        n = self.n_slider.get()
        
        # ADIM 1: Veri Oluşturma
        # y = 2.5*X + 5 + gürültü (gerçek ilişki)
        np.random.seed(42)
        X = np.linspace(0, 10, n)
        y_true = 2.5 * X + 5
        noise = np.random.normal(0, 3, n)
        y = y_true + noise
        
        # ADIM 2: OLS Algoritması
        model, a_slope, b_intercept = calculate_linear_regression_ols(X, y)
        
        # Modeli sakla (istatistik tablosu için)
        self.current_model = model
        self.stats_btn.config(state='normal')
        
        # ADIM 3: Tahminler
        X_with_const = sm.add_constant(X)
        y_pred = model.predict(X_with_const)
        
        # ADIM 4: Performans Metrikleri
        r2, rmse = calculate_performance_metrics(y, y_pred)
        
        # Artıklar (Residuals)
        residuals = y - y_pred
        
        # ADIM 5: Görselleştirme
        
        # 5a. Ana scatter plot
        self.ax1_main.clear()
        self.ax1_main.scatter(X, y, alpha=0.6, s=50, label='Gerçek Veri', color='blue')
        self.ax1_main.plot(X, y_pred, 'r-', linewidth=3, label='OLS Regresyon Doğrusu')
        self.ax1_main.set_xlabel('X (Bağımsız Değişken)', fontsize=12, fontweight='bold')
        self.ax1_main.set_ylabel('Y (Bağımlı Değişken)', fontsize=12, fontweight='bold')
        self.ax1_main.set_title('En Küçük Kareler Yöntemi (OLS)', fontsize=14, fontweight='bold')
        self.ax1_main.legend(fontsize=10)
        self.ax1_main.grid(True, alpha=0.3)
        self.canvas1.draw()
        
        # 5b. Metrikleri güncelle
        self.equation_label.config(text=f"Y = {a_slope:.3f}*X + {b_intercept:.3f}")
        self.r2_label.config(text=f"R² = {r2:.4f} (Açıklama Gücü)")
        self.rmse_label.config(text=f"RMSE = {rmse:.4f} (Ortalama Hata)")
        
        # 5c. Hata analizi grafikleri
        # Histogram: Artıkların dağılımı
        self.ax2_hist.clear()
        self.ax2_hist.hist(residuals, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        self.ax2_hist.set_xlabel('Artıklar (y_gerçek - y_tahmin)', fontsize=11, fontweight='bold')
        self.ax2_hist.set_ylabel('Frekans', fontsize=11, fontweight='bold')
        self.ax2_hist.set_title(' Residuals Histogram (Artıkların Dağılımı)', 
                               fontsize=13, fontweight='bold', pad=10)
        self.ax2_hist.axvline(0, color='red', linestyle='--', linewidth=2, label='Sıfır çizgisi (İdeal)')
        self.ax2_hist.legend(fontsize=10)
        self.ax2_hist.grid(True, alpha=0.3)
        
        # Scatter: Artıklar vs Tahminler
        self.ax2_resid.clear()
        self.ax2_resid.scatter(y_pred, residuals, alpha=0.7, s=60, color='coral', edgecolors='black', linewidth=0.5)
        self.ax2_resid.axhline(0, color='red', linestyle='--', linewidth=2, label='İdeal (hata=0)')
        self.ax2_resid.set_xlabel('Tahmin Edilen Değerler', fontsize=11, fontweight='bold')
        self.ax2_resid.set_ylabel('Artıklar', fontsize=11, fontweight='bold')
        self.ax2_resid.set_title(' Residuals Scatter (Artıklar vs. Tahmin)', 
                                fontsize=13, fontweight='bold', pad=10)
        self.ax2_resid.legend(fontsize=10)
        self.ax2_resid.grid(True, alpha=0.3)
        
        # Alt bilgilendirme
        self.fig2.text(0.5, 0.01, 
                      '💡 İdeal durumda: Artıklar normal dağılım göstermeli ve sıfır etrafında rastgele dağılmalıdır',
                      ha='center', fontsize=10, style='italic', weight='bold')
        
        self.fig2.tight_layout(pad=2.5, rect=[0, 0.03, 1, 1])
        self.canvas2.draw()
        
        # Not: İstatistiksel detaylar için "P-value & ANOVA Tablosu" butonuna tıklayın
    
    def show_statistics_window(self):
        """P-value ve ANOVA tablosunu ayrı pencerede göster"""
        if self.current_model is None:
            return
        
        stats_window = Toplevel(self.root)
        stats_window.title(" İSTATİSTİK TABLOSU - P-value & ANOVA")
        stats_window.geometry("1200x800")
        stats_window.configure(bg='white')
        
        title_label = tk.Label(stats_window, text="DETAYLI İSTATİSTİK ANALİZİ", 
                               font=('Arial', 18, 'bold'), bg='white', fg='black', pady=20)
        title_label.pack()
        
        fig = Figure(figsize=(12, 10), facecolor='white', dpi=100)
        canvas = FigureCanvasTkAgg(fig, stats_window)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
        
        #fig.suptitle('İSTATİSTİK TABLOSU - P-VALUES & ANOVA', 
        #            fontsize=20, fontweight='bold', y=0.98)
        
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 1], hspace=0.5,
                             top=0.93, bottom=0.05, left=0.1, right=0.9)
        
        ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        
        model = self.current_model
        
        # TABLO 1: Model İstatistikleri
        ax1.text(0.5, 1.15, 'MODEL İSTATİSTİKLERİ ve ANOVA', 
                ha='center', va='top', fontsize=16, fontweight='bold', transform=ax1.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', edgecolor='black', linewidth=2))
        
        table1_data = [
            ['Metrik', 'Değer', 'Açıklama'],
            ['R² (R-squared)', f'{model.rsquared:.6f}', 'Modelin açıklama gücü'],
            ['Düzeltilmiş R²', f'{model.rsquared_adj:.6f}', 'Düzeltilmiş açıklama gücü'],
            ['F-İstatistiği (ANOVA)', f'{model.fvalue:.4f}', 'Model anlamlılık testi'],
            ['Prob(F-statistic)', f'{model.f_pvalue:.8f}', 'ANOVA P-VALUE'],
        ]
        
        table1 = ax1.table(cellText=table1_data, cellLoc='left',
                          bbox=[0.05, 0.05, 0.9, 0.80], colWidths=[0.35, 0.25, 0.4])
        table1.auto_set_font_size(False)
        table1.set_fontsize(13)
        
        for i in range(3):
            table1[(0, i)].set_facecolor('black')
            table1[(0, i)].set_text_props(weight='bold', color='white', fontsize=14)
            table1[(0, i)].set_edgecolor('black')
            table1[(0, i)].set_linewidth(2)
        
        for i in range(1, len(table1_data)):
            for j in range(3):
                cell = table1[(i, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
                if i == 4:
                    cell.set_facecolor('#ffeb3b')
                    if j == 2:
                        cell.set_text_props(weight='bold', fontsize=13)
        
        # TABLO 2: Parametre P-VALUES
        ax2.text(0.5, 1.12, 'PARAMETRE TAHMİNLERİ ve P-VALUES', 
                ha='center', va='top', fontsize=16, fontweight='bold', transform=ax2.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', edgecolor='black', linewidth=2))
        
        table2_data = [
            ['Parametre', 'Katsayı', 'Std. Hata', 't-değeri', 'P-value', 'Güven Aralığı (95%)'],
            ['Sabit Terim (b)', f'{model.params[0]:.6f}', f'{model.bse[0]:.6f}', 
             f'{model.tvalues[0]:.4f}', f'{model.pvalues[0]:.8f}',
             f'[{model.conf_int()[0][0]:.4f}, {model.conf_int()[0][1]:.4f}]'],
            ['Eğim (a)', f'{model.params[1]:.6f}', f'{model.bse[1]:.6f}', 
             f'{model.tvalues[1]:.4f}', f'{model.pvalues[1]:.8f}',
             f'[{model.conf_int()[1][0]:.4f}, {model.conf_int()[1][1]:.4f}]']
        ]
        
        table2 = ax2.table(cellText=table2_data, cellLoc='center',
                          bbox=[0.02, 0.02, 0.96, 0.83], colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2])
        table2.auto_set_font_size(False)
        table2.set_fontsize(12)
        
        for i in range(6):
            table2[(0, i)].set_facecolor('black')
            table2[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)
            table2[(0, i)].set_edgecolor('black')
            table2[(0, i)].set_linewidth(2)
        
        for i in range(1, len(table2_data)):
            for j in range(6):
                cell = table2[(i, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
                if j == 4:
                    cell.set_facecolor('#ffeb3b')
                    cell.set_text_props(weight='bold', fontsize=12)
        
        # TABLO 3: Ek İstatistikler
        ax3.text(0.5, 1.15, 'EK İSTATİSTİKLER', 
                ha='center', va='top', fontsize=16, fontweight='bold', transform=ax3.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', edgecolor='black', linewidth=2))
        
        table3_data = [
            ['İstatistik', 'Değer', 'Açıklama'],
            ['Log-Likelihood', f'{model.llf:.4f}', 'Logaritmik olabilirlik'],
            ['AIC (Akaike)', f'{model.aic:.4f}', 'Akaike Bilgi Kriteri'],
            ['BIC (Bayesian)', f'{model.bic:.4f}', 'Bayesian Bilgi Kriteri'],
            ['Gözlem Sayısı', f'{int(model.nobs)}', 'Toplam veri noktası sayısı'],
        ]
        
        table3 = ax3.table(cellText=table3_data, cellLoc='left',
                          bbox=[0.05, 0.05, 0.9, 0.80], colWidths=[0.35, 0.25, 0.4])
        table3.auto_set_font_size(False)
        table3.set_fontsize(13)
        
        for i in range(3):
            table3[(0, i)].set_facecolor('black')
            table3[(0, i)].set_text_props(weight='bold', color='white', fontsize=14)
            table3[(0, i)].set_edgecolor('black')
            table3[(0, i)].set_linewidth(2)
        
        for i in range(1, len(table3_data)):
            for j in range(3):
                cell = table3[(i, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
        
        fig.text(0.5, 0.02, " P-value < 0.05 ise parametre istatistiksel olarak anlamlıdır",
                ha='center', fontsize=12, style='italic', weight='bold')
        
        canvas.draw()
    
    # ========================================================================
    # TAB 2: İNTERAKTİF ALAN
    # ========================================================================
    def create_tab2_interactive(self):
        """
        Sekme 2: İnteraktif Oyun Alanı
        
        Kullanıcı kendi noktalarını ekleyip regresyon görebilir
        """
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="🎯 İnteraktif Alan")
        
        main_frame = tk.Frame(tab2, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Sol: Grafik alanı
        canvas_frame = tk.Frame(main_frame, bg='white', relief='solid', bd=2)
        canvas_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.fig_inter = Figure(figsize=(8, 6), facecolor='white')
        self.ax_inter = self.fig_inter.add_subplot(111)
        self.ax_inter.set_xlim(0, 10)
        self.ax_inter.set_ylim(0, 10)
        self.ax_inter.set_xlabel('X', fontsize=12, fontweight='bold')
        self.ax_inter.set_ylabel('Y', fontsize=12, fontweight='bold')
        self.ax_inter.set_title(' Noktaları Eklemek İçin Tıklayın', fontsize=13, fontweight='bold')
        self.ax_inter.grid(True, alpha=0.3)
        
        self.canvas_inter = FigureCanvasTkAgg(self.fig_inter, canvas_frame)
        self.canvas_inter.get_tk_widget().pack(fill='both', expand=True)
        
        # Veri listeleri
        self.X_inter = []
        self.y_inter = []
        
        # Tıklama eventi
        self.canvas_inter.mpl_connect('button_press_event', self.on_click_add_point)
        
        # Sağ: Kontrol paneli
        control_frame = tk.Frame(main_frame, bg='white', relief='solid', bd=2, width=300)
        control_frame.pack(side='left', fill='y', padx=5)
        control_frame.pack_propagate(False)
        
        tk.Label(control_frame, text="🎮 KONTROL PANELİ", 
                font=('Arial', 14, 'bold'), bg='white', fg='#2196F3').pack(pady=20)
        
        tk.Button(control_frame, text=" HESAPLA (LS)", command=self.calculate_interactive_regression,
                 font=('Arial', 13, 'bold'), bg='#4CAF50', fg='white',
                 padx=30, pady=15, cursor='hand2').pack(pady=10)
        
        self.inter_stats_btn = tk.Button(control_frame, text=" İstatistik Tablosu", 
                                         command=self.show_interactive_statistics,
                                         font=('Arial', 12, 'bold'), bg='white', fg='black',
                                         padx=20, pady=15, cursor='hand2', relief='solid', bd=2,
                                         state='disabled')
        self.inter_stats_btn.pack(pady=10)
        
        self.inter_error_btn = tk.Button(control_frame, text="📉 Hata Analizi", 
                                        command=self.show_interactive_residuals,
                                        font=('Arial', 12, 'bold'), bg='white', fg='black',
                                        padx=20, pady=15, cursor='hand2', relief='solid', bd=2,
                                        state='disabled')
        self.inter_error_btn.pack(pady=10)
        
        tk.Button(control_frame, text=" Temizle", command=self.clear_interactive_points,
                 font=('Arial', 13, 'bold'), bg='#f44336', fg='white',
                 padx=30, pady=15, cursor='hand2').pack(pady=10)
        
        # Model ve veri sakla
        self.interactive_model = None
        self.interactive_X = None
        self.interactive_y = None
        self.interactive_y_pred = None
        
        tk.Frame(control_frame, height=2, bg='#e0e0e0').pack(fill='x', padx=20, pady=20)
        
        tk.Label(control_frame, text=" SONUÇLAR", 
                font=('Arial', 13, 'bold'), bg='white', fg='#666').pack(pady=10)
        
        self.inter_equation = tk.Label(control_frame, text="Denklem: -", 
                                      font=('Arial', 11), bg='white', fg='#d32f2f',
                                      wraplength=250)
        self.inter_equation.pack(pady=10, padx=10)
        
        self.inter_r2 = tk.Label(control_frame, text="R² = -", font=('Arial', 11), bg='white')
        self.inter_r2.pack(pady=5)
        
        self.inter_points = tk.Label(control_frame, text="Nokta Sayısı: 0", font=('Arial', 11), bg='white')
        self.inter_points.pack(pady=5)
    
    def on_click_add_point(self, event):
        """Grafiğe nokta ekle"""
        if event.inaxes == self.ax_inter and event.button == 1:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.X_inter.append(x)
                self.y_inter.append(y)
                
                self.ax_inter.scatter(x, y, c='blue', s=80, alpha=0.7, edgecolors='black')
                self.canvas_inter.draw()
                
                self.inter_points.config(text=f"Nokta Sayısı: {len(self.X_inter)}")
    
    def calculate_interactive_regression(self):
        """İnteraktif noktalara OLS uygula"""
        if len(self.X_inter) < 2:
            self.inter_equation.config(text="En az 2 nokta gerekli!")
            return
        
        X = np.array(self.X_inter)
        y = np.array(self.y_inter)
        
        # OLS Algoritması
        model, a_slope, b_intercept = calculate_linear_regression_ols(X, y)
        
        # Tahmin
        X_with_const = sm.add_constant(X)
        y_pred = model.predict(X_with_const)
        
        # Sakla
        self.interactive_model = model
        self.interactive_X = X
        self.interactive_y = y
        self.interactive_y_pred = y_pred
        self.inter_stats_btn.config(state='normal')
        self.inter_error_btn.config(state='normal')
        
        # Metrikler
        r2, _ = calculate_performance_metrics(y, y_pred)
        
        # Çiz
        self.ax_inter.clear()
        self.ax_inter.set_xlim(0, 10)
        self.ax_inter.set_ylim(0, 10)
        self.ax_inter.set_xlabel('X', fontsize=12, fontweight='bold')
        self.ax_inter.set_ylabel('Y', fontsize=12, fontweight='bold')
        self.ax_inter.set_title('Regresyon Sonucu', fontsize=13, fontweight='bold')
        self.ax_inter.grid(True, alpha=0.3)
        
        self.ax_inter.scatter(X, y, c='blue', s=80, alpha=0.7, edgecolors='black', label='Veri')
        
        X_line = np.linspace(0, 10, 100)
        y_line = b_intercept + a_slope * X_line
        self.ax_inter.plot(X_line, y_line, 'r-', linewidth=3, label='OLS Regresyon')
        
        self.ax_inter.legend(fontsize=10)
        self.canvas_inter.draw()
        
        # Sonuçları göster
        self.inter_equation.config(text=f"Y = {a_slope:.3f}*X + {b_intercept:.3f}")
        self.inter_r2.config(text=f"R² = {r2:.4f}")
    
    def clear_interactive_points(self):
        """Tüm noktaları temizle"""
        self.X_inter = []
        self.y_inter = []
        
        self.ax_inter.clear()
        self.ax_inter.set_xlim(0, 10)
        self.ax_inter.set_ylim(0, 10)
        self.ax_inter.set_xlabel('X', fontsize=12, fontweight='bold')
        self.ax_inter.set_ylabel('Y', fontsize=12, fontweight='bold')
        self.ax_inter.set_title(' Noktaları Eklemek İçin Tıklayın', fontsize=13, fontweight='bold')
        self.ax_inter.grid(True, alpha=0.3)
        self.canvas_inter.draw()
        
        self.inter_equation.config(text="Denklem: -")
        self.inter_r2.config(text="R² = -")
        self.inter_points.config(text="Nokta Sayısı: 0")
        self.inter_stats_btn.config(state='disabled')
        self.inter_error_btn.config(state='disabled')
        self.interactive_model = None
        self.interactive_X = None
        self.interactive_y = None
        self.interactive_y_pred = None
    
    def show_interactive_residuals(self):
        """İnteraktif veriler için hata analizi göster"""
        if self.interactive_model is None or self.interactive_y is None:
            return
        
        residuals_window = Toplevel(self.root)
        residuals_window.geometry("1400x700")
        residuals_window.configure(bg='white')
        
        tk.Label(residuals_window, text=" HATA ANALİZİ - RESIDUALS", 
                font=('Arial', 18, 'bold'), bg='white', fg='black', pady=20).pack()
        
        residuals = self.interactive_y - self.interactive_y_pred
        
        fig = Figure(figsize=(14, 6), facecolor='white', dpi=100)
        gs = fig.add_gridspec(1, 2, wspace=0.3, left=0.08, right=0.95, top=0.88, bottom=0.12)
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Histogram
        n, bins, patches = ax1.hist(residuals, bins=20, color='steelblue', 
                                     edgecolor='black', alpha=0.8, linewidth=1.5)
        
        cm = plt.cm.coolwarm
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c), 'alpha', 0.8)
        
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma) * len(residuals) * (bins[1]-bins[0]), 
                 'r-', linewidth=3, label=f'Normal Dağılım\nμ={mu:.3f}, σ={sigma:.3f}')
        
        ax1.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax1.set_title("RESIDUALS HISTOGRAM\n(Artıkların Dağılımı)", 
                     fontweight='bold', fontsize=14, pad=15,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))
        ax1.set_xlabel("Artık Değerleri", fontweight='bold', fontsize=12)
        ax1.set_ylabel("Frekans", fontweight='bold', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        # Scatter
        scatter = ax2.scatter(self.interactive_y_pred, residuals, 
                             c=np.abs(residuals), cmap='plasma', 
                             s=100, alpha=0.8, edgecolors='black', linewidth=1.5)
        
        ax2.axhline(0, color='red', linestyle='--', linewidth=2.5, 
                   alpha=0.9, label='Y=0 (İdeal)', zorder=5)
        ax2.axhline(residuals.std(), color='orange', linestyle=':', 
                   linewidth=2, alpha=0.7, label=f'+1σ ({residuals.std():.3f})')
        ax2.axhline(-residuals.std(), color='orange', linestyle=':', 
                   linewidth=2, alpha=0.7, label=f'-1σ ({-residuals.std():.3f})')
        
        ax2.set_title("RESIDUALS SCATTER PLOT\n(Artıklar vs. Tahmin)", 
                     fontweight='bold', fontsize=14, pad=15,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        ax2.set_xlabel("Tahmin Edilen Y", fontweight='bold', fontsize=12)
        ax2.set_ylabel("Artıklar", fontweight='bold', fontsize=12)
        ax2.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        cbar = fig.colorbar(scatter, ax=ax2, pad=0.02)
        cbar.set_label('Artık Büyüklüğü', rotation=270, labelpad=20, fontweight='bold', fontsize=11)
        
        fig.text(0.5, 0.02, "💡 İdeal durumda artıklar sıfır etrafında rastgele dağılmalıdır",
                ha='center', fontsize=12, style='italic', weight='bold')
        
        canvas = FigureCanvasTkAgg(fig, residuals_window)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=(0, 20))
        canvas.draw()
    
    def show_interactive_statistics(self):
        """İnteraktif veriler için istatistik tablosu"""
        if self.interactive_model is None:
            return
        
        # Yeni pencere oluştur
        stats_window = Toplevel(self.root)
        stats_window.title("📊 İSTATİSTİK TABLOSU - İnteraktif Veriler")
        stats_window.geometry("1200x800")
        stats_window.configure(bg='white')
        
        title_label = tk.Label(stats_window, text="📊 İNTERAKTİF VERİLER - İSTATİSTİK ANALİZİ", 
                               font=('Arial', 18, 'bold'), bg='white', fg='black', pady=20)
        title_label.pack()
        
        fig = Figure(figsize=(12, 10), facecolor='white', dpi=100)
        canvas = FigureCanvasTkAgg(fig, stats_window)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
        
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 1], hspace=0.5,
                             top=0.93, bottom=0.05, left=0.1, right=0.9)
        
        ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        
        model = self.interactive_model
        
        # TABLO 1: Model İstatistikleri
        ax1.text(0.5, 1.15, 'MODEL İSTATİSTİKLERİ ve ANOVA', 
                ha='center', va='top', fontsize=16, fontweight='bold', transform=ax1.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', edgecolor='black', linewidth=2))
        
        table1_data = [
            ['Metrik', 'Değer', 'Açıklama'],
            ['R² (R-squared)', f'{model.rsquared:.6f}', 'Modelin açıklama gücü'],
            ['Düzeltilmiş R²', f'{model.rsquared_adj:.6f}', 'Düzeltilmiş açıklama gücü'],
            ['F-İstatistiği (ANOVA)', f'{model.fvalue:.4f}', 'Model anlamlılık testi'],
            ['Prob(F-statistic)', f'{model.f_pvalue:.8f}', 'ANOVA P-VALUE'],
        ]
        
        table1 = ax1.table(cellText=table1_data, cellLoc='left',
                          bbox=[0.05, 0.05, 0.9, 0.80], colWidths=[0.35, 0.25, 0.4])
        table1.auto_set_font_size(False)
        table1.set_fontsize(13)
        
        for i in range(3):
            table1[(0, i)].set_facecolor('black')
            table1[(0, i)].set_text_props(weight='bold', color='white', fontsize=14)
            table1[(0, i)].set_edgecolor('black')
            table1[(0, i)].set_linewidth(2)
        
        for i in range(1, len(table1_data)):
            for j in range(3):
                cell = table1[(i, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
                if i == 4:
                    cell.set_facecolor('#ffeb3b')
                    if j == 2:
                        cell.set_text_props(weight='bold', fontsize=13)
        
        # TABLO 2: Parametre P-VALUES
        ax2.text(0.5, 1.12, 'PARAMETRE TAHMİNLERİ ve P-VALUES', 
                ha='center', va='top', fontsize=16, fontweight='bold', transform=ax2.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', edgecolor='black', linewidth=2))
        
        table2_data = [
            ['Parametre', 'Katsayı', 'Std. Hata', 't-değeri', 'P-value', 'Güven Aralığı (95%)'],
            ['Sabit Terim (b)', f'{model.params[0]:.6f}', f'{model.bse[0]:.6f}', 
             f'{model.tvalues[0]:.4f}', f'{model.pvalues[0]:.8f}',
             f'[{model.conf_int()[0][0]:.4f}, {model.conf_int()[0][1]:.4f}]'],
            ['Eğim (a)', f'{model.params[1]:.6f}', f'{model.bse[1]:.6f}', 
             f'{model.tvalues[1]:.4f}', f'{model.pvalues[1]:.8f}',
             f'[{model.conf_int()[1][0]:.4f}, {model.conf_int()[1][1]:.4f}]']
        ]
        
        table2 = ax2.table(cellText=table2_data, cellLoc='center',
                          bbox=[0.02, 0.02, 0.96, 0.83], colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2])
        table2.auto_set_font_size(False)
        table2.set_fontsize(12)
        
        for i in range(6):
            table2[(0, i)].set_facecolor('black')
            table2[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)
            table2[(0, i)].set_edgecolor('black')
            table2[(0, i)].set_linewidth(2)
        
        for i in range(1, len(table2_data)):
            for j in range(6):
                cell = table2[(i, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
                if j == 4:
                    cell.set_facecolor('#ffeb3b')
                    cell.set_text_props(weight='bold', fontsize=12)
        
        # TABLO 3: Ek İstatistikler
        ax3.text(0.5, 1.15, 'EK İSTATİSTİKLER', 
                ha='center', va='top', fontsize=16, fontweight='bold', transform=ax3.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', edgecolor='black', linewidth=2))
        
        table3_data = [
            ['İstatistik', 'Değer', 'Açıklama'],
            ['Log-Likelihood', f'{model.llf:.4f}', 'Logaritmik olabilirlik'],
            ['AIC (Akaike)', f'{model.aic:.4f}', 'Akaike Bilgi Kriteri'],
            ['BIC (Bayesian)', f'{model.bic:.4f}', 'Bayesian Bilgi Kriteri'],
            ['Gözlem Sayısı', f'{int(model.nobs)}', 'Toplam veri noktası sayısı'],
        ]
        
        table3 = ax3.table(cellText=table3_data, cellLoc='left',
                          bbox=[0.05, 0.05, 0.9, 0.80], colWidths=[0.35, 0.25, 0.4])
        table3.auto_set_font_size(False)
        table3.set_fontsize(13)
        
        for i in range(3):
            table3[(0, i)].set_facecolor('black')
            table3[(0, i)].set_text_props(weight='bold', color='white', fontsize=14)
            table3[(0, i)].set_edgecolor('black')
            table3[(0, i)].set_linewidth(2)
        
        for i in range(1, len(table3_data)):
            for j in range(3):
                cell = table3[(i, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
        
        fig.text(0.5, 0.02, "⭐ P-value < 0.05 ise parametre istatistiksel olarak anlamlıdır",
                ha='center', fontsize=12, style='italic', weight='bold')
        
        canvas.draw()
    
    # ========================================================================
    # TAB 3: CSV DOSYASI ANALİZİ
    # ========================================================================
    def create_tab3_csv_analysis(self):
        """
        Sekme 3: CSV Dosyasından Veri Yükleme ve Analiz
        
        Kullanıcı CSV dosyası seçer ve regresyon analizi yapar
        """
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="📁 CSV Analizi")
        
        main_frame = tk.Frame(tab3, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Sol: Grafik alanı
        canvas_frame = tk.Frame(main_frame, bg='white', relief='solid', bd=2)
        canvas_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.fig_csv = Figure(figsize=(8, 6), facecolor='white')
        self.ax_csv = self.fig_csv.add_subplot(111)
        self.ax_csv.set_xlabel('X (Bağımsız Değişken)', fontsize=12, fontweight='bold')
        self.ax_csv.set_ylabel('Y (Bağımlı Değişken)', fontsize=12, fontweight='bold')
        self.ax_csv.set_title('📁 CSV Dosyası Seçin', fontsize=13, fontweight='bold')
        self.ax_csv.grid(True, alpha=0.3)
        
        self.canvas_csv = FigureCanvasTkAgg(self.fig_csv, canvas_frame)
        self.canvas_csv.get_tk_widget().pack(fill='both', expand=True)
        
        # CSV veri listeleri
        self.X_csv = None
        self.y_csv = None
        self.csv_data = None
        self.csv_x_label = "X"
        self.csv_y_label = "Y"
        
        # Sağ: Kontrol paneli
        control_frame = tk.Frame(main_frame, bg='white', relief='solid', bd=2, width=300)
        control_frame.pack(side='left', fill='y', padx=5)
        control_frame.pack_propagate(False)
        
        tk.Label(control_frame, text="📁 CSV KONTROL PANELİ", 
                font=('Arial', 14, 'bold'), bg='white', fg='#2196F3').pack(pady=20)
        
        # Dosya seçme butonu
        tk.Button(control_frame, text="📂 CSV Dosyası Seç", command=self.load_csv_file,
                 font=('Arial', 13, 'bold'), bg='#2196F3', fg='white',
                 padx=30, pady=15, cursor='hand2').pack(pady=10)
        
        # Dosya bilgisi
        self.csv_file_label = tk.Label(control_frame, text="Dosya: -", 
                                       font=('Arial', 10), bg='white', fg='gray',
                                       wraplength=250)
        self.csv_file_label.pack(pady=5, padx=10)
        
        tk.Frame(control_frame, height=2, bg='#e0e0e0').pack(fill='x', padx=20, pady=10)
        
        # Hesapla butonu
        self.csv_calc_btn = tk.Button(control_frame, text="📊 HESAPLA (OLS)", 
                                      command=self.calculate_csv_regression,
                                      font=('Arial', 13, 'bold'), bg='#4CAF50', fg='white',
                                      padx=30, pady=15, cursor='hand2', state='disabled')
        self.csv_calc_btn.pack(pady=10)
        
        # İstatistik Tablosu butonu
        self.csv_stats_btn = tk.Button(control_frame, text="📈 İstatistik Tablosu", 
                                       command=self.show_csv_statistics,
                                       font=('Arial', 12, 'bold'), bg='white', fg='black',
                                       padx=20, pady=15, cursor='hand2', relief='solid', bd=2,
                                       state='disabled')
        self.csv_stats_btn.pack(pady=10)
        
        # Hata Analizi butonu
        self.csv_error_btn = tk.Button(control_frame, text="📉 Hata Analizi", 
                                       command=self.show_csv_residuals,
                                       font=('Arial', 12, 'bold'), bg='white', fg='black',
                                       padx=20, pady=15, cursor='hand2', relief='solid', bd=2,
                                       state='disabled')
        self.csv_error_btn.pack(pady=10)
        
        tk.Button(control_frame, text="🗑️ Temizle", command=self.clear_csv_data,
                 font=('Arial', 13, 'bold'), bg='#f44336', fg='white',
                 padx=30, pady=15, cursor='hand2').pack(pady=10)
        
        # CSV model ve veri sakla
        self.csv_model = None
        self.csv_X_data = None
        self.csv_y_data = None
        self.csv_y_pred = None
        
        tk.Frame(control_frame, height=2, bg='#e0e0e0').pack(fill='x', padx=20, pady=20)
        
        tk.Label(control_frame, text="📋 SONUÇLAR", 
                font=('Arial', 13, 'bold'), bg='white', fg='#666').pack(pady=10)
        
        self.csv_equation = tk.Label(control_frame, text="Denklem: -", 
                                     font=('Arial', 11), bg='white', fg='#d32f2f',
                                     wraplength=250)
        self.csv_equation.pack(pady=10, padx=10)
        
        self.csv_r2 = tk.Label(control_frame, text="R² = -", font=('Arial', 11), bg='white')
        self.csv_r2.pack(pady=5)
        
        self.csv_data_info = tk.Label(control_frame, text="Veri Sayısı: 0", 
                                      font=('Arial', 11), bg='white')
        self.csv_data_info.pack(pady=5)
    
    def load_csv_file(self):
        """CSV dosyası yükle"""
        filename = filedialog.askopenfilename(
            title="CSV Dosyası Seç",
            filetypes=[("CSV dosyaları", "*.csv"), ("Tüm dosyalar", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            import os
            
            # CSV'yi oku
            df = pd.read_csv(filename)
            
            # En az 2 sütun olmalı
            if len(df.columns) < 2:
                messagebox.showerror("Hata", "CSV dosyası en az 2 sütun içermelidir!\n(X, Y)")
                return
            
            # Sütun sayısına göre veri seç
            if len(df.columns) == 2:
                # 2 sütun: 1. sütun X, 2. sütun Y (Person ID yok)
                self.X_csv = df.iloc[:, 0].values
                self.y_csv = df.iloc[:, 1].values
                x_col_name = df.columns[0]
                y_col_name = df.columns[1]
                file_info = f"Dosya: {os.path.basename(filename)}\nVeri: {len(self.X_csv)} satır\nX: {x_col_name}\nY: {y_col_name}"
            else:
                # 3+ sütun: İlk sütunu atla (Person ID), 2. sütun X, 3. sütun Y
                self.X_csv = df.iloc[:, 1].values
                self.y_csv = df.iloc[:, 2].values
                x_col_name = df.columns[1]
                y_col_name = df.columns[2]
                file_info = f"Dosya: {os.path.basename(filename)}\nVeri: {len(self.X_csv)} satır\nID: {df.columns[0]}\nX: {x_col_name}\nY: {y_col_name}"
            
            self.csv_data = df
            self.csv_x_label = x_col_name
            self.csv_y_label = y_col_name
            
            # Dosya bilgisini güncelle
            self.csv_file_label.config(text=file_info, fg='green')
            
            # Hesapla butonunu aktif et
            self.csv_calc_btn.config(state='normal')
            
            # Veriyi göster
            self.ax_csv.clear()
            self.ax_csv.scatter(self.X_csv, self.y_csv, alpha=0.7, s=60, 
                               color='blue', edgecolors='black', linewidth=0.5)
            self.ax_csv.set_xlabel(f'{x_col_name}', fontsize=12, fontweight='bold')
            self.ax_csv.set_ylabel(f'{y_col_name}', fontsize=12, fontweight='bold')
            self.ax_csv.set_title(f'CSV Verisi Yüklendi: {x_col_name} vs {y_col_name}', 
                                 fontsize=13, fontweight='bold')
            self.ax_csv.grid(True, alpha=0.3)
            self.canvas_csv.draw()
            
            self.csv_data_info.config(text=f"Veri Sayısı: {len(self.X_csv)}")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Dosya yüklenirken hata oluştu:\n{str(e)}")
    
    def calculate_csv_regression(self):
        """CSV verilerine OLS uygula"""
        if self.X_csv is None or self.y_csv is None:
            return
        
        # OLS Algoritması
        model, a_slope, b_intercept = calculate_linear_regression_ols(self.X_csv, self.y_csv)
        
        # Tahmin
        X_with_const = sm.add_constant(self.X_csv)
        y_pred = model.predict(X_with_const)
        
        # Sakla
        self.csv_model = model
        self.csv_X_data = self.X_csv
        self.csv_y_data = self.y_csv
        self.csv_y_pred = y_pred
        
        # Butonları aktif et
        self.csv_stats_btn.config(state='normal')
        self.csv_error_btn.config(state='normal')
        
        # Metrikler
        r2, _ = calculate_performance_metrics(self.y_csv, y_pred)
        
        # Çiz
        self.ax_csv.clear()
        self.ax_csv.scatter(self.X_csv, self.y_csv, alpha=0.7, s=60, 
                           color='blue', edgecolors='black', linewidth=0.5, label='Veri Noktaları')
        
        # Regresyon doğrusu
        X_line = np.linspace(self.X_csv.min(), self.X_csv.max(), 100)
        y_line = b_intercept + a_slope * X_line
        self.ax_csv.plot(X_line, y_line, 'r-', linewidth=3, label='OLS Regresyon')
        
        # Sütun isimlerini kullan
        self.ax_csv.set_xlabel(f'{self.csv_x_label}', fontsize=12, fontweight='bold')
        self.ax_csv.set_ylabel(f'{self.csv_y_label}', fontsize=12, fontweight='bold')
        self.ax_csv.set_title(f'Regresyon: {self.csv_y_label} = f({self.csv_x_label})', 
                             fontsize=13, fontweight='bold')
        self.ax_csv.legend(fontsize=10)
        self.ax_csv.grid(True, alpha=0.3)
        self.canvas_csv.draw()
        
        # Sonuçları göster (sütun isimleriyle)
        self.csv_equation.config(text=f"{self.csv_y_label} = {a_slope:.3f}*{self.csv_x_label} + {b_intercept:.3f}")
        self.csv_r2.config(text=f"R² = {r2:.4f}")
    
    def show_csv_statistics(self):
        """CSV veriler için istatistik tablosu"""
        if self.csv_model is None:
            return
        
        # İnteraktif ile aynı fonksiyon yapısı, sadece csv_model kullanıyor
        stats_window = Toplevel(self.root)
        stats_window.title("📊 İSTATİSTİK TABLOSU - CSV Verisi")
        stats_window.geometry("1200x800")
        stats_window.configure(bg='white')
        
        tk.Label(stats_window, text="📊 CSV VERİSİ - İSTATİSTİK ANALİZİ", 
                font=('Arial', 18, 'bold'), bg='white', fg='black', pady=20).pack()
        
        fig = Figure(figsize=(12, 10), facecolor='white', dpi=100)
        canvas = FigureCanvasTkAgg(fig, stats_window)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
        
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 1], hspace=0.5,
                             top=0.93, bottom=0.05, left=0.1, right=0.9)
        
        ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        
        model = self.csv_model
        
        # TABLO 1
        ax1.text(0.5, 1.15, 'MODEL İSTATİSTİKLERİ ve ANOVA', 
                ha='center', va='top', fontsize=16, fontweight='bold', transform=ax1.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', edgecolor='black', linewidth=2))
        
        table1_data = [
            ['Metrik', 'Değer', 'Açıklama'],
            ['R² (R-squared)', f'{model.rsquared:.6f}', 'Modelin açıklama gücü'],
            ['Düzeltilmiş R²', f'{model.rsquared_adj:.6f}', 'Düzeltilmiş açıklama gücü'],
            ['F-İstatistiği (ANOVA)', f'{model.fvalue:.4f}', 'Model anlamlılık testi'],
            ['Prob(F-statistic)', f'{model.f_pvalue:.8f}', 'ANOVA P-VALUE'],
        ]
        
        table1 = ax1.table(cellText=table1_data, cellLoc='left',
                          bbox=[0.05, 0.05, 0.9, 0.80], colWidths=[0.35, 0.25, 0.4])
        table1.auto_set_font_size(False)
        table1.set_fontsize(13)
        
        for i in range(3):
            table1[(0, i)].set_facecolor('black')
            table1[(0, i)].set_text_props(weight='bold', color='white', fontsize=14)
            table1[(0, i)].set_edgecolor('black')
            table1[(0, i)].set_linewidth(2)
        
        for i in range(1, len(table1_data)):
            for j in range(3):
                cell = table1[(i, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
                if i == 4:
                    cell.set_facecolor('#ffeb3b')
                    if j == 2:
                        cell.set_text_props(weight='bold', fontsize=13)
        
        # TABLO 2
        ax2.text(0.5, 1.12, 'PARAMETRE TAHMİNLERİ ve P-VALUES', 
                ha='center', va='top', fontsize=16, fontweight='bold', transform=ax2.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', edgecolor='black', linewidth=2))
        
        table2_data = [
            ['Parametre', 'Katsayı', 'Std. Hata', 't-değeri', 'P-value', 'Güven Aralığı (95%)'],
            ['Sabit Terim (b)', f'{model.params[0]:.6f}', f'{model.bse[0]:.6f}', 
             f'{model.tvalues[0]:.4f}', f'{model.pvalues[0]:.8f}',
             f'[{model.conf_int()[0][0]:.4f}, {model.conf_int()[0][1]:.4f}]'],
            ['Eğim (a)', f'{model.params[1]:.6f}', f'{model.bse[1]:.6f}', 
             f'{model.tvalues[1]:.4f}', f'{model.pvalues[1]:.8f}',
             f'[{model.conf_int()[1][0]:.4f}, {model.conf_int()[1][1]:.4f}]']
        ]
        
        table2 = ax2.table(cellText=table2_data, cellLoc='center',
                          bbox=[0.02, 0.02, 0.96, 0.83], colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2])
        table2.auto_set_font_size(False)
        table2.set_fontsize(12)
        
        for i in range(6):
            table2[(0, i)].set_facecolor('black')
            table2[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)
            table2[(0, i)].set_edgecolor('black')
            table2[(0, i)].set_linewidth(2)
        
        for i in range(1, len(table2_data)):
            for j in range(6):
                cell = table2[(i, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
                if j == 4:
                    cell.set_facecolor('#ffeb3b')
                    cell.set_text_props(weight='bold', fontsize=12)
        
        # TABLO 3
        ax3.text(0.5, 1.15, 'EK İSTATİSTİKLER', 
                ha='center', va='top', fontsize=16, fontweight='bold', transform=ax3.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', edgecolor='black', linewidth=2))
        
        table3_data = [
            ['İstatistik', 'Değer', 'Açıklama'],
            ['Log-Likelihood', f'{model.llf:.4f}', 'Logaritmik olabilirlik'],
            ['AIC (Akaike)', f'{model.aic:.4f}', 'Akaike Bilgi Kriteri'],
            ['BIC (Bayesian)', f'{model.bic:.4f}', 'Bayesian Bilgi Kriteri'],
            ['Gözlem Sayısı', f'{int(model.nobs)}', 'CSV dosyasındaki veri sayısı'],
        ]
        
        table3 = ax3.table(cellText=table3_data, cellLoc='left',
                          bbox=[0.05, 0.05, 0.9, 0.80], colWidths=[0.35, 0.25, 0.4])
        table3.auto_set_font_size(False)
        table3.set_fontsize(13)
        
        for i in range(3):
            table3[(0, i)].set_facecolor('black')
            table3[(0, i)].set_text_props(weight='bold', color='white', fontsize=14)
            table3[(0, i)].set_edgecolor('black')
            table3[(0, i)].set_linewidth(2)
        
        for i in range(1, len(table3_data)):
            for j in range(3):
                cell = table3[(i, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
        
        fig.text(0.5, 0.02, "⭐ P-value < 0.05 ise parametre istatistiksel olarak anlamlıdır",
                ha='center', fontsize=12, style='italic', weight='bold')
        
        canvas.draw()
    
    def show_csv_residuals(self):
        """CSV verileri için hata analizi"""
        if self.csv_model is None or self.csv_y_data is None:
            return
        
        residuals_window = Toplevel(self.root)
        residuals_window.geometry("1400x700")
        residuals_window.configure(bg='white')
        
        tk.Label(residuals_window, text="📉 HATA ANALİZİ - CSV VERİSİ", 
                font=('Arial', 18, 'bold'), bg='white', fg='black', pady=20).pack()
        
        residuals = self.csv_y_data - self.csv_y_pred
        
        fig = Figure(figsize=(14, 6), facecolor='white', dpi=100)
        gs = fig.add_gridspec(1, 2, wspace=0.3, left=0.08, right=0.95, top=0.88, bottom=0.12)
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Histogram
        n, bins, patches = ax1.hist(residuals, bins=20, color='steelblue', 
                                     edgecolor='black', alpha=0.8, linewidth=1.5)
        
        cm = plt.cm.coolwarm
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c), 'alpha', 0.8)
        
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma) * len(residuals) * (bins[1]-bins[0]), 
                 'r-', linewidth=3, label=f'Normal Dağılım\nμ={mu:.3f}, σ={sigma:.3f}')
        
        ax1.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax1.set_title("RESIDUALS HISTOGRAM\n(Artıkların Dağılımı)", 
                     fontweight='bold', fontsize=14, pad=15,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))
        ax1.set_xlabel("Artık Değerleri", fontweight='bold', fontsize=12)
        ax1.set_ylabel("Frekans", fontweight='bold', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        # Scatter
        scatter = ax2.scatter(self.csv_y_pred, residuals, 
                             c=np.abs(residuals), cmap='plasma', 
                             s=100, alpha=0.8, edgecolors='black', linewidth=1.5)
        
        ax2.axhline(0, color='red', linestyle='--', linewidth=2.5, 
                   alpha=0.9, label='Y=0 (İdeal)', zorder=5)
        ax2.axhline(residuals.std(), color='orange', linestyle=':', 
                   linewidth=2, alpha=0.7, label=f'+1σ ({residuals.std():.3f})')
        ax2.axhline(-residuals.std(), color='orange', linestyle=':', 
                   linewidth=2, alpha=0.7, label=f'-1σ ({-residuals.std():.3f})')
        
        ax2.set_title("RESIDUALS SCATTER PLOT\n(Artıklar vs. Tahmin)", 
                     fontweight='bold', fontsize=14, pad=15,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        ax2.set_xlabel("Tahmin Edilen Y", fontweight='bold', fontsize=12)
        ax2.set_ylabel("Artıklar", fontweight='bold', fontsize=12)
        ax2.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        cbar = fig.colorbar(scatter, ax=ax2, pad=0.02)
        cbar.set_label('Artık Büyüklüğü', rotation=270, labelpad=20, fontweight='bold', fontsize=11)
        
        fig.text(0.5, 0.02, "💡 İdeal durumda artıklar sıfır etrafında rastgele dağılmalıdır",
                ha='center', fontsize=12, style='italic', weight='bold')
        
        canvas = FigureCanvasTkAgg(fig, residuals_window)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=(0, 20))
        canvas.draw()
    
    def clear_csv_data(self):
        """CSV verilerini temizle"""
        self.X_csv = None
        self.y_csv = None
        self.csv_data = None
        self.csv_x_label = "X"
        self.csv_y_label = "Y"
        
        self.ax_csv.clear()
        self.ax_csv.set_xlabel('X (Bağımsız Değişken)', fontsize=12, fontweight='bold')
        self.ax_csv.set_ylabel('Y (Bağımlı Değişken)', fontsize=12, fontweight='bold')
        self.ax_csv.set_title('📁 CSV Dosyası Seçin', fontsize=13, fontweight='bold')
        self.ax_csv.grid(True, alpha=0.3)
        self.canvas_csv.draw()
        
        self.csv_file_label.config(text="Dosya: -", fg='gray')
        self.csv_equation.config(text="Denklem: -")
        self.csv_r2.config(text="R² = -")
        self.csv_data_info.config(text="Veri Sayısı: 0")
        
        self.csv_calc_btn.config(state='disabled')
        self.csv_stats_btn.config(state='disabled')
        self.csv_error_btn.config(state='disabled')
        
        self.csv_model = None
        self.csv_X_data = None
        self.csv_y_data = None
        self.csv_y_pred = None


# ============================================================================
# PROGRAMI BAŞLAT
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = LinearRegressionApp(root)
    root.mainloop()
