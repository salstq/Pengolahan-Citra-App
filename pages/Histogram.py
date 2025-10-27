import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import skew, kurtosis, chi2_contingency
from math import log2

st.set_page_config(page_title="Analisis Statistik Citra", layout="wide")
st.title("📈 Analisis Statistik Citra")

st.markdown("""
### 🧩 Deskripsi
Upload satu atau dua citra untuk melihat analisis statistik seperti:
- Pearson Correlation antar kanal RGB  
- Skewness  
- Kurtosis  
- Entropy  
- Chi-Square  
Serta *matching score* antara dua citra.
""")

# ========== Fungsi bantu ==========
def to_grayscale(img):
    return np.array(img.convert("L"))

def calc_entropy(hist):
    hist_norm = hist / np.sum(hist)
    hist_norm = hist_norm[hist_norm > 0]
    return -np.sum(hist_norm * np.log2(hist_norm))

def chi_square_stat(hist):
    expected = np.ones_like(hist) * np.mean(hist)
    return np.sum((hist - expected) ** 2 / expected)

def pearson_rgb(img):
    arr = np.array(img)
    r, g, b = arr[:,:,0].flatten(), arr[:,:,1].flatten(), arr[:,:,2].flatten()
    corr_rg = np.corrcoef(r, g)[0,1]
    corr_rb = np.corrcoef(r, b)[0,1]
    corr_gb = np.corrcoef(g, b)[0,1]
    return corr_rg, corr_rb, corr_gb

def match_images(img1, img2):
    g1 = to_grayscale(img1)
    g2 = to_grayscale(img2)
    min_h = min(g1.shape[0], g2.shape[0])
    min_w = min(g1.shape[1], g2.shape[1])
    g1 = g1[:min_h, :min_w]
    g2 = g2[:min_h, :min_w]
    corr = np.corrcoef(g1.flatten(), g2.flatten())[0,1]
    return corr

# ========== Upload ==========
col1, col2 = st.columns(2)
with col1:
    img1 = st.file_uploader("Upload Citra Pertama", type=["png", "jpg", "jpeg"])
with col2:
    img2 = st.file_uploader("Upload Citra Kedua (opsional)", type=["png", "jpg", "jpeg"])

if img1 is not None:
    image1 = Image.open(img1).convert("RGB")

    col_img1, col_img2, col_match = st.columns(3)
    col_img1.image(image1, caption="Citra Pertama", use_container_width=True)

    if img2 is not None:
        image2 = Image.open(img2).convert("RGB")
        col_img2.image(image2, caption="Citra Kedua", use_container_width=True)
        match_score = match_images(image1, image2)
        col_match.metric("Matching Score", f"{match_score:.4f}")
    else:
        col_img2.empty()
        col_match.info("Upload citra kedua untuk melihat nilai matching")

    # ===== Analisis Statistik =====
    gray = to_grayscale(image1)
    hist, _ = np.histogram(gray, bins=256, range=(0,256))

    entropy_val = calc_entropy(hist)
    skew_val = skew(gray.flatten())
    kurt_val = kurtosis(gray.flatten())
    chi_val = chi_square_stat(hist)
    pearson_vals = pearson_rgb(image1)

    # ===== Histogram =====
    fig, ax = plt.subplots()
    ax.plot(hist, color="gray")
    ax.set_title("Histogram Grayscale")
    st.pyplot(fig)

    # ===== Baris Bawah =====
    col_stats, col_rgb, col_match2 = st.columns(3)

    # Statistik utama
    stats_data = {
        "Statistik": ["Entropy", "Skewness", "Kurtosis", "Chi-Square"],
        "Nilai": [f"{entropy_val:.4f}", f"{skew_val:.4f}", f"{kurt_val:.4f}", f"{chi_val:.4f}"]
    }
    df_stats = pd.DataFrame(stats_data)
    col_stats.dataframe(df_stats, use_container_width=True)

    # Pearson RGB
    pearson_data = {
        "Pasangan Kanal": ["R-G", "R-B", "G-B"],
        "Pearson Correlation": [f"{pearson_vals[0]:.4f}", f"{pearson_vals[1]:.4f}", f"{pearson_vals[2]:.4f}"]
    }
    df_pearson = pd.DataFrame(pearson_data)
    col_rgb.dataframe(df_pearson, use_container_width=True)

    # Matching jika ada dua gambar
    if img2 is not None:
        col_match2.success(f"**Matching Score:** {match_score:.4f}")
    else:
        col_match2.warning("Belum ada citra kedua.")
else:
    st.info("Silakan upload minimal satu citra untuk memulai analisis.")
