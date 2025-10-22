# 3_📊_Threshold_Equalization.py
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import pandas as pd

st.set_page_config(page_title="Threshold & Equalization", page_icon="📊", layout="wide")

# -------------------------
# Helper functions
# -------------------------

def pil_to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        return img.convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def image_to_array(img: Image.Image) -> np.ndarray:
    return np.array(pil_to_rgb(img), dtype=np.uint8)


def to_grayscale_arr(arr: np.ndarray) -> np.ndarray:
    """Convert HxWx3 uint8 image array to single-channel grayscale (uint8)."""
    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    return np.clip(gray, 0, 255).astype(np.uint8)


def find_two_peaks(hist: np.ndarray) -> list:
    """Find two highest local peaks indices in histogram array (length 256).
    Returns sorted list [peak1, peak2] (peak intensities)."""
    peaks = []
    # local maxima detection (naive)
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append((hist[i], i))
    if not peaks:
        # fallback: global max
        idx = int(np.argmax(hist))
        return [idx]
    # sort peaks by count desc
    peaks_sorted = sorted(peaks, key=lambda x: x[0], reverse=True)
    top = [peaks_sorted[0][1]]
    if len(peaks_sorted) > 1:
        top.append(peaks_sorted[1][1])
    # ensure two peaks if possible
    return sorted(top)


def compute_threshold_from_peaks(peaks: list) -> int:
    if len(peaks) >= 2:
        return int(round((peaks[0] + peaks[1]) / 2))
    elif len(peaks) == 1:
        return peaks[0]
    else:
        return 128


def to_binary_image(gray_arr: np.ndarray, threshold: int) -> np.ndarray:
    out = np.where(gray_arr > threshold, 255, 0).astype(np.uint8)
    return out


def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """Perform histogram equalization on single-channel grayscale image (0-255)."""
    flat = gray.flatten()
    hist, bins = np.histogram(flat, bins=256, range=(0,255))
    cdf = hist.cumsum()
    # normalize cdf to [0,255]
    cdf_min = cdf[cdf > 0][0] if np.any(cdf>0) else 0
    cdf_norm = (cdf - cdf_min) / (cdf[-1] - cdf_min) * 255.0
    cdf_norm = np.clip(cdf_norm, 0, 255).astype(np.uint8)
    # map
    equalized = cdf_norm[flat]
    return equalized.reshape(gray.shape)


def plot_histogram_rgb(arr: np.ndarray):
    # arr: HxWx3 uint8
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    colors = ['r','g','b']
    labels = ['R','G','B']
    for i, c in enumerate(colors):
        hist, bins = np.histogram(arr[..., i].flatten(), bins=256, range=(0,255))
        ax.plot(hist, color=c, label=labels[i])
    ax.set_title('Histogram RGB')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_histogram_gray(gray: np.ndarray):
    fig, ax = plt.subplots(1,1,figsize=(6,3))
    hist, bins = np.histogram(gray.flatten(), bins=256, range=(0,255))
    ax.plot(hist, color='k')
    ax.set_title('Histogram Grayscale')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Count')
    plt.tight_layout()
    return fig, hist


def st_image_original(pil_img: Image.Image, caption=None):
    w, h = pil_img.size
    st.image(pil_img, caption=caption, use_column_width=False, width=w)

# -------------------------
# UI
# -------------------------
st.title("Thresholding dan Histogram Equalization 📊")
st.markdown("""
Upload satu citra, kemudian:
- Tampilkan Histogram RGB dan Grayscale dalam bentuk grafik dan tampilkan angkanya.
- Tampilkan nilai Threshold dari dua puncak histogram, kemudian ubah menjadi citra biner berdasarkan threshold.
- Lakukan Histogram Equalization (grayscale) dan tampilkan hasilnya.

**Catatan:** Hasil citra biner akan berupa 0 (hitam) dan 255 (putih).
""")

uploaded = st.file_uploader("Upload Gambar (png/jpg/jpeg)", type=["png","jpg","jpeg"], key="thresh")
if not uploaded:
    st.info("Silakan upload gambar untuk memulai proses.")
else:
    try:
        img = Image.open(uploaded)
        img = pil_to_rgb(img)
        arr = image_to_array(img)

        # compute grayscale
        gray = to_grayscale_arr(arr)

        # compute histograms
        fig_rgb = plot_histogram_rgb(arr)
        fig_gray, hist_gray = plot_histogram_gray(gray)

        # find two peaks
        peaks = find_two_peaks(hist_gray)
        if len(peaks) >= 2:
            peak_info = f"Peak intensities found at: {peaks[0]} and {peaks[1]}"
        else:
            peak_info = f"Peak intensity found at: {peaks[0]} (only one peak detected)"
        threshold = compute_threshold_from_peaks(peaks)

        # create binary image
        bin_img_arr = to_binary_image(gray, threshold)
        bin_pil = Image.fromarray(bin_img_arr)

        # equalization
        eq_gray = histogram_equalization(gray)
        eq_pil = Image.fromarray(eq_gray)

        # Layout similar to page sebelumnya: top row images, bottom row plots + values
        st.subheader("Hasil Proses")
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.caption("Gambar Input")
            st_image_original(Image.fromarray(arr), caption=f"Input — size: {img.size}")
        with c2:
            st.caption("Citra Biner (Threshold)")
            st_image_original(bin_pil.convert('RGB'), caption=f"Biner (threshold={threshold}) — size: {bin_pil.size}")
        with c3:
            st.caption("Hasil Equalization (Grayscale)")
            st_image_original(eq_pil.convert('RGB'), caption=f"Equalized — size: {eq_pil.size}")

        # Bottom row: histograms and values
        st.subheader("Histogram & Nilai")
        d1, d2, d3 = st.columns([1,1,1])
        with d1:
            st.caption("Histogram RGB")
            st.pyplot(fig_rgb)
            # show small summary table of top counts for each channel
            r_hist, _ = np.histogram(arr[...,0].flatten(), bins=256, range=(0,255))
            g_hist, _ = np.histogram(arr[...,1].flatten(), bins=256, range=(0,255))
            b_hist, _ = np.histogram(arr[...,2].flatten(), bins=256, range=(0,255))
            top_r = np.argmax(r_hist)
            top_g = np.argmax(g_hist)
            top_b = np.argmax(b_hist)
            st.write(f"Top intensitas — R: {top_r}, G: {top_g}, B: {top_b}")
        with d2:
            st.caption("Histogram Grayscale")
            st.pyplot(fig_gray)
            st.write(peak_info)
            st.write(f"Perhitungan threshold (midpoint): {threshold}")

            # Show histogram numeric table (few values) and full as downloadable CSV
            hist_df = pd.DataFrame({'intensity': np.arange(256), 'count': hist_gray})
            st.dataframe(hist_df.head(20))
            csv = hist_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Histogram Grayscale CSV", data=csv, file_name='hist_gray.csv', mime='text/csv')

        with d3:
            st.caption("Histogram Equalization - Before & After (Grayscale)")
            # plot before and after
            fig, axs = plt.subplots(1,2,figsize=(8,3))
            hist_before, _ = np.histogram(gray.flatten(), bins=256, range=(0,255))
            hist_after, _ = np.histogram(eq_gray.flatten(), bins=256, range=(0,255))
            axs[0].plot(hist_before); axs[0].set_title('Before')
            axs[1].plot(hist_after); axs[1].set_title('After')
            plt.tight_layout()
            st.pyplot(fig)

            # show small stats
            st.write(f"Mean sebelum: {np.mean(gray):.2f} — Mean sesudah: {np.mean(eq_gray):.2f}")

        # Downloads for images
        st.markdown('---')
        st.subheader('Download Hasil')
        buf_in = io.BytesIO()
        Image.fromarray(arr).save(buf_in, format='PNG')
        st.download_button('Download Input (PNG)', data=buf_in.getvalue(), file_name='input.png', mime='image/png')

        buf_bin = io.BytesIO()
        bin_pil.convert('RGB').save(buf_bin, format='PNG')
        st.download_button('Download Binary (PNG)', data=buf_bin.getvalue(), file_name='binary.png', mime='image/png')

        buf_eq = io.BytesIO()
        eq_pil.convert('RGB').save(buf_eq, format='PNG')
        st.download_button('Download Equalized (PNG)', data=buf_eq.getvalue(), file_name='equalized.png', mime='image/png')

    except Exception as e:
        st.error(f"Terjadi error saat memproses gambar: {e}")
