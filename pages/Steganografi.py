import streamlit as st
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Steganografi - Salsa", layout="wide")

# ============================================================
# Utility
# ============================================================
def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def bytes_to_image(b):
    return Image.open(io.BytesIO(b))

def to_uint8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)

# ============================================================
# LSB EMBED & EXTRACT
# ============================================================
def lsb_embed(image, text):
    img = np.array(image).copy()
    flat = img.reshape(-1, 3)

    bits = ''.join([format(ord(c), '08b') for c in text])
    bits += '00000000'  # terminator

    if len(bits) > len(flat):
        raise ValueError("Payload terlalu besar!")

    for i, bit in enumerate(bits):
        flat[i][2] = (flat[i][2] & 0xFE) | int(bit)

    stego = flat.reshape(img.shape)
    return Image.fromarray(stego)

def lsb_extract(image):
    img = np.array(image)
    flat = img.reshape(-1, 3)
    bits = ""
    text = ""

    for pixel in flat:
        bits += str(pixel[2] & 1)
        if len(bits) == 8:
            char = chr(int(bits, 2))
            if char == '\x00':
                break
            text += char
            bits = ""
    return text

# ============================================================
# HISTOGRAM SHIFTING (BLUE CHANNEL)
# ============================================================
def hs_embed(image, text):
    img = np.array(image).copy()
    blue = img[:, :, 2]

    h, w = blue.shape
    flat = blue.flatten()

    bits = ''.join([format(ord(c), '08b') for c in text]) + '00000000'
    if len(bits) > len(flat):
        raise ValueError("Payload terlalu besar!")

    flat = np.where(flat > 200, flat, flat + 1)

    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & 0xFE) | int(bit)

    img[:, :, 2] = flat.reshape(h, w)
    return Image.fromarray(to_uint8(img))

def hs_extract(image):
    img = np.array(image)
    flat = img[:, :, 2].flatten()

    bits = ""
    text = ""

    for v in flat:
        bits += str(v & 1)
        if len(bits) == 8:
            char = chr(int(bits, 2))
            if char == '\x00':
                break
            text += char
            bits = ""
    return text

# ============================================================
# PVD STEGANOGRAFI (FIX — warna 99.9% sama dengan citra asli)
# ============================================================

RANGES = [
    (0, 7, 3),
    (8, 15, 3),
    (16, 31, 4),
    (32, 63, 5),
    (64, 127, 6),
    (128, 255, 7)
]

def get_range_info(diff):
    for L, H, k in RANGES:
        if L <= diff <= H:
            return L, H, k
    return 0, 7, 3


def pvd_embed(image, text):
    # convert to numpy
    img = np.array(image).copy()
    
    # gunakan channel biru asli (tidak grayscale)
    blue = img[:, :, 2].astype(int)

    flat = blue.flatten()
    h, w = blue.shape

    # payload bit
    bits = ''.join(format(ord(c), '08b') for c in text) + '00000000'
    bit_index = 0
    total_bits = len(bits)

    for i in range(0, len(flat) - 1, 2):
        if bit_index >= total_bits:
            break

        p1, p2 = flat[i], flat[i + 1]
        diff = abs(p1 - p2)

        L, H, k = get_range_info(diff)

        segment = bits[bit_index:bit_index + k]
        if len(segment) < k:
            segment += '0' * (k - len(segment))

        bit_index += k
        value = int(segment, 2)

        target_diff = L + value

        # adjust pixel pair
        if p1 >= p2:
            if diff > target_diff:
                p1 -= (diff - target_diff)
            else:
                p1 += (target_diff - diff)
        else:
            if diff > target_diff:
                p2 -= (diff - target_diff)
            else:
                p2 += (target_diff - diff)

        p1 = np.clip(p1, 0, 255)
        p2 = np.clip(p2, 0, 255)

        flat[i], flat[i + 1] = p1, p2

    # rebuild blue channel
    new_blue = flat.reshape(h, w)

    # masukkan kembali tanpa sentuh channel lain
    img[:, :, 2] = new_blue.astype(np.uint8)

    return Image.fromarray(img.astype(np.uint8))



def pvd_extract(image):
    img = np.array(image)
    blue = img[:, :, 2].astype(int)
    flat = blue.flatten()

    bits = ""
    text = ""

    for i in range(0, len(flat) - 1, 2):
        p1, p2 = flat[i], flat[i + 1]
        diff = abs(p1 - p2)

        L, H, k = get_range_info(diff)

        value = diff - L
        if value < 0:
            continue

        segment = format(value, f'0{k}b')
        bits += segment

        # decode
        while len(bits) >= 8:
            byte = bits[:8]
            bits = bits[8:]
            char = chr(int(byte, 2))
            if char == '\x00':
                return text
            text += char

    return text




# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🕵️ Steganografi Citra (LSB • Histogram Shifting • PVD) – RGB")

uploaded = st.file_uploader("Upload Gambar", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Citra Awal", use_column_width=True)

    text = st.text_area("Teks yang akan disisipkan:")

    tab1, tab2, tab3 = st.tabs(["🔵 LSB", "🟣 Histogram Shifting", "🟢 PVD"])

    # ----------------------------------------------------------
    # TAB 1 — LSB
    # ----------------------------------------------------------
    with tab1:
        st.header("LSB Steganografi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Embed LSB", key="embed_lsb"):
                stego = lsb_embed(img, text)
                st.session_state["lsb"] = image_to_bytes(stego)
                st.image(stego, caption="Citra Stego (LSB)", use_column_width=True)
                st.download_button("Download Hasil", image_to_bytes(stego), "lsb_stego.png", key="dl_lsb")
        
        with col2:
            if st.button("Extract LSB", key="extract_lsb"):
                # jika ada stego dalam session_state (hasil embed)
                if "lsb" in st.session_state:
                    stego = bytes_to_image(st.session_state["lsb"])
                else:
                    stego = img  # gunakan gambar upload
        
                extracted = lsb_extract(stego)
                st.success(extracted)

    # ----------------------------------------------------------
    # TAB 2 — HISTOGRAM SHIFTING
    # ----------------------------------------------------------
    with tab2:
        st.header("Histogram Shifting (Blue Channel)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Embed HS", key="embed_hs"):
                stego = hs_embed(img, text)
                st.session_state["hs"] = image_to_bytes(stego)
                st.image(stego, caption="Citra Stego (HS)", use_column_width=True)
                st.download_button("Download Hasil", image_to_bytes(stego), "hs_stego.png", key="dl_hs")
        
        with col2:
            if st.button("Extract HS", key="extract_hs"):
                if "hs" in st.session_state:
                    stego = bytes_to_image(st.session_state["hs"])
                else:
                    stego = img  # fallback ke upload
                
                extracted = hs_extract(stego)
                st.success(extracted)
                
    # ----------------------------------------------------------
    # TAB 3 — PVD
    # ----------------------------------------------------------
    with tab3:
        st.header("PVD Steganografi (Blue Channel)")
    
        col1, col2 = st.columns(2)
    
        with col1:
            if st.button("Embed PVD", key="embed_pvd"):
                stego = pvd_embed(img, text)
                st.session_state["pvd"] = image_to_bytes(stego)
                st.image(stego, caption="Citra Stego (PVD)", use_column_width=True)
                st.download_button("Download Hasil", image_to_bytes(stego), "pvd_stego.png", key="dl_pvd")
    
        with col2:
            if st.button("Extract PVD", key="extract_pvd"):
                if "pvd" in st.session_state:
                    stego = bytes_to_image(st.session_state["pvd"])
                else:
                    stego = img
        
                extracted = pvd_extract(stego)
                st.success(extracted)



