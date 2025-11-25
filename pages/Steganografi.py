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
# PVD STEGANOGRAPHY (BENAR) - Grayscale → Blue Channel
# ============================================================

# Range table (L-H, bit capacity)
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
    img = np.array(image).copy()

    # convert to grayscale
    gray = np.array(Image.fromarray(img).convert("L"))

    flat = gray.flatten().astype(int)
    h, w = gray.shape

    # prepare payload bits
    bits = ''.join(format(ord(c), '08b') for c in text) + '00000000'
    bit_index = 0

    for i in range(0, len(flat) - 1, 2):

        p1, p2 = flat[i], flat[i + 1]
        diff = abs(p1 - p2)

        L, H, k = get_range_info(diff)

        if bit_index >= len(bits):
            break

        seg = bits[bit_index:bit_index + k]
        if len(seg) < k:
            seg += '0' * (k - len(seg))
        bit_index += k

        value = int(seg, 2)

        # target diff
        new_diff = L + value

        # adjust p1 and p2
        if p1 >= p2:
            if diff > new_diff:
                p1 -= (diff - new_diff)
            else:
                p1 += (new_diff - diff)
        else:
            if diff > new_diff:
                p2 -= (diff - new_diff)
            else:
                p2 += (new_diff - diff)

        # clamp
        p1 = max(0, min(255, p1))
        p2 = max(0, min(255, p2))

        flat[i], flat[i + 1] = p1, p2

    stego_gray = flat.reshape(h, w)

    # masukkan ke channel blue saja
    b = stego_gray
    r = img[:, :, 0]
    g = img[:, :, 1]

    out = np.dstack([r, g, b])
    return Image.fromarray(out.astype(np.uint8))


def pvd_extract(image):
    img = np.array(image)
    blue = img[:, :, 2]

    flat = blue.flatten().astype(int)

    bits = ""

    for i in range(0, len(flat) - 1, 2):
        p1, p2 = flat[i], flat[i + 1]
        diff = abs(p1 - p2)

        L, H, k = get_range_info(diff)

        value = diff - L
        if value < 0:
            continue

        segment = format(value, f'0{k}b')
        bits += segment

        # decode per 8-bit char
        while len(bits) >= 8:
            byte = bits[:8]
            bits = bits[8:]
            char = chr(int(byte, 2))

            if char == '\x00':
                return ""  # end
            yield char


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

        if st.button("Embed LSB"):
            stego = lsb_embed(img, text)
            st.session_state["lsb"] = stego
            st.image(stego, caption="Citra Stego (LSB)", use_column_width=True)
            st.download_button("Download Hasil", image_to_bytes(stego), "lsb_stego.png")

        if st.button("Extract LSB"):
            if "lsb" in st.session_state:
                extracted = lsb_extract(st.session_state["lsb"])
                st.success(extracted)
            else:
                st.error("Belum ada citra stego!")

    # ----------------------------------------------------------
    # TAB 2 — HISTOGRAM SHIFTING
    # ----------------------------------------------------------
    with tab2:
        st.header("Histogram Shifting (Blue Channel)")

        if st.button("Embed HS"):
            stego = hs_embed(img, text)
            st.session_state["hs"] = stego
            st.image(stego, caption="Citra Stego (HS)", use_column_width=True)
            st.download_button("Download Hasil", image_to_bytes(stego), "hs_stego.png")

        if st.button("Extract HS"):
            if "hs" in st.session_state:
                extracted = hs_extract(st.session_state["hs"])
                st.success(extracted)
            else:
                st.error("Belum ada citra stego!")

    # ----------------------------------------------------------
    # TAB 3 — PVD
    # ----------------------------------------------------------
    with tab3:
        st.header("PVD Steganografi (Blue Channel)")

        if st.button("Embed PVD"):
            stego = pvd_embed(img, text)
            st.session_state["pvd"] = stego
            st.image(stego, caption="Citra Stego (PVD)", use_column_width=True)
            st.download_button("Download Hasil", image_to_bytes(stego), "pvd_stego.png")

        if st.button("Extract PVD"):
            if "pvd" in st.session_state:
                extracted = "".join(list(pvd_extract(st.session_state["pvd"])))
                st.success(extracted)
            else:
                st.error("Belum ada citra stego!")
