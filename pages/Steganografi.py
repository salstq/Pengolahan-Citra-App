import streamlit as st
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Aplikasi Steganografi", layout="wide")
st.title("🕵️‍♂️ Aplikasi Steganografi Teks (LSB, Histogram Shifting, PVD)")

# =================================================================
# ===================== 1. FUNGSI LSB ==============================
# =================================================================

def embed_lsb(image, message):
    img = np.array(image)
    flat = img.flatten()

    # Ubah teks ke biner
    binary_msg = ''.join([format(ord(c), '08b') for c in message])
    binary_msg += "0000000000000000"   # tanda selesai (2 byte null)

    if len(binary_msg) > len(flat):
        raise ValueError("Pesan terlalu besar untuk disisipkan!")

    for i in range(len(binary_msg)):
        flat[i] = (flat[i] & 0xFE) | int(binary_msg[i])

    stego = flat.reshape(img.shape)
    return Image.fromarray(stego.astype(np.uint8))


def extract_lsb(image):
    img = np.array(image)
    flat = img.flatten()

    bits = ""
    for p in flat:
        bits += str(p & 1)

    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if byte == "00000000":
            break
        chars.append(chr(int(byte, 2)))

    return ''.join(chars)


# =================================================================
# ================ 2. HISTOGRAM SHIFTING ===========================
# =================================================================

def embed_histogram_shifting(image, message):
    img = np.array(image).astype(int)
    h, w, c = img.shape

    binary_msg = ''.join([format(ord(c), '08b') for c in message]) + "00000000"
    msg_idx = 0

    stego = img.copy()

    # ambil channel R saja agar sederhana
    channel = stego[:,:,0].flatten()

    for i in range(len(channel)):
        if msg_idx >= len(binary_msg):
            break

        if binary_msg[msg_idx] == "1":
            channel[i] += 1  # shift histogram
        msg_idx += 1

    stego[:,:,0] = channel.reshape(h, w)
    return Image.fromarray(stego.clip(0,255).astype(np.uint8))


def extract_histogram_shifting(image):
    img = np.array(image).astype(int)
    channel = img[:,:,0].flatten()

    bits = []

    for val in channel:
        if val % 2 == 1:
            bits.append("1")
        else:
            bits.append("0")

    # ubah ke teks
    chars = []
    for i in range(0, len(bits), 8):
        byte = ''.join(bits[i:i+8])
        if byte == "00000000":
            break
        chars.append(chr(int(byte,2)))

    return ''.join(chars)


# =================================================================
# ====================== 3. PVD METHOD =============================
# =================================================================

def embed_pvd(image, message):
    img = np.array(image)
    h, w, c = img.shape

    binary_msg = ''.join([format(ord(c), '08b') for c in message]) + "00000000"
    msg_idx = 0

    stego = img.copy()

    # gunakan pasangan pixel pada channel R saja
    pair = stego[:,:,0].flatten()

    for i in range(0, len(pair)-1, 2):
        if msg_idx >= len(binary_msg):
            break

        p1 = pair[i]
        p2 = pair[i+1]

        d = abs(p2 - p1)
        b = int(binary_msg[msg_idx])   # ambil 1 bit

        if b == 1:
            d += 1
        else:
            d -= 1

        # update p2
        if p2 >= p1:
            p2 = p1 + d
        else:
            p2 = p1 - d

        pair[i+1] = max(0, min(255, p2))
        msg_idx += 1

    stego[:,:,0] = pair.reshape(h, w)
    return Image.fromarray(stego.astype(np.uint8))


def extract_pvd(image):
    img = np.array(image)
    pair = img[:,:,0].flatten()

    bits = []

    for i in range(0, len(pair)-1, 2):
        p1 = pair[i]
        p2 = pair[i+1]
        d = abs(p2 - p1)

        bits.append("1" if d % 2 == 1 else "0")

    chars = []
    for i in range(0, len(bits), 8):
        byte = ''.join(bits[i:i+8])
        if byte == "00000000":
            break
        chars.append(chr(int(byte,2)))

    return ''.join(chars)


# =================================================================
# ======================== STREAMLIT UI =============================
# =================================================================

st.sidebar.header("Pilih Metode Steganografi")
metode = st.sidebar.selectbox(
    "Metode Steganografi",
    ["LSB", "Histogram Shifting", "PVD"],
)

mode = st.sidebar.radio("Mode", ["Embed (Sisipkan Teks)", "Extract (Ambil Teks)"])

uploaded = st.file_uploader("Upload Gambar", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    st.subheader("Citra Asli")
    st.image(img, use_container_width=True)

    if mode == "Embed (Sisipkan Teks)":
        text = st.text_area("Masukkan Teks Rahasia")

        if st.button("Sisipkan Teks"):
            if metode == "LSB":
                stego = embed_lsb(img, text)
            elif metode == "Histogram Shifting":
                stego = embed_histogram_shifting(img, text)
            else:
                stego = embed_pvd(img, text)

            st.subheader("Citra Hasil Steganografi")
            st.image(stego, use_container_width=True)

            # download button
            buf = io.BytesIO()
            stego.save(buf, format="PNG")
            st.download_button("Download Stego Image", buf.getvalue(), file_name="stego.png")

    else:
        if st.button("Ekstrak Teks"):
            if metode == "LSB":
                result = extract_lsb(img)
            elif metode == "Histogram Shifting":
                result = extract_histogram_shifting(img)
            else:
                result = extract_pvd(img)

            st.success("Teks berhasil diambil:")
            st.code(result)
