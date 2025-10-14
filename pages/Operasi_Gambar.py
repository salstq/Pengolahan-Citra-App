import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import io

st.set_page_config(page_title="Operasi Pengolahan Gambar", page_icon="🎨", layout="wide")

# -------------------------
# Helper functions (all inside this file per request)
# -------------------------

def pil_to_rgb(img: Image.Image) -> Image.Image:
    """Convert image to RGB (drop alpha if exists)."""
    if img.mode in ("RGBA", "LA"):
        return img.convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def image_to_array(img: Image.Image) -> np.ndarray:
    """Return H x W x 3 uint8 numpy array"""
    return np.array(pil_to_rgb(img), dtype=np.uint8)


def array_to_image(arr: np.ndarray) -> Image.Image:
    """Convert HxWx3 uint8 array back to PIL Image"""
    arr_clipped = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_clipped)


def rgb_table_full_binary(arr: np.ndarray) -> pd.DataFrame:
    """Return full DataFrame of 'r g b' binary strings for the whole image."""
    h, w = arr.shape[:2]
    # Build full DataFrame row by row (can be heavy)
    try:
        rows = []
        for row in arr:
            rows.append([f"{int(p[0]):08b} {int(p[1]):08b} {int(p[2]):08b}" for p in row])
        df = pd.DataFrame(rows)
        return df
    except MemoryError:
        raise


def rgb_table_preview_binary(arr: np.ndarray, max_preview=50) -> pd.DataFrame:
    """Return preview DataFrame (max_preview x max_preview) in binary."""
    h, w = arr.shape[:2]
    ph = min(h, max_preview)
    pw = min(w, max_preview)
    df_preview = pd.DataFrame([
        [f"{int(p[0]):08b} {int(p[1]):08b} {int(p[2]):08b}" for p in row[:pw]]
        for row in arr[:ph]
    ])
    return df_preview


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert to grayscale (kept as 3 identical channels)."""
    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    gray3 = np.stack([gray, gray, gray], axis=-1)
    return np.clip(gray3, 0, 255).astype(np.uint8)


def brightening(arr: np.ndarray, factor: float) -> np.ndarray:
    """Brighten by scaling. factor >= 0."""
    res = arr.astype(float) * factor
    return np.clip(res, 0, 255).astype(np.uint8)


def arithmetic_op(arr1: np.ndarray, arr2: np.ndarray, op: str) -> np.ndarray:
    """Perform element-wise arithmetic. Both arrays must be same shape."""
    a = arr1.astype(float)
    b = arr2.astype(float)
    if op == "+":
        out = a + b
    elif op == "-":
        out = a - b
    elif op == "*":
        # normalized multiply to keep result in 0..255: (a*b)/255
        out = a * b / 255.0
    elif op == "/":
        # avoid division by zero by using small epsilon where b==0
        safe_b = np.where(b == 0, 1e-6, b)
        out = a / safe_b * 255.0
    else:
        out = a
    return np.clip(out, 0, 255).astype(np.uint8)


def boolean_op(arr1: np.ndarray, arr2: np.ndarray, op: str) -> np.ndarray:
    """Perform bitwise boolean operation on uint8 channels."""
    a = arr1.astype(np.uint8)
    b = arr2.astype(np.uint8)
    op_lower = op.lower()
    if op_lower == "and":
        out = np.bitwise_and(a, b)
    elif op_lower == "or":
        out = np.bitwise_or(a, b)
    elif op_lower == "xor":
        out = np.bitwise_xor(a, b)
    else:
        out = a
    return out


def rgb_at_coord(arr: np.ndarray, x: int, y: int):
    """Return (r,g,b) at given x,y or None if out of bounds."""
    h, w = arr.shape[:2]
    if y < 0 or y >= h or x < 0 or x >= w:
        return None
    return tuple(int(v) for v in arr[y, x][:3])


def images_same_size(img1: Image.Image, img2: Image.Image) -> bool:
    return img1.size == img2.size


def pil_image_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# -------------------------
# UI
# -------------------------
st.title("Operasi Pengolahan Gambar 🎨")
st.markdown(
    """
    Halaman ini mendukung operasi:
    - Grayscale (1 gambar)
    - Brightening (1 gambar)
    - Geometri (Rotasi, Flipping) (1 gambar)
    - Aritmatika (+, -, *, /) (2 gambar)  — **tidak meresize otomatis**
    - Boolean (AND, OR, XOR) (2 gambar)  — **tidak meresize otomatis**
    
    **Catatan penting:** Tabel RGB full (dalam bentuk biner) akan menampilkan seluruh pixel dan **bisa sangat berat** untuk gambar besar.
    Jika ukuran dua gambar berbeda saat memilih operasi 2-gambar, proses akan diblokir; upload ulang gambar dengan dimensi yang sama.
    """
)

st.sidebar.markdown("## Pengaturan")
st.sidebar.write("Pastikan gambar tidak terlalu besar jika ingin melihat tabel RGB full (besar = lambat/makan memori).")

# Operation selector
operation = st.selectbox("Pilih operasi:", [
    "Grayscale (1 gambar)",
    "Brightening (1 gambar)",
    "Geometri (Rotasi / Flipping) (1 gambar)",
    "Aritmatika (+, -, *, /) (2 gambar)",
    "Boolean (AND / OR / XOR) (2 gambar)"
])

# Dynamic params
rotate_deg = 0
flip_mode = "None"
bright_factor = 1.0
arith_op = "+"
bool_op = "AND"

if operation == "Brightening (1 gambar)":
    bright_factor = st.slider("Faktor Brightening (0.0 - 5.0). 1.0 = original", 0.0, 5.0, 1.2, 0.1)
if operation == "Geometri (Rotasi / Flipping) (1 gambar)":
    rotate_deg = st.number_input("Rotasi (derajat, positif = searah jarum jam)", value=0, step=1)
    flip_mode = st.selectbox("Flip:", ["None", "Horizontal", "Vertical"])
if operation == "Aritmatika (+, -, *, /) (2 gambar)":
    arith_op = st.selectbox("Operator aritmatika", ["+", "-", "*", "/"])
if operation == "Boolean (AND / OR / XOR) (2 gambar)":
    bool_op = st.selectbox("Operator boolean", ["AND", "OR", "XOR"])

st.markdown("---")

# Upload area
st.header("Upload Gambar")
if "2 gambar" in operation or "Aritmatika" in operation or "Boolean" in operation:
    uploaded_file1 = st.file_uploader("Upload Gambar 1 (png/jpg/jpeg)", type=["png", "jpg", "jpeg"], key="g1")
    uploaded_file2 = st.file_uploader("Upload Gambar 2 (png/jpg/jpeg)", type=["png", "jpg", "jpeg"], key="g2")
else:
    uploaded_file1 = st.file_uploader("Upload Gambar (png/jpg/jpeg)", type=["png", "jpg", "jpeg"], key="g_single")
    uploaded_file2 = None

# check ready to process
can_process = False
if operation in ["Aritmatika (+, -, *, /) (2 gambar)", "Boolean (AND / OR / XOR) (2 gambar)"]:
    if uploaded_file1 and uploaded_file2:
        can_process = True
else:
    if uploaded_file1:
        can_process = True

if not can_process:
    st.info("Upload gambar sesuai kebutuhan operasi untuk mengaktifkan tombol Proses.")
else:
    # Load images (PIL)
    img1 = Image.open(uploaded_file1) if uploaded_file1 else None
    img1 = pil_to_rgb(img1) if img1 else None
    img2 = Image.open(uploaded_file2) if uploaded_file2 else None
    img2 = pil_to_rgb(img2) if img2 else None

    # Convert to numpy arrays (for later)
    arr1 = image_to_array(img1) if img1 is not None else None
    arr2 = image_to_array(img2) if img2 is not None else None

    # Helper to render a PIL image at its original width (no scaling)
    def st_image_original(pil_img, caption=None):
        w, h = pil_img.size
        # streamlit's width parameter sets the displayed width in pixels
        st.image(pil_img, caption=caption, use_column_width=False, width=w)

    # If user uploaded but hasn't pressed proses, still show immediate preview and tables
    # We'll set result as same as input initially (so grid remains consistent)
    result_img = None
    result_arr = None

    # Default behavior: if 2 images are provided and operation is 2-image, we won't auto-process arithmetic/boolean
    # until user clicks Proses. But per requirement, we should render grid immediately. For 1-image ops, show input and output=copy.

    if uploaded_file2 is None:
        # Single-image preview: show 2x2 grid (Input | Output) and below their RGB tables
        # Initially output is copy of input (until user clicks Proses)
        result_img = img1.copy()
        result_arr = arr1.copy()

        # Top row: images
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Gambar Asli")
            st_image_original(img1, caption=f"Asli — size: {img1.size}")
        with c2:
            st.subheader("Gambar Output (sementara)")
            st_image_original(result_img, caption=f"Output — size: {result_img.size}")

        # Bottom row: RGB tables (binary), render full immediately (may be heavy)
        st.subheader("Tabel RGB - (Biner, FULL)")
        d1, d2 = st.columns(2)
        with d1:
            st.caption("RGB Asli (biner)")
            try:
                df1 = rgb_table_full_binary(arr1)
                st.dataframe(df1, use_container_width=True)
            except MemoryError:
                st.error("Gagal menampilkan tabel RGB penuh karena memori tidak cukup. Gunakan gambar lebih kecil.")
        with d2:
            st.caption("RGB Output (biner)")
            try:
                df_out = rgb_table_full_binary(result_arr)
                st.dataframe(df_out, use_container_width=True)
            except MemoryError:
                st.error("Gagal menampilkan tabel RGB penuh karena memori tidak cukup. Gunakan gambar lebih kecil.")

    else:
        # Two-image preview: top row Input1 | Input2 | Output (initially copy of input1)
        result_img = img1.copy()
        result_arr = arr1.copy()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Input 1")
            st_image_original(img1, caption=f"Input1 — size: {img1.size}")
        with c2:
            st.subheader("Input 2")
            st_image_original(img2, caption=f"Input2 — size: {img2.size}")
        with c3:
            st.subheader("Output (sementara)")
            st_image_original(result_img, caption=f"Output — size: {result_img.size}")

        # Bottom row: RGB tables (biner) for each column
        st.subheader("Tabel RGB - (Biner, FULL)")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.caption("RGB Input 1 (biner)")
            try:
                df1 = rgb_table_full_binary(arr1)
                st.dataframe(df1, use_container_width=True)
            except MemoryError:
                st.error("Gagal menampilkan tabel RGB penuh Input 1 karena memori tidak cukup.")
        with d2:
            st.caption("RGB Input 2 (biner)")
            try:
                df2 = rgb_table_full_binary(arr2)
                st.dataframe(df2, use_container_width=True)
            except MemoryError:
                st.error("Gagal menampilkan tabel RGB penuh Input 2 karena memori tidak cukup.")
        with d3:
            st.caption("RGB Output (biner)")
            try:
                df_out = rgb_table_full_binary(result_arr)
                st.dataframe(df_out, use_container_width=True)
            except MemoryError:
                st.error("Gagal menampilkan tabel RGB penuh Output karena memori tidak cukup.")

    # Process button (user-triggered) — performs chosen operation and refreshes output in place
    if st.button("Proses"):
        st.write("---")
        st.header("Hasil Proses")

        # If operation needs two images, ensure sizes match (per user's choice: resize=no)
        if operation in ["Aritmatika (+, -, *, /) (2 gambar)", "Boolean (AND / OR / XOR) (2 gambar)"]:
            if not images_same_size(img1, img2):
                st.error("Ukuran gambar berbeda. Untuk operasi 2-gambar resize otomatis DINONAKTIFKAN. Silakan upload kedua gambar dengan ukuran sama.")
            else:
                try:
                    if operation == "Aritmatika (+, -, *, /) (2 gambar)":
                        result_arr = arithmetic_op(arr1, arr2, arith_op)
                        result_img = array_to_image(result_arr)
                        st.subheader(f"Gambar Hasil — Aritmatika (operator: {arith_op})")
                    elif operation == "Boolean (AND / OR / XOR) (2 gambar)":
                        result_arr = boolean_op(arr1, arr2, bool_op)
                        result_img = array_to_image(result_arr)
                        st.subheader(f"Gambar Hasil — Boolean (operator: {bool_op})")

                    # Display top row images at original sizes
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.caption("Input 1")
                        st_image_original(img1, caption=f"Input1 — size: {img1.size}")
                    with c2:
                        st.caption("Input 2")
                        st_image_original(img2, caption=f"Input2 — size: {img2.size}")
                    with c3:
                        st.caption("Hasil")
                        st_image_original(result_img, caption=f"Hasil — size: {result_img.size}")

                    # Bottom: RGB tables binary
                    st.subheader("Tabel RGB - Hasil (Biner, FULL)")
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        st.caption("RGB Input1 (biner)")
                        df1 = rgb_table_full_binary(arr1)
                        st.dataframe(df1, use_container_width=True)
                    with d2:
                        st.caption("RGB Input2 (biner)")
                        df2 = rgb_table_full_binary(arr2)
                        st.dataframe(df2, use_container_width=True)
                    with d3:
                        st.caption("RGB Hasil (biner)")
                        dfres = rgb_table_full_binary(result_arr)
                        st.dataframe(dfres, use_container_width=True)

                except MemoryError:
                    st.error("Operasi gagal karena memori tidak cukup saat membangun tabel full. Coba gunakan gambar lebih kecil atau batasi tampilan tabel.")
                except Exception as e:
                    st.error(f"Terjadi error saat memproses: {e}")
        else:
            # Single-image operations
            try:
                if operation == "Grayscale (1 gambar)":
                    result_arr = to_grayscale(arr1)
                    result_img = array_to_image(result_arr)
                    st.subheader("Gambar Hasil — Grayscale")
                elif operation == "Brightening (1 gambar)":
                    result_arr = brightening(arr1, bright_factor)
                    result_img = array_to_image(result_arr)
                    st.subheader(f"Gambar Hasil — Brightening (factor={bright_factor})")
                elif operation == "Geometri (Rotasi / Flipping) (1 gambar)":
                    tmp = img1
                    # PIL rotates counter-clockwise for positive angle; user requested positive = clockwise -> negate
                    if rotate_deg != 0:
                        tmp = tmp.rotate(-rotate_deg, expand=True)
                    if flip_mode == "Horizontal":
                        tmp = ImageOps.mirror(tmp)
                    elif flip_mode == "Vertical":
                        tmp = ImageOps.flip(tmp)
                    result_img = pil_to_rgb(tmp)
                    result_arr = image_to_array(result_img)
                    st.subheader(f"Gambar Hasil — Geometri (rot={rotate_deg}, flip={flip_mode})")
                else:
                    st.warning("Operasi belum dikenali untuk 1-gambar. Output = copy input.")

                # Display 2x2 grid: Input | Output
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Gambar Asli")
                    st_image_original(img1, caption=f"Asli — size: {img1.size}")
                with c2:
                    st.caption("Gambar Hasil")
                    st_image_original(result_img, caption=f"Hasil — size: {result_img.size}")

                # Bottom row: RGB tables binary
                st.subheader("Tabel RGB - (Biner, FULL)")
                d1, d2 = st.columns(2)
                with d1:
                    st.caption("RGB Asli (biner)")
                    df1 = rgb_table_full_binary(arr1)
                    st.dataframe(df1, use_container_width=True)
                with d2:
                    st.caption("RGB Hasil (biner)")
                    dfres = rgb_table_full_binary(result_arr)
                    st.dataframe(dfres, use_container_width=True)

            except MemoryError:
                st.error("Operasi gagal karena memori tidak cukup saat membangun tabel full. Coba gunakan gambar lebih kecil atau batasi tampilan tabel.")
            except Exception as e:
                st.error(f"Terjadi error saat memproses: {e}")

        # provide download button for result image if exists
        if result_img is not None:
            st.markdown("---")
            st.subheader("Download Hasil")
            img_bytes = pil_image_to_bytes(result_img, fmt="PNG")
            st.download_button("Download Hasil (PNG)", data=img_bytes, file_name="result.png", mime="image/png")

            # Optionally download CSV for full RGB table (result) — warn about size
            if result_arr is not None:
                try:
                    dfres = rgb_table_full_binary(result_arr)
                    csv_bytes = dfres.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV (Hasil)", data=csv_bytes, file_name="rgb_result_biner.csv", mime="text/csv")
                except MemoryError:
                    st.error("Gagal membuat CSV karena memori tidak cukup.")
                except Exception as e:
                    st.error(f"Gagal membuat CSV: {e}")

# -------------------------
# Requirements hint (sidebar)
# -------------------------
st.sidebar.markdown("---")
st.sidebar.write("Requirements (example):")
st.sidebar.code("""streamlit
pillow
numpy
pandas
""")
st.sidebar.write("Notes:")
st.sidebar.write("- Tabel RGB full akan menampilkan seluruh pixel (dalam format biner 8-bit) dan dapat memakan banyak memori dan waktu render di browser.")
st.sidebar.write("- Untuk operasi dua gambar, aplikasi **TIDAK** meresize otomatis. Upload gambar dengan ukuran yang sama.")
