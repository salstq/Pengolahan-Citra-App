# 2_🎨_Operasi_Pengolahan_Gambar.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates
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

def rgb_table_full(arr: np.ndarray) -> pd.DataFrame:
    """Return full DataFrame of '(r,g,b)' strings for the whole image."""
    h, w = arr.shape[:2]
    # Build full DataFrame row by row (can be heavy)
    df = pd.DataFrame([["({},{},{})".format(int(p[0]), int(p[1]), int(p[2])) for p in row] for row in arr])
    return df

def rgb_table_preview(arr: np.ndarray, max_preview=50) -> pd.DataFrame:
    """Return preview DataFrame (max_preview x max_preview)."""
    h, w = arr.shape[:2]
    ph = min(h, max_preview)
    pw = min(w, max_preview)
    df_preview = pd.DataFrame([["({},{},{})".format(int(p[0]), int(p[1]), int(p[2])) for p in row[:pw]] for row in arr[:ph]])
    return df_preview

def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert to grayscale (kept as 3 identical channels)."""
    gray = (0.2989 * arr[...,0] + 0.5870 * arr[...,1] + 0.1140 * arr[...,2])
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

def small_neighborhood_df(arr: np.ndarray, x: int, y: int, radius=2):
    """Return DataFrame of small neighborhood around (x,y)."""
    h, w = arr.shape[:2]
    ys = range(max(0, y-radius), min(h, y+radius+1))
    xs = range(max(0, x-radius), min(w, x+radius+1))
    data = []
    for yy in ys:
        row = []
        for xx in xs:
            row.append("({},{},{})".format(*tuple(arr[yy, xx][:3])))
        data.append(row)
    df = pd.DataFrame(data, index=[f"y={yy}" for yy in ys], columns=[f"x={xx}" for xx in xs])
    return df

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
    
    **Catatan penting:** Tabel RGB full akan menampilkan seluruh pixel dan **bisa sangat berat** untuk gambar besar.
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
    uploaded_file1 = st.file_uploader("Upload Gambar 1 (png/jpg/jpeg)", type=["png","jpg","jpeg"], key="g1")
    uploaded_file2 = st.file_uploader("Upload Gambar 2 (png/jpg/jpeg)", type=["png","jpg","jpeg"], key="g2")
else:
    uploaded_file1 = st.file_uploader("Upload Gambar (png/jpg/jpeg)", type=["png","jpg","jpeg"], key="g_single")
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

    # Show input images
    st.subheader("Gambar Input")
    if img2 is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.image(img1, caption=f"Input 1 — size: {img1.size}", use_column_width=True)
        with c2:
            st.image(img2, caption=f"Input 2 — size: {img2.size}", use_column_width=True)
    else:
        st.image(img1, caption=f"Input — size: {img1.size}", use_column_width=True)

    # Clickable coordinate for single-image ops
    if operation in ["Grayscale (1 gambar)", "Brightening (1 gambar)", "Geometri (Rotasi / Flipping) (1 gambar)"]:
        st.markdown("**Klik pada gambar untuk melihat koordinat & RGB**")
        coords = streamlit_image_coordinates(img1, key="op_click")
        if coords is not None:
            x, y = int(coords["x"]), int(coords["y"])
            st.success(f"Koordinat yang diklik: (x={x}, y={y})")
            rgb_val = rgb_at_coord(arr1, x, y)
            if rgb_val is not None:
                st.write(f"RGB = {rgb_val}")
                st.markdown(f"<div style='width:60px;height:60px;background-color:rgb{rgb_val};border:1px solid #000'></div>", unsafe_allow_html=True)
                st.write("Area sekitar (radius=2):")
                st.dataframe(small_neighborhood_df(arr1, x, y, radius=2), use_container_width=True)

    # Process button
    if st.button("Proses"):
        st.write("---")
        st.header("Hasil Proses")
        result_arr = None
        result_img = None

        # If operation needs two images, ensure sizes match (per user's choice: resize=no)
        if operation in ["Aritmatika (+, -, *, /) (2 gambar)", "Boolean (AND / OR / XOR) (2 gambar)"]:
            if not images_same_size(img1, img2):
                st.error("Ukuran gambar berbeda. Untuk operasi 2-gambar resize otomatis DINONAKTIFKAN. Silakan upload kedua gambar dengan ukuran sama.")
                st.stop()

        # Perform selected operation
        try:
            if operation == "Grayscale (1 gambar)":
                result_arr = to_grayscale(arr1)
                result_img = array_to_image(result_arr)
                st.subheader("Gambar Hasil — Grayscale")
                st.image(result_img, use_column_width=True)

                st.subheader("Tabel RGB - Sebelum (FULL)")
                df_before = rgb_table_full(arr1)
                st.dataframe(df_before, use_container_width=True)

                st.subheader("Tabel RGB - Sesudah (FULL)")
                df_after = rgb_table_full(result_arr)
                st.dataframe(df_after, use_container_width=True)

            elif operation == "Brightening (1 gambar)":
                result_arr = brightening(arr1, bright_factor)
                result_img = array_to_image(result_arr)
                st.subheader(f"Gambar Hasil — Brightening (factor={bright_factor})")
                st.image(result_img, use_column_width=True)

                st.subheader("Tabel RGB - Sebelum (FULL)")
                df_before = rgb_table_full(arr1)
                st.dataframe(df_before, use_container_width=True)

                st.subheader("Tabel RGB - Sesudah (FULL)")
                df_after = rgb_table_full(result_arr)
                st.dataframe(df_after, use_container_width=True)

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
                st.image(result_img, use_column_width=True)

                st.subheader("Tabel RGB - Sebelum (FULL)")
                df_before = rgb_table_full(arr1)
                st.dataframe(df_before, use_container_width=True)

                st.subheader("Tabel RGB - Sesudah (FULL)")
                df_after = rgb_table_full(result_arr)
                st.dataframe(df_after, use_container_width=True)

            elif operation == "Aritmatika (+, -, *, /) (2 gambar)":
                result_arr = arithmetic_op(arr1, arr2, arith_op)
                result_img = array_to_image(result_arr)
                st.subheader(f"Gambar Hasil — Aritmatika (operator: {arith_op})")
                c1, c2, c3 = st.columns(3)
                with c1: st.image(img1, caption="Input 1", use_column_width=True)
                with c2: st.image(img2, caption="Input 2", use_column_width=True)
                with c3: st.image(result_img, caption="Hasil", use_column_width=True)

                st.subheader("Tabel RGB - Input 1 (FULL)")
                df1 = rgb_table_full(arr1)
                st.dataframe(df1, use_container_width=True)

                st.subheader("Tabel RGB - Input 2 (FULL)")
                df2 = rgb_table_full(arr2)
                st.dataframe(df2, use_container_width=True)

                st.subheader("Tabel RGB - Hasil (FULL)")
                dfres = rgb_table_full(result_arr)
                st.dataframe(dfres, use_container_width=True)

            elif operation == "Boolean (AND / OR / XOR) (2 gambar)":
                result_arr = boolean_op(arr1, arr2, bool_op)
                result_img = array_to_image(result_arr)
                st.subheader(f"Gambar Hasil — Boolean (operator: {bool_op})")
                c1, c2, c3 = st.columns(3)
                with c1: st.image(img1, caption="Input 1", use_column_width=True)
                with c2: st.image(img2, caption="Input 2", use_column_width=True)
                with c3: st.image(result_img, caption="Hasil", use_column_width=True)

                st.subheader("Tabel RGB - Input 1 (FULL)")
                df1 = rgb_table_full(arr1)
                st.dataframe(df1, use_container_width=True)

                st.subheader("Tabel RGB - Input 2 (FULL)")
                df2 = rgb_table_full(arr2)
                st.dataframe(df2, use_container_width=True)

                st.subheader("Tabel RGB - Hasil (FULL)")
                dfres = rgb_table_full(result_arr)
                st.dataframe(dfres, use_container_width=True)

            else:
                st.warning("Operasi belum dikenali.")
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
                if st.checkbox("Simpan Tabel RGB hasil ke CSV? (mungkin besar)", key="download_csv"):
                    try:
                        dfres = rgb_table_full(result_arr)
                        csv_bytes = dfres.to_csv(index=False).encode("utf-8")
                        st.download_button("Download CSV (Hasil)", data=csv_bytes, file_name="rgb_result.csv", mime="text/csv")
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
streamlit-image-coordinates
""")
st.sidebar.write("Notes:")
st.sidebar.write("- Tabel RGB full akan menampilkan seluruh pixel dan dapat memakan banyak memori dan waktu render di browser.")
st.sidebar.write("- Untuk operasi dua gambar, aplikasi **TIDAK** meresize otomatis. Upload gambar dengan ukuran yang sama.")
