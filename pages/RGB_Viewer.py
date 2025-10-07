import streamlit as st
import pandas as pd
import numpy as np
from PIL import image
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("RGB Viewer")
uploaded_file = st.file_uploader("Unggah Gambar:", type = [png, jpg, jepg])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    st.write(f"Ukuran gambar: {width} x {height} (width, height)")

    pixels = [[f'{r}, {g}, {b}' for r,g,b in row] for row in img_array[:,:,:3]]
    df_full = pd.DataFrame(pixels)
    st.write('Tabel seluruh pixel RGB:')
    st.dataframe(df_full, user_container_width = True)

    st.write("Klik pada area gambar untuk melihat ukuran pixel RGB:")
    coords = streamlit_image_coordinates(image)

    if coords is not None:
        x, y = coords["x"], coords["y"]
        st.success(f"Koordinat gambar yang dipilih: x = {x}, y = {y}")

        if y < image_array.shape[0] and x < image_array.shape[1]:
            r, g, b = image_array[y, x][:3]
            st.write(f"Ukuran pixel RGB pada koordinat tersebut adalah ({r}, {g}, {b})")
            st.markdown(f"<div style = 'width : 100px; height : 100px; background-color : rgb({r},{g},{b});border : 1px solid #000'></div>",
                        unsafe_allow_html=True)
            
             
    
