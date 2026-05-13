import streamlit as st
from PIL import Image
from infer import predict

st.title("Handwriting OCR - DenseNet121")

mode = st.radio("Chọn input:", ["Upload ảnh", "Vẽ trực tiếp"])

# ===== Upload =====
if mode == "Upload ảnh":
    file = st.file_uploader("Chọn ảnh", type=["png", "jpg"])

    if file:
        img = Image.open(file)
        st.image(img)

        if st.button("Nhận dạng"):
            result = predict(img)
            st.success(result)

# ===== Draw =====
else:
    from streamlit_drawable_canvas import st_canvas

    canvas = st_canvas(
        fill_color="white",
        stroke_width=1,
        stroke_color="black",
        background_color="white",
        height=120,
        width=1000,
    )

    if canvas.image_data is not None:
        img = Image.fromarray(canvas.image_data.astype("uint8"))

        if st.button("Nhận dạng"):
            result = predict(img)
            st.success(result)