import streamlit as st

st.title("Datathon 2024: Mango Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Image uploaded successfully.")