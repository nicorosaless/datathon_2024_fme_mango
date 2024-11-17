import streamlit as st
import numpy as np
import os
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image as keras_img
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image

# Cargar MobileNet para la extracción de características
mobilenet = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Cargar el modelo entrenado para atributos (ajustar la ruta según tu entorno)
# model_path = "embedding_only_model.h5"  # Ruta de tu modelo
# attribute_model = load_model(model_path)

# Preprocesar imagen y extraer embeddings usando MobileNet
def process_image(image):
    """
    Preprocesa una imagen y devuelve los embeddings utilizando MobileNet.
    """
    img_data = keras_img.img_to_array(image)  # Convertir la imagen a un array
    img_data = keras_img.smart_resize(img_data, (224, 224))  # Redimensionar a 224x224
    img_data = preprocess_input(img_data)  # Normalizar la imagen para MobileNet
    img_data = np.expand_dims(img_data, axis=0)  # Expandir dimensiones
    embeddings = mobilenet.predict(img_data)  # Extraer los embeddings
    return embeddings

# Interfaz de usuario de Streamlit
st.title('Predicción de Atributos de Imagen')

st.write('Sube una imagen para predecir los atributos de la prenda.')

# Subir archivo de imagen
uploaded_image = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagen cargada", use_container_width=True)
    
    # Obtener los embeddings de la imagen
    st.write("Extrayendo embeddings...")
    embeddings = process_image(image)
    
    # Mostrar los embeddings en Streamlit
    st.write("Embeddings extraídos:")
    st.write(embeddings)
    
    # Realizar la predicción de atributos si tienes un modelo entrenado
    # st.write("Prediciendo atributos...")
    # predicted_attributes = predict_attributes(image)
    
    # Mostrar las predicciones (si tienes el modelo cargado)
    # st.write("Atributos predichos:")
    # st.write(predicted_attributes)
