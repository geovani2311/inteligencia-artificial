import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf  # Importe o TensorFlow aqui

# Título da aplicação
st.title("Detecção de COVID-19 em Tomografias")

# Componente para fazer o upload da imagem
uploaded_file = st.file_uploader("Faça o upload de uma imagem de tomografia (PNG)", type=["png"])

if uploaded_file is not None:
    model = tf.keras.models.load_model('modelo_covid.h5') # Carregar o modelo aqui
    # Aqui você colocará o código para processar a imagem e fazer a previsão
    st.image(uploaded_file, caption="Imagem de Tomografia Carregada", use_container_width=True)
    st.write("Aguarde a análise...")

# Abrir a imagem carregada usando Pillow
    img = Image.open(uploaded_file).convert('RGB')

    # Redimensionar a imagem para o tamanho esperado pelo modelo
    img_redimensionada = img.resize((150, 150))

    # Converter a imagem redimensionada para um array numpy
    img_array = np.array(img_redimensionada)

    # Normalizar os valores dos pixels
    img_normalizada = img_array.astype('float32') / 255.0

    # Expandir as dimensões para que a imagem tenha o formato esperado pelo modelo (batch_size, altura, largura, canais)
    img_expandida = np.expand_dims(img_normalizada, axis=0)

    # Fazer a previsão usando o modelo carregado
    prediction = model.predict(img_expandida)

    # Interpretar a previsão
    if prediction[0][0] > 0.5:
        st.error("Resultado: A imagem provavelmente indica presença de COVID-19.")
    else:
        st.success("Resultado: A imagem provavelmente NÃO indica presença de COVID-19.")

    st.write(f"Probabilidade da imagem indicar COVID-19: {prediction[0][0]:.4f}")