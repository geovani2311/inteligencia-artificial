import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf  # Importe o TensorFlow aqui para carregar o modelo depois

# Título da aplicação
st.title("Detecção de COVID-19 em Tomografias")

# Componente para fazer o upload da imagem
uploaded_file = st.file_uploader("Faça o upload de uma imagem de tomografia (PNG)", type=["png"])

if uploaded_file is not None:
    # Aqui você colocará o código para processar a imagem e fazer a previsão
    st.image(uploaded_file, caption="Imagem de Tomografia Carregada", use_column_width=True)
    st.write("Aguarde a análise...")