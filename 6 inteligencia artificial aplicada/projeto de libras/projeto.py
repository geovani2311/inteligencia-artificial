import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Função para processar vídeo e gerar landmarks com conexões
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    out_frames = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                frame_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                landmarks_list.append(frame_landmarks)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            out_frames.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    cap.release()

    # Salvar vídeo processado temporário
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))

    for f in out_frames:
        out.write(f)
    out.release()

    return landmarks_list, temp_out.name

# Função para calcular métricas adicionais
def calculate_metrics(landmarks):
    # Exemplo simples: distância entre ombros e altura média
    distances = []
    heights = []
    for frame in landmarks:
        frame = np.array(frame)
        left_shoulder = frame[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = frame[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        distance = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
        distances.append(distance)

        # Altura média do corpo (y médio dos pontos)
        heights.append(np.mean(frame[:,1]))

    return np.mean(distances), np.mean(heights)

# Função para calcular diferença média do corpo inteiro
def compare_landmarks(landmarks1, landmarks2):
    min_frames = min(len(landmarks1), len(landmarks2))
    diffs = []

    for i in range(min_frames):
        arr1 = np.array(landmarks1[i])
        arr2 = np.array(landmarks2[i])
        diff = np.linalg.norm(arr1 - arr2)
        diffs.append(diff)

    return np.mean(diffs)

# App Streamlit
st.title("Comparador de Movimentos com MediaPipe")

video_user = st.file_uploader("Envie seu vídeo", type=["mp4", "mov", "avi"])
video_pro = st.file_uploader("Envie o vídeo do profissional", type=["mp4", "mov", "avi"])

if video_user and video_pro:
    st.write("Processando vídeos automaticamente...")

    # Salvar vídeos temporários
    tfile_user = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile_user.write(video_user.read())
    tfile_user.close()

    tfile_pro = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile_pro.write(video_pro.read())
    tfile_pro.close()

    user_landmarks, user_out = process_video(tfile_user.name)
    pro_landmarks, pro_out = process_video(tfile_pro.name)

    # Exibir vídeos lado a lado
    st.write("### Visualização dos Vídeos com Landmarks")
    col1, col2 = st.columns(2)

    with col1:
        video_file_user = open(user_out, 'rb')
        st.video(video_file_user.read())
        video_file_user.close()

    with col2:
        video_file_pro = open(pro_out, 'rb')
        st.video(video_file_pro.read())
        video_file_pro.close()

    # Calcular diferença média do corpo inteiro
    score = compare_landmarks(user_landmarks, pro_landmarks)
    st.write(f"### Diferença média entre movimentos (corpo inteiro): {score:.4f}")

    # Calcular métricas adicionais
    user_shoulder_distance, user_height = calculate_metrics(user_landmarks)
    pro_shoulder_distance, pro_height = calculate_metrics(pro_landmarks)

    st.write(f"### Métricas adicionais")
    st.write(f"Distância média entre ombros - Usuário: {user_shoulder_distance:.4f}, Profissional: {pro_shoulder_distance:.4f}")
    st.write(f"Altura média do corpo - Usuário: {user_height:.4f}, Profissional: {pro_height:.4f}")

    # Remover arquivos temporários
    os.remove(user_out)
    os.remove(pro_out)
    os.remove(tfile_user.name)
    os.remove(tfile_pro.name)