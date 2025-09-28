import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# Inicializa as soluções do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calcula o ângulo entre três pontos (landmarks).
    Útil para analisar a biomecânica do movimento.
    """
    a = np.array(a)  # Primeiro ponto (ex: quadril)
    b = np.array(b)  # Ponto do meio (ex: joelho)
    c = np.array(c)  # Ponto final (ex: tornozelo)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def process_video(video_file, column_name):
    """
    Processa um arquivo de vídeo para detectar a pose, desenhar landmarks,
    calcular ângulos e exibir o resultado no Streamlit.
    """
    st.header(f"Análise do Vídeo: {column_name}")
    
    # Usa um arquivo temporário para que o OpenCV possa abri-lo
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Placeholder para exibir o vídeo processado
    frame_placeholder = st.empty()
    
    # Placeholders para as métricas
    metrics_placeholder = st.empty()
    
    max_kick_leg_angle = 0
    min_support_leg_angle = 180
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write(f"Fim da análise do vídeo {column_name}.")
                break

            # Redimensiona o frame para um tamanho padrão para exibição
            frame = cv2.resize(frame, (640, 480))
            
            # Converte a imagem de BGR para RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Faz a detecção da pose
            results = pose.process(image)

            # Converte a imagem de volta para BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extrai os landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # --- Lógica para determinar perna de chute e apoio ---
                # Considera a perna direita como a de chute por padrão
                # (Pode ser melhorado com detecção de bola)
                hip_kick = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_kick = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_kick = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                hip_support = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_support = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle_support = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calcula os ângulos
                kick_leg_angle = calculate_angle(hip_kick, knee_kick, ankle_kick)
                support_leg_angle = calculate_angle(hip_support, knee_support, ankle_support)
                
                # Atualiza as métricas
                if kick_leg_angle > max_kick_leg_angle:
                    max_kick_leg_angle = kick_leg_angle
                
                if support_leg_angle < min_support_leg_angle:
                    min_support_leg_angle = support_leg_angle

                # Exibe o ângulo no vídeo
                cv2.putText(image, f"Chute: {int(kick_leg_angle)}", 
                            tuple(np.multiply(knee_kick, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f"Apoio: {int(support_leg_angle)}", 
                            tuple(np.multiply(knee_support, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            except:
                pass # Se nenhum landmark for detectado

            # Desenha os landmarks na imagem
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                     )               
            
            # Exibe o frame processado
            frame_placeholder.image(image, channels="BGR", use_container_width=True)
            
            # Exibe as métricas atualizadas
            with metrics_placeholder.container():
                st.markdown("---")
                st.markdown(f"#### Métricas de Desempenho ({column_name})")
                st.info(f"Máxima Extensão da Perna de Chute: **{max_kick_leg_angle:.2f}°**")
                st.warning(f"Mínima Flexão da Perna de Apoio (Estabilidade): **{min_support_leg_angle:.2f}°**")
            
            # Adiciona um pequeno delay para a visualização
            time.sleep(0.03)

    cap.release()
    return max_kick_leg_angle, min_support_leg_angle

# --- Interface do Streamlit ---
st.set_page_config(layout="wide", page_title="Analisador de Chute de Futebol")

st.title("⚽ Analisador de Chute com Visão Computacional")
st.markdown("""
Esta aplicação utiliza o **MediaPipe** para analisar a biomecânica de um chute de futebol.
- **Faça o upload de dois vídeos**: um de um chute amador e outro de um profissional.
- A aplicação irá **detectar os pontos corporais** (landmarks) em cada frame.
- **Calculará ângulos importantes** para avaliar a técnica do chute.
- **Exibirá os resultados lado a lado** para uma comparação clara.
""")
st.markdown("---")

# Colunas para os uploads
col1_upload, col2_upload = st.columns(2)

with col1_upload:
    amateur_video = st.file_uploader("Carregue o Vídeo Amador", type=["mp4", "mov", "avi", "asf", "m4v"])

with col2_upload:
    professional_video = st.file_uploader("Carregue o Vídeo Profissional", type=["mp4", "mov", "avi", "asf", "m4v"])

# Colunas para exibir os vídeos e as análises
col1_display, col2_display = st.columns(2)

results_data = {}

if amateur_video is not None and professional_video is not None:
    if st.button("▶️ Iniciar Análise Comparativa"):
        with col1_display:
            amateur_kick_angle, amateur_support_angle = process_video(amateur_video, "Amador")
            results_data["Amador"] = {"kick": amateur_kick_angle, "support": amateur_support_angle}
        
        with col2_display:
            prof_kick_angle, prof_support_angle = process_video(professional_video, "Profissional")
            results_data["Profissional"] = {"kick": prof_kick_angle, "support": prof_support_angle}
        
        # --- Resumo Comparativo ---
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("🏁 Resumo Comparativo Final")
        
        col1_res, col2_res, col3_res = st.columns(3)
        
        with col1_res:
            st.metric(
                label="Máx. Extensão do Chute (Amador)",
                value=f"{results_data['Amador']['kick']:.2f}°"
            )
            st.metric(
                label="Máx. Extensão do Chute (Profissional)",
                value=f"{results_data['Profissional']['kick']:.2f}°",
                delta=f"{results_data['Profissional']['kick'] - results_data['Amador']['kick']:.2f}°"
            )
        
        with col2_res:
            st.metric(
                label="Estabilidade do Apoio (Amador)",
                value=f"{results_data['Amador']['support']:.2f}°"
            )
            st.metric(
                label="Estabilidade do Apoio (Profissional)",
                value=f"{results_data['Profissional']['support']:.2f}°",
                delta=f"{results_data['Profissional']['support'] - results_data['Amador']['support']:.2f}° (menor é mais estável)"
            )
        
        with col3_res:
            st.subheader("💡 Insights")
            if results_data['Profissional']['kick'] > results_data['Amador']['kick']:
                st.success("**Potência Superior:** O profissional alcança uma maior extensão da perna, o que geralmente se traduz em mais potência no chute.")
            else:
                st.warning("**Potência a ser melhorada:** O chute amador teve uma extensão similar ou maior, mas isso pode indicar falta de controle.")
            
            if results_data['Profissional']['support'] < results_data['Amador']['support']:
                st.success("**Base Sólida:** A perna de apoio do profissional se mantém mais flexionada e estável, essencial para o equilíbrio e precisão.")
            else:
                st.warning("**Equilíbrio a ser melhorado:** O amador tende a ter a perna de apoio mais reta, o que pode comprometer o equilíbrio.")

elif amateur_video is not None or professional_video is not None:
    st.info("Por favor, carregue os dois vídeos para iniciar a análise comparativa.")
else:
    st.info("Aguardando o upload dos vídeos...")