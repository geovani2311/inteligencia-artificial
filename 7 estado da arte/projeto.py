# -*- coding: utf-8 -*-
# Importa as bibliotecas necessárias
import streamlit as st  # Para criar a interface web
import cv2  # OpenCV, para processamento de imagem e vídeo
import mediapipe as mp  # Para detecção de pose e landmarks corporais
import numpy as np  # Para cálculos numéricos, especialmente com arrays
import tempfile  # Para criar arquivos temporários
import time  # Para adicionar pausas (delay)

# Inicializa as soluções do MediaPipe que serão usadas
mp_drawing = mp.solutions.drawing_utils  # Ferramenta para desenhar os landmarks e conexões
mp_pose = mp.solutions.pose  # Modelo de detecção de pose

def calculate_angle(a, b, c):
    """
    Calcula o ângulo entre três pontos (landmarks) usando trigonometria.
    É a base para a análise biomecânica do movimento.
    
    Args:
        a (list): Coordenadas do primeiro ponto (ex: quadril).
        b (list): Coordenadas do ponto do meio, onde o ângulo é calculado (ex: joelho).
        c (list): Coordenadas do ponto final (ex: tornozelo).
        
    Returns:
        float: O ângulo calculado em graus.
    """
    # Converte os pontos para arrays NumPy para facilitar os cálculos
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Usa a função arctan2 para calcular o ângulo entre os vetores (b-a) e (b-c)
    # O resultado é em radianos
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    
    # Converte o ângulo de radianos para graus
    angle = np.abs(radians * 180.0 / np.pi)

    # Garante que o ângulo esteja entre 0 e 180 graus
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def process_video(video_file, column_name):
    """
    Processa um arquivo de vídeo para detectar a pose, desenhar landmarks,
    calcular ângulos e exibir o resultado em tempo real no Streamlit.
    
    Args:
        video_file (UploadedFile): O arquivo de vídeo carregado pelo usuário no Streamlit.
        column_name (str): O nome da coluna onde o vídeo será exibido ("Amador" ou "Profissional").
        
    Returns:
        tuple: Uma tupla contendo o ângulo máximo do chute e o ângulo mínimo de apoio.
    """
    st.header(f"Análise do Vídeo: {column_name}")
    
    # O Streamlit lê o arquivo em memória. Para o OpenCV abri-lo,
    # é mais seguro salvá-lo como um arquivo temporário no disco.
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    # Abre o vídeo a partir do caminho do arquivo temporário
    cap = cv2.VideoCapture(tfile.name)
    
    # Cria espaços reservados na interface do Streamlit.
    # Estes espaços serão atualizados a cada frame, sem recarregar a página.
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Inicializa as variáveis para armazenar os melhores resultados encontrados
    max_kick_leg_angle = 0   # Queremos a maior extensão (ângulo máximo)
    min_support_leg_angle = 180 # Queremos a maior flexão (ângulo mínimo)
    
    # Inicializa o detector de pose do MediaPipe
    # min_detection_confidence: Confiança mínima para a detecção inicial de uma pessoa.
    # min_tracking_confidence: Confiança mínima para rastrear a pessoa nos frames seguintes.
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Loop para ler o vídeo frame a frame
        while cap.isOpened():
            ret, frame = cap.read()
            # Se 'ret' for falso, significa que o vídeo acabou
            if not ret:
                st.write(f"Fim da análise do vídeo {column_name}.")
                break

            # Redimensiona o frame para um tamanho padrão para garantir consistência
            frame = cv2.resize(frame, (640, 480))
            
            # O OpenCV lê imagens no formato BGR, mas o MediaPipe espera RGB.
            # Fazemos a conversão de cores.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # Trava a imagem para otimizar o processamento

            # Processa a imagem e detecta a pose
            results = pose.process(image)

            # Destrava a imagem para que possamos desenhar nela
            image.flags.writeable = True
            # Converte a imagem de volta para BGR para que o OpenCV possa exibi-la corretamente
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Tenta extrair os landmarks. O 'try-except' evita erros se nenhuma pose for detectada.
            try:
                landmarks = results.pose_landmarks.landmark
                
                # --- Lógica para determinar perna de chute e apoio ---
                # Suposição: a perna direita é a de chute. Isso pode ser melhorado
                # com detecção de bola ou análise de movimento.
                
                # Coleta as coordenadas (x, y) dos pontos de interesse da perna de chute
                hip_kick = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_kick = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_kick = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Coleta as coordenadas (x, y) dos pontos de interesse da perna de apoio
                hip_support = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_support = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle_support = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calcula os ângulos atuais usando a função que criamos
                kick_leg_angle = calculate_angle(hip_kick, knee_kick, ankle_kick)
                support_leg_angle = calculate_angle(hip_support, knee_support, ankle_support)
                
                # Atualiza as métricas se um novo recorde for encontrado
                if kick_leg_angle > max_kick_leg_angle:
                    max_kick_leg_angle = kick_leg_angle
                
                if support_leg_angle < min_support_leg_angle:
                    min_support_leg_angle = support_leg_angle

                # Exibe o valor do ângulo diretamente no vídeo, perto do joelho
                # As coordenadas dos landmarks são normalizadas (0 a 1), então multiplicamos pela resolução do frame.
                cv2.putText(image, f"Chute: {int(kick_leg_angle)}", 
                            tuple(np.multiply(knee_kick, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f"Apoio: {int(support_leg_angle)}", 
                            tuple(np.multiply(knee_support, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            except:
                # Se nenhum landmark for detectado no frame, o código simplesmente continua.
                pass

            # Desenha o "esqueleto" da pose na imagem
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                     )               
            
            # Atualiza o placeholder do frame com a imagem processada
            frame_placeholder.image(image, channels="BGR", use_container_width=True)
            
            # Atualiza o placeholder das métricas com os valores mais recentes
            with metrics_placeholder.container():
                st.markdown("---")
                st.markdown(f"#### Métricas de Desempenho ({column_name})")
                st.info(f"Máxima Extensão da Perna de Chute: **{max_kick_leg_angle:.2f}°**")
                st.warning(f"Mínima Flexão da Perna de Apoio (Estabilidade): **{min_support_leg_angle:.2f}°**")
            
            # Adiciona um pequeno delay para a visualização não ser muito rápida
            time.sleep(0.03)

    # Libera o arquivo de vídeo após o processamento
    cap.release()
    # Retorna os valores finais encontrados
    return max_kick_leg_angle, min_support_leg_angle

# --- Interface do Streamlit ---

# Configura a página para usar o layout "wide" (tela cheia) e define um título
st.set_page_config(layout="wide", page_title="Analisador de Chute de Futebol")

# Título principal da aplicação
st.title("⚽ Analisador de Chute com Visão Computacional")

# Texto de instrução para o usuário usando Markdown
st.markdown("""
Esta aplicação utiliza o **MediaPipe** para analisar a biomecânica de um chute de futebol.
- **Faça o upload de dois vídeos**: um de um chute amador e outro de um profissional.
- A aplicação irá **detectar os pontos corporais** (landmarks) em cada frame.
- **Calculará ângulos importantes** para avaliar a técnica do chute.
- **Exibirá os resultados lado a lado** para uma comparação clara.
""")
st.markdown("---")

# Cria duas colunas para os botões de upload de vídeo
col1_upload, col2_upload = st.columns(2)

with col1_upload:
    amateur_video = st.file_uploader("Carregue o Vídeo Amador", type=["mp4", "mov", "avi", "asf", "m4v"])

with col2_upload:
    professional_video = st.file_uploader("Carregue o Vídeo Profissional", type=["mp4", "mov", "avi", "asf", "m4v"])

# Cria duas colunas para exibir os vídeos e as análises
col1_display, col2_display = st.columns(2)

# Dicionário para armazenar os resultados finais
results_data = {}

# A análise só começa se ambos os vídeos tiverem sido carregados
if amateur_video is not None and professional_video is not None:
    # Cria o botão para iniciar a análise
    if st.button("▶️ Iniciar Análise Comparativa"):
        # Processa o vídeo amador na primeira coluna
        with col1_display:
            amateur_kick_angle, amateur_support_angle = process_video(amateur_video, "Amador")
            results_data["Amador"] = {"kick": amateur_kick_angle, "support": amateur_support_angle}
        
        # Processa o vídeo profissional na segunda coluna
        with col2_display:
            prof_kick_angle, prof_support_angle = process_video(professional_video, "Profissional")
            results_data["Profissional"] = {"kick": prof_kick_angle, "support": prof_support_angle}
        
        # --- Resumo Comparativo ---
        # Exibe uma linha horizontal para separar as seções
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("🏁 Resumo Comparativo Final")
        
        # Cria três colunas para o resumo final
        col1_res, col2_res, col3_res = st.columns(3)
        
        with col1_res:
            # Mostra as métricas de extensão do chute
            st.metric(
                label="Máx. Extensão do Chute (Amador)",
                value=f"{results_data['Amador']['kick']:.2f}°"
            )
            st.metric(
                label="Máx. Extensão do Chute (Profissional)",
                value=f"{results_data['Profissional']['kick']:.2f}°",
                # O 'delta' mostra a diferença entre o profissional e o amador
                delta=f"{results_data['Profissional']['kick'] - results_data['Amador']['kick']:.2f}°"
            )
        
        with col2_res:
            # Mostra as métricas de estabilidade da perna de apoio
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
            # Fornece insights automatizados baseados nos resultados
            st.subheader("💡 Insights")
            if results_data['Profissional']['kick'] > results_data['Amador']['kick']:
                st.success("**Potência Superior:** O profissional alcança uma maior extensão da perna, o que geralmente se traduz em mais potência no chute.")
            else:
                st.warning("**Potência a ser melhorada:** O chute amador teve uma extensão similar ou maior, mas isso pode indicar falta de controle.")
            
            if results_data['Profissional']['support'] < results_data['Amador']['support']:
                st.success("**Base Sólida:** A perna de apoio do profissional se mantém mais flexionada e estável, essencial para o equilíbrio e precisão.")
            else:
                st.warning("**Equilíbrio a ser melhorado:** O amador tende a ter a perna de apoio mais reta, o que pode comprometer o equilíbrio.")

# Mensagens de status para o usuário
elif amateur_video is not None or professional_video is not None:
    st.info("Por favor, carregue os dois vídeos para iniciar a análise comparativa.")
else:
    st.info("Aguardando o upload dos vídeos...")
