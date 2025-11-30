import streamlit as st
import cv2
import numpy as np
import io
import pandas as pd # Importado para criar o DataFrame da linha do tempo

# Configuração da página Streamlit
st.set_page_config(
    page_title="Analisador de Visibilidade de Marca (CV)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes de Configuração de CV ---
# Mínimo de 'bons' matches para considerar que o logo foi detectado
MIN_MATCH_COUNT = 10 
# Taxa de similaridade para o teste de razão de Lowe (menor é mais rigoroso)
LOWE_RATIO_THRESHOLD = 0.7

# Função auxiliar para formatar segundos em M:S
def format_time(seconds):
    """Converte segundos totais em formato de string M:S."""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}m {remaining_seconds}s"

# Função para inicializar o detector ORB e o matcher
def initialize_cv_components():
    """Inicializa o detector ORB e o matcher Brute-Force."""
    # ORB é rápido e eficiente para detecção de features.
    orb = cv2.ORB_create(nfeatures=5000)
    # BFMatcher (Brute-Force Matcher) com a distância de Hamming, ideal para ORB.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return orb, bf

# Função principal de detecção de logo em um quadro
def analyze_frame(frame, logo_img, orb, bf, logo_kp, logo_des):
    """
    Detecta o logo no frame usando Feature Matching (ORB) e desenha o bounding box.

    Args:
        frame (np.array): O quadro atual do vídeo (BGR).
        logo_img (np.array): A imagem de referência do logo (Grayscale).
        orb, bf: Componentes de CV inicializados.
        logo_kp, logo_des: Keypoints e Descriptores do logo de referência.

    Returns:
        bool: True se o logo for detectado, False caso contrário.
        np.array: O frame com a caixa delimitadora desenhada (ou o frame original).
    """
    if logo_des is None:
        return False, frame

    # Converter o frame para escala de cinza para a detecção de features (ORB funciona melhor em grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Detectar key points e descritores no frame atual
    frame_kp, frame_des = orb.detectAndCompute(gray_frame, None)

    if frame_des is None:
        return False, frame

    # 2. Encontrar os 2 melhores matches para cada descritor do frame
    matches = bf.knnMatch(logo_des, frame_des, k=2)

    # 3. Aplicar o Teste da Razão de Lowe para filtrar bons matches
    good_matches = []
    for m, n in matches:
        if m.distance < LOWE_RATIO_THRESHOLD * n.distance:
            good_matches.append(m)

    # 4. Verificar se há matches suficientes
    if len(good_matches) > MIN_MATCH_COUNT:
        # Logo foi encontrado!

        # Obter os keypoints correspondentes (pontos de origem e destino)
        src_pts = np.float32([logo_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Encontrar a matriz de Homografia (M) usando RANSAC (robusto contra outliers)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Se a Homografia foi calculada com sucesso (M não é None)
        if M is not None:
            # Pegar as coordenadas dos 4 cantos da imagem do logo (objeto a ser encontrado)
            h, w = logo_img.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            # Aplicar a transformação de perspectiva para obter as coordenadas no frame de vídeo
            dst = cv2.perspectiveTransform(pts, M)
            
            # Desenhar o polígono (a caixa delimitadora)
            # O cv2.polylines exige pontos inteiros
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA) # Cor verde

            # Desenhar uma mensagem de sucesso no frame
            cv2.putText(frame, 
                        "MARCA DETECTADA!", 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, 
                        (0, 255, 0), # Verde
                        3, 
                        cv2.LINE_AA)
            
            return True, frame
        else:
            # Homografia falhou, trata como não detectado
            return False, frame
    else:
        # Logo não detectado
        return False, frame

def main_app():
    """Função principal que roda o aplicativo Streamlit."""

    st.title("👁️ Análise de Visibilidade de Marca em Vídeos")
    st.markdown("Use esta ferramenta para carregar um vídeo e o seu logo de marca para calcular o tempo total de aparição. O sistema usa a técnica de *Feature Matching* (ORB) do OpenCV para maior robustez.")

    # --- Uploads de Arquivos ---
    col1, col2 = st.columns(2)

    with col1:
        video_file = st.file_uploader("Upload do Vídeo do Influencer (.mp4, .mov, etc.)", type=['mp4', 'mov', 'avi'])
    
    with col2:
        logo_file = st.file_uploader("Upload da Imagem do Logo de Marca (.png, .jpg)", type=['png', 'jpg', 'jpeg'])

    
    # --- Processamento ---
    if video_file and logo_file:
        
        st.divider()

        # 1. Preparar a imagem do logo
        try:
            # Ler o arquivo de imagem do logo
            logo_data = logo_file.read()
            logo_np = np.frombuffer(logo_data, np.uint8)
            logo_img = cv2.imdecode(logo_np, cv2.IMREAD_GRAYSCALE) # Leitura em escala de cinza para ORB
            
            if logo_img is None:
                st.error("Erro ao carregar ou decodificar a imagem do logo. Certifique-se de que é um arquivo de imagem válido.")
                return

        except Exception as e:
            st.error(f"Erro ao processar a imagem do logo: {e}")
            return

        # 2. Inicializar o detector ORB e calcular os descritores do logo
        orb, bf = initialize_cv_components()
        logo_kp, logo_des = orb.detectAndCompute(logo_img, None)

        if logo_des is None or len(logo_kp) < MIN_MATCH_COUNT:
            st.warning("O logo de referência tem poucos pontos de interesse. Tente usar uma imagem de logo mais detalhada.")
            return

        st.subheader("⚙️ Análise em Andamento...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Salvar o arquivo de vídeo temporariamente (Necessário para cv2.VideoCapture)
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(video_file.getbuffer())

        # 3. Processar o vídeo
        cap = cv2.VideoCapture(tfile)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_per_frame = 1.0 / fps if fps > 0 else 0
        
        total_detection_time = 0.0
        current_frame_index = 0
        
        # Lista para armazenar o status de detecção (para o gráfico da linha do tempo)
        detection_timeline = []
        
        # Placeholder para o feed de vídeo processado (melhor para Streamlit)
        frame_display_placeholder = st.empty()
        
        if frame_count == 0 or fps == 0:
            st.error("Erro: O vídeo não pôde ser lido ou tem FPS zero. Tente outro formato de arquivo.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Fim do vídeo

            # Análise do quadro - Passamos o frame BGR original para desenhar a caixa colorida.
            detected, output_frame = analyze_frame(frame, logo_img, orb, bf, logo_kp, logo_des)
            
            current_time = current_frame_index * duration_per_frame
            
            # Se detectado, adiciona o tempo do quadro ao total e registra na linha do tempo
            if detected:
                total_detection_time += duration_per_frame
                detection_timeline.append({'Tempo (s)': current_time, 'Aparicão da Marca': 1})
            else:
                detection_timeline.append({'Tempo (s)': current_time, 'Aparicão da Marca': 0})
            
            # Exibir o frame processado no Streamlit (convertendo de BGR para RGB para exibição)
            output_rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            
            # Definindo uma largura de 600px para economizar espaço
            frame_display_placeholder.image(output_rgb_frame, channels="RGB", width=600)


            # Atualizar progresso
            current_frame_index += 1
            progress = min(100, int(current_frame_index / frame_count * 100))
            progress_bar.progress(progress)
            status_text.text(f"Quadro {current_frame_index}/{frame_count} - Tempo de aparição total: {total_detection_time:.2f} segundos")

        # 4. Finalizar e Exibir Resultados
        cap.release()
        progress_bar.empty()
        status_text.empty()
        st.subheader("✅ Análise Concluída!")

        # --- NOVAS MÉTRICAS ---
        total_video_duration = frame_count * duration_per_frame
        unseen_time = total_video_duration - total_detection_time

        total_duration_formatted = format_time(total_video_duration)
        detected_time_formatted = format_time(total_detection_time)
        unseen_time_formatted = format_time(unseen_time)
        
        # Exibir o resultado final de forma destacada
        st.header("📊 Resumo das Métricas de Visibilidade")
        col_total, col_detected, col_unseen = st.columns(3)

        # Métrica 1: Duração Total
        with col_total:
            st.metric(label="Duração Total do Vídeo", value=total_duration_formatted)

        # Métrica 2: Tempo de Aparição da Marca
        with col_detected:
            # Calcular delta como porcentagem
            detection_percentage = (total_detection_time / total_video_duration) if total_video_duration > 0 else 0
            st.metric(label="Tempo Total de Aparição da Marca", 
                      value=detected_time_formatted, 
                      delta=f"{detection_percentage:.1%} do vídeo", 
                      delta_color="normal")

        # Métrica 3: Tempo Sem Aparição (Diferença)
        with col_unseen:
            unseen_percentage = (unseen_time / total_video_duration) if total_video_duration > 0 else 0
            st.metric(label="Tempo Sem Aparição da Marca", 
                      value=unseen_time_formatted, 
                      delta=f"-{unseen_percentage:.1%} do vídeo",
                      delta_color="inverse")
                      
        st.info(f"O vídeo foi processado a uma taxa de {fps:.2f} quadros por segundo (FPS).")
        st.divider()

        # --- GRÁFICO DE LINHA (LINHA DO TEMPO) ---
        st.header("📈 Linha do Tempo da Aparição da Marca")
        st.markdown("O gráfico abaixo mostra quando a marca estava **visível (pico em 1)** ao longo do tempo (eixo X em segundos).")

        if detection_timeline:
            df_timeline = pd.DataFrame(detection_timeline)
            df_timeline = df_timeline.set_index('Tempo (s)')
            st.line_chart(df_timeline)
        else:
            st.warning("Não foi possível gerar a linha do tempo, pois a marca não foi detectada em nenhum quadro.")

    else:
        st.info("Por favor, faça o upload do arquivo de vídeo e da imagem do logo para iniciar a análise.")

if __name__ == '__main__':
    main_app()
