# ===============================================================
# ESTE É O CÓDIGO BASE PARA RODAR NO SEU COMPUTADOR (.PY)
# Objetivo: Apenas visualizar a detecção da mão pela webcam.
# Bibliotecas necessárias: opencv-python, mediapipe
# ===============================================================

import cv2
import mediapipe as mp

# Inicializa as soluções do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Configura o detector de mãos
# - max_num_hands: Número máximo de mãos para detectar.
# - min_detection_confidence: Confiança mínima para a detecção ser considerada bem-sucedida.
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Inicia a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

print("Pressione a tecla 'q' para fechar a janela.")

# Loop principal
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Não foi possível capturar o quadro da câmera.")
        break

    # Converte a imagem para RGB, pois o MediaPipe trabalha com esse formato
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Processa a imagem para detectar as mãos
    results = hands.process(image_rgb)

    # Desenha os landmarks (pontos-chave) e as conexões na imagem original (BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

    # Vira a imagem horizontalmente para um efeito de "espelho" e a exibe
    cv2.imshow('Detector de Mão em Libras', cv2.flip(image, 1))

    # Condição para parar o loop: pressionar a tecla 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Libera os recursos ao finalizar
hands.close()
cap.release()
cv2.destroyAllWindows()
print("Aplicação finalizada.")