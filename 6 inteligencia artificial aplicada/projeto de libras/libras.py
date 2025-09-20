import cv2
import mediapipe as mp

# Adendos importantes para o funcionamento do MediaPipe e também do OpenCV
# Utilizei a Venv para instalar as bibliotecas pois estava dando conflito com outras versões mas se não for o caso de quem for rodar o código, basta instalar as bibliotecas normalmente

mp_drawing = mp.solutions.drawing_utils # Utilizado para desenhar os pontos de referência da mão
mp_hands = mp.solutions.hands # Utilizado para detectar as mãos
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) # Aqui eu to pedindo pra ele pegar somente uma mão e com uma confiança mínima de 70%
cap = cv2.VideoCapture(0) # Aqui eu to pegando a câmera do meu computador que não funciona no colab by the way

print("Pressione a tecla 'q' para fechar a janela.")

# aqui é um loop infinito que vai rodar até a gente apertar a tecla 'q'
while cap.isOpened(): # enquanto a câmera estiver aberta
    success, image = cap.read() # aqui eu to lendo o que a câmera está capturando
    if not success: # se não tiver sucesso, ele vai printar uma mensagem de erro e quebrar o loop
        print("Erro ao capturar imagem.")
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # aqui eu to convertendo a imagem de BGR para RGB porque o MediaPipe trabalha com RGB
    results = hands.process(image_rgb) # aqui eu to processando a imagem para detectar as mãos

    # Este 'if' pergunta: "Você encontrou alguma mão na foto?"
    if results.multi_hand_landmarks:
        # Se a resposta for sim, ele entra aqui.
        for hand_landmarks in results.multi_hand_landmarks:
            # Pegamos nosso "lápis" e desenhamos o esqueleto ('hand_landmarks')
            # na foto original ('image').
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Detector de Mao em Libras', cv2.flip(image, 1)) # aqui eu to mostrando a imagem em uma janela, e flipando ela pra ficar como um espelho

    if cv2.waitKey(5) & 0xFF == ord('q'): # aqui eu to esperando a tecla 'q' ser apertada para sair do loop
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
print("Aplicacao finalizada.")

# -----------------------------------------------------------------------\\------------------------------------------------------------------------------------------\\--------------------------------------------------------

# até aqui  
# Tira uma foto.
# Procura uma mão na foto.
# Se achar, desenha um esqueleto sobre ela.
# Mostra a foto na tela.
# Repete tudo de novo.   