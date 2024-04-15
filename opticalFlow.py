import cv2
import numpy as np

# Inicializa o vídeo e o objeto OpenCV
video = cv2.VideoCapture('video.mp4')
width, height = int(video.get(3)), int(video.get(4))
opencv = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Atualiza o modelo de fundo
    mask = opencv.apply(frame)

    # Aplica operações morfológicas para melhorar a segmentação
    mask = cv2.dilate(mask, None, iterations=3)
    mask = cv2.erode(mask, None, iterations=3)

    # Encontra contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenha os contornos na imagem original
    result = frame.copy()
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

    # Exibe a imagem resultante
    cv2.imshow('Video com Deteccao de Contornos', result)

    if cv2.waitKey(30) & 0xFF == 27:  # Pressione 'Esc' para sair
        break

# Libera os recursos
video.release()
cv2.destroyAllWindows()
