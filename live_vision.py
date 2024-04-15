import cv2
import numpy as np

# Função de callback para os controles deslizantes
def update_hsv_values(x):
    pass

# Inicialize a captura de vídeo a partir da câmera (0 para a câmera padrão)
cap = cv2.VideoCapture(0)

# Crie uma janela para exibir os controles deslizantes
cv2.namedWindow('Controles HSV')

# Defina os valores iniciais dos controles deslizantes
h_min = 0
s_min = 208
v_min = 0
h_max = 255
s_max = 255
v_max = 255

# Crie controles deslizantes para os valores HSV
cv2.createTrackbar('H Min', 'Controles HSV', h_min, 255, update_hsv_values)
cv2.createTrackbar('S Min', 'Controles HSV', s_min, 255, update_hsv_values)
cv2.createTrackbar('V Min', 'Controles HSV', v_min, 255, update_hsv_values)
cv2.createTrackbar('H Max', 'Controles HSV', h_max, 255, update_hsv_values)
cv2.createTrackbar('S Max', 'Controles HSV', s_max, 255, update_hsv_values)
cv2.createTrackbar('V Max', 'Controles HSV', v_max, 255, update_hsv_values)

while True:
    # Capture um quadro da câmera
    ret, frame = cap.read()
    if not ret:
        break

    # Obtenha os valores dos controles deslizantes
    h_min = cv2.getTrackbarPos('H Min', 'Controles HSV')
    s_min = cv2.getTrackbarPos('S Min', 'Controles HSV')
    v_min = cv2.getTrackbarPos('V Min', 'Controles HSV')
    h_max = cv2.getTrackbarPos('H Max', 'Controles HSV')
    s_max = cv2.getTrackbarPos('S Max', 'Controles HSV')
    v_max = cv2.getTrackbarPos('V Max', 'Controles HSV')

    # Converta o quadro para o espaço de cores HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    # Crie uma máscara usando os valores de limiar definidos
    mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)

    # Encontre os contornos na máscara
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhe as linhas detectadas no quadro original em verde
    frame_linhas = frame.copy()
    cv2.drawContours(frame_linhas, contornos, -1, (0, 255, 0), 2)

    # Exibir o quadro com as linhas verdes
    cv2.imshow('Webcam com Linhas Verdes', frame_linhas)

    # Exibir a máscara
    cv2.imshow('Mascara', mask)

    # Pressione a tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a captura da câmera, feche as janelas e encerre o programa
cap.release()
cv2.destroyAllWindows()
