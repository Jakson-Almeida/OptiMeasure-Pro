import cv2
import numpy as np
import math

x_mouse, y_mouse = [0, 0]

def button_callback(event, x, y, flags, param):
    global x_mouse, y_mouse
    if event == cv2.EVENT_MOUSEMOVE:
        x_mouse = x
        y_mouse = y

def plot_arrow(img, size, color, orientation, pose):
    """
    Plota uma seta em uma imagem utilizando OpenCV.

    Parâmetros:
    - img: Imagem na qual a seta será desenhada.
    - size: Tamanho da seta.
    - color: Cor da seta no formato (B, G, R).
    - orientation: Ângulo de orientação da seta em graus (0° aponta para a direita).
    - pose: Posição da seta no formato (x, y).
    """

    # Converte o ângulo para radianos
    angle_rad = math.radians(orientation)

    # Calcula as coordenadas dos pontos finais da seta
    x1 = int(pose[0])
    y1 = int(pose[1])
    x2 = int(pose[0] + size * math.cos(angle_rad))
    y2 = int(pose[1] - size * math.sin(angle_rad))

    # Desenha a linha principal da seta
    cv2.line(img, (x1, y1), (x2, y2), color, 2)

    # Calcula as coordenadas dos pontos para as pontas da seta
    x3 = int(x2 - size * 0.3 * math.cos(angle_rad + math.pi / 6))
    y3 = int(y2 + size * 0.3 * math.sin(angle_rad + math.pi / 6))
    x4 = int(x2 - size * 0.3 * math.cos(angle_rad - math.pi / 6))
    y4 = int(y2 + size * 0.3 * math.sin(angle_rad - math.pi / 6))

    # Desenha as pontas da seta
    cv2.line(img, (x2, y2), (x3, y3), color, 2)
    cv2.line(img, (x2, y2), (x4, y4), color, 2)

cv2.namedWindow('Arrow')
cv2.setMouseCallback('Arrow', button_callback)

# Exemplo de uso
image = np.zeros((500, 500, 3), dtype=np.uint8)  # Imagem preta
arrow_color = (0, 255, 0)  # Cor verde
arrow_size = 50
arrow_orientation = 45  # 45 graus
arrow_pose = (250, 250)  # Posição no centro da imagem

while True:
    arrow_orientation -= 0.1
    image = np.zeros((500, 500, 3), dtype=np.uint8)  # Imagem preta
    arrow_pose = [x_mouse, y_mouse]
    plot_arrow(image, arrow_size, arrow_color, arrow_orientation, arrow_pose)

    # Exibe a imagem com a seta
    cv2.imshow("Arrow", image)

    # Press the 'q' key to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Close window button
    if cv2.getWindowProperty('Arrow', cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
