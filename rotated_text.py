import cv2
import numpy as np

# Função para rotacionar um ponto em torno de outro ponto
def rotate_point(point, angle, center):
    angle_rad = np.radians(angle)
    # angle_rad = angle
    x, y = point[0] - center[0], point[1] - center[1]
    new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return int(new_x + center[0]), int(new_y + center[1])

# Configurações de texto
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
thickness = 2
lineType = cv2.LINE_AA

# Angulo em graus
angle = 45

# Ponto base
bottomLeftCornerOfText = (30, 100)

# Ponto de rotação (por exemplo, canto inferior esquerdo do texto)
rotation_center = bottomLeftCornerOfText

# Criar matriz de rotação
rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1)

# Aplicar rotação ao ponto de texto
rotated_text_point = rotate_point(bottomLeftCornerOfText, angle, rotation_center)

# Adicionar texto ao frame
frame_lines = np.zeros((200, 400, 3), dtype=np.uint8)  # Substitua isso pelo seu frame real
cv2.putText(frame_lines, 'Fiber angle: {:3.2f} graus'.format(-angle), rotated_text_point, font, fontScale, fontColor, thickness, lineType)

# Exibir o frame (apenas para fins de teste)
cv2.imshow('Rotated Text', frame_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
