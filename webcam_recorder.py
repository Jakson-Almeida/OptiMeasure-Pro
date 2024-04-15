#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

# Inicialize a captura de vídeo a partir da câmera (0 para a câmera padrão)
cap = cv2.VideoCapture(0)

# Defina a largura e altura do vídeo
largura = int(cap.get(3))
altura = int(cap.get(4))

# Define o codec e cria um objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' para formato .mp4
out = cv2.VideoWriter('video.mp4', fourcc, 20.0, (largura, altura))

while True:
    # Capture um quadro da câmera
    ret, frame = cap.read()
    if not ret:
        break

    # Grave o quadro no vídeo
    out.write(frame)

    # Exiba o quadro
    cv2.imshow('Gravacao de Video', frame)

    # Pressione a tecla 'q' para parar a gravação e sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a captura da câmera e o objeto VideoWriter
cap.release()
out.release()

# Feche todas as janelas
cv2.destroyAllWindows()
