#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# Carregue a imagem
imagem = cv2.imread('./data/img.jpg')

# Converta a imagem para o espaço de cores HSV
imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# Defina os valores mínimos e máximos de H, S e V para criar a máscara
h_min = 0
s_min = 208
v_min = 0
h_max = 255
s_max = 255
v_max = 255

lower_bound = np.array([h_min, s_min, v_min])
upper_bound = np.array([h_max, s_max, v_max])

# Crie uma máscara usando os valores de limiar definidos
mask = cv2.inRange(imagem_hsv, lower_bound, upper_bound)

# Encontre os contornos na máscara
contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenhe as linhas detectadas na imagem original em verde
imagem_linhas = imagem.copy()
cv2.drawContours(imagem_linhas, contornos, -1, (0, 255, 0), 2)

# Exibir a imagem com as linhas verdes
cv2.imshow('Imagem com Linhas Verdes', imagem_linhas)

# Exibir a máscara
cv2.imshow('Máscara', mask)

# Aguarde até que uma tecla seja pressionada e, em seguida, feche as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
