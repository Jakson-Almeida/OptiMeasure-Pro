import pygame
import sys

# Inicializa o pygame
pygame.init()

# Configura a tela
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Digite seu texto")

# Define cores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Configura a fonte
font = pygame.font.Font(None, 48)

# Variável para guardar o texto que está sendo digitado
text = ''

# Loop principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                # Remove o último caractere
                text = text[:-1]
            elif event.key == pygame.K_RETURN:
                # Encerra a entrada de texto
                print("Texto finalizado:", text)
                running = False
            else:
                # Adiciona o novo caractere
                text += event.unicode

    # Limpa a tela
    screen.fill(BLACK)

    # Renderiza o texto
    text_surface = font.render(text, True, WHITE)
    screen.blit(text_surface, (50, 200))

    # Atualiza a tela
    pygame.display.flip()

# Finaliza o pygame
pygame.quit()
sys.exit()
