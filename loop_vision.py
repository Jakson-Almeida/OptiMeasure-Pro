import cv2
import numpy as np
import pygame.camera
from classes_lib import find_rect_polygon, find_bounding_polygon, point_inside_polygon, find_intersection_points, find_rect_medium_angle, compact_list, find_LEFTDOWN_Point, find_LEFTUP_Point, find_RIGHTDOWN_Point, find_RIGHTUP_Point, rotate_points, PlayPause, Cursor, TextInput
import time
import datetime

# Detecting the fiber
fiber_detect = False
fiber_finded = False
x_mouse = 0
y_mouse = 0

# Initialize plot flags
plot_polygon = False
plot_contours = False
plot_mask = False
plot_hawk_eye = False
plot_filtered_mask = True
plot_rotated_frame = False
plot_text_camp = False
video_play = True
change_angle = True
RIGHT_BUTTON_CLICKED = False
LEFT_BUTTON_CLICKED = False
SPACE_KEY_CLICKED = False


# OpenCV text
font      = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
fontColor     = ( 30,  30,  30)
fontColorDark = (210, 210, 210)
thickness = 2
lineType  = 1

PC = PlayPause(pose=(20,20), size=30)
CS = Cursor()
TI = TextInput()

# Text Imput pose
TI.setPose((30, 400))

def rotate_point(point, angle, center):
    angle_rad = np.radians(angle)
    x, y = point[0] - center[0], point[1] - center[1]
    new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return int(new_x + center[0]), int(new_y + center[1])

# Callback function for the HSV sliders
def update_hsv_values(x):
    pass

def button_callback(event, x, y, flags, param):
    global x_mouse, y_mouse, plot_polygon, fiber_detect, video_play, change_angle, RIGHT_BUTTON_CLICKED, LEFT_BUTTON_CLICKED
    if event == cv2.EVENT_LBUTTONDOWN:
        LEFT_BUTTON_CLICKED = True
        if plot_rotated_frame:
            change_angle = not change_angle
        # if fiber_detect:
        #     plot_polygon = not plot_polygon
        if PC.is_on((x, y)):
            PC.play = not PC.play
            video_play = PC.play
        # print(f'Clique esquerdo do mouse em ({x}, {y})')
    if event == cv2.EVENT_RBUTTONDOWN:
        # print(f'Clique direito do mouse em ({x}, {y})')
        RIGHT_BUTTON_CLICKED = True
    if event == cv2.EVENT_MOUSEMOVE:
        x_mouse = x
        y_mouse = y

cv2.namedWindow('OptiMeasure Pro')
cv2.setMouseCallback('OptiMeasure Pro', button_callback)

pygame.camera.init()

# Lista todas as webcams disponíveis
cameras = pygame.camera.list_cameras()
for i, camera in enumerate(cameras):
    print(f"Webcam {i + 1}: {camera}")

# Try to initialize video capture from the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# If the webcam is not available, switch to video file
if not cap.isOpened():
    print("Webcam by 0 index not found. Trying index 1.")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Webcam by 1 index not found. Trying index 2.")
        cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Webcam by 2 index not found. Trying index 3.")
            cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("Webcam not found. Using video file.")
                cap.release()
                cap = cv2.VideoCapture('video.mp4')

# Create a window to display the HSV sliders
cv2.namedWindow('HSV Controls')

# Set initial values for the HSV sliders
h_min = 0
s_min = 60
v_min = 0
h_max = 255
s_max = 255
v_max = 255

# Set an area threshold below which contours will be considered noise
area_threshold = 400  # Adjust as needed

# Create sliders for HSV values
cv2.createTrackbar('H Min', 'HSV Controls', h_min, 255, update_hsv_values)
cv2.createTrackbar('S Min', 'HSV Controls', s_min, 255, update_hsv_values)
cv2.createTrackbar('V Min', 'HSV Controls', v_min, 255, update_hsv_values)
cv2.createTrackbar('H Max', 'HSV Controls', h_max, 255, update_hsv_values)
cv2.createTrackbar('S Max', 'HSV Controls', s_max, 255, update_hsv_values)
cv2.createTrackbar('V Max', 'HSV Controls', v_max, 255, update_hsv_values)
cv2.createTrackbar('Filter', 'HSV Controls', area_threshold, 3000, update_hsv_values)

# To recorder
last_frame = None
last_angulo = 0

while True:
    try:
        # Capture a frame from the video
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video when it ends
            continue
        if video_play:
            last_frame = frame.copy()
        elif last_frame is not None:
            frame = last_frame.copy()

        #########################################################
        if plot_rotated_frame:
            # Especifica o ângulo de rotação (em graus)
            angulo = x_mouse
            if not change_angle:
                angulo = last_angulo
            last_angulo = angulo

            # Calcula o centro da imagem
            altura, largura = frame.shape[:2]
            centro = (largura // 2, altura // 2)

            # Cria a matriz de rotação
            matriz_rotacao = cv2.getRotationMatrix2D(centro, angulo, 1.0)

            fontColorSelected = fontColorDark if plot_mask else fontColor
            cv2.putText(frame,'OptiMeasure Pro', 
                (40, 40), 
                font, 
                fontScale,
                fontColorSelected,
                thickness,
                lineType)

            # Aplica a matriz de rotação à imagem
            frame_rotacionado = cv2.warpAffine(frame, matriz_rotacao, (largura, altura), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            frame = frame_rotacionado.copy()

            # cv2.imshow('Frame Rotacionado', frame_rotacionado)

        #########################################################

        # Get the values from the HSV sliders
        h_min = cv2.getTrackbarPos('H Min', 'HSV Controls')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Controls')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Controls')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Controls')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Controls')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Controls')
        area_threshold = cv2.getTrackbarPos('Filter', 'HSV Controls')

        # Convert the frame to the HSV color space
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # Create a mask using the defined threshold values
        mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)

        # Morfologic operations
        mask = cv2.dilate(mask, None, iterations=8)
        mask = cv2.erode (mask, None, iterations=8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

        if plot_filtered_mask:
            mask_filtered = mask
            mask_filtered = np.zeros_like(mask)
            cv2.fillPoly(mask_filtered, filtered_contours, color=255)
            mask = mask_filtered
            contours = filtered_contours

        # Draw the detected lines on the original frame in green
        frame_lines = frame.copy()

        angle = 0

        if plot_mask:
            mask_as_image = np.zeros_like(frame)
            cv2.fillPoly(mask_as_image, contours, color=255)
            mask_as_image[mask > 0] = 255
            frame_lines = mask_as_image

        # Plot polygon and contours if enabled
        fiber_finded = False
        bounding_polygon = find_bounding_polygon(contours, nL=4)
        if bounding_polygon is not None and len(bounding_polygon) >= 2:
            fiber_finded = True
            fiber_detect = point_inside_polygon([x_mouse, y_mouse], bounding_polygon)
            angle = find_rect_medium_angle(compact_list(bounding_polygon))
            CS.update_parameters(bounding_polygon, x_mouse, y_mouse, angle+90, LEFT_BUTTON_CLICKED or SPACE_KEY_CLICKED, mask)
            # cv2.imshow('Mask', mask)

            if plot_hawk_eye:
                angulo = angle
                frame = frame_lines.copy()

                # Calcula o centro da imagem
                altura, largura = frame.shape[:2]
                centro = (largura // 2, altura // 2)

                # matriz_pontos = np.array([find_LEFTDOWN_Point(compact_list(bounding_polygon)), find_LEFTUP_Point(compact_list(bounding_polygon)), find_RIGHTDOWN_Point(compact_list(bounding_polygon)), find_RIGHTUP_Point(compact_list(bounding_polygon))])
                matriz_pontos = np.array(compact_list(bounding_polygon))

                # Calcular as médias das coordenadas x e y
                centroide_x = np.mean(matriz_pontos[:, 0])
                centroide_y = np.mean(matriz_pontos[:, 1])
                desloca = int(centro[0] - centroide_x), int(centro[1] - centroide_y)

                # Cria a matriz de rotação
                matriz_rotacao = cv2.getRotationMatrix2D(centro, angulo, 1.0)

                # Definir a matriz de translação
                matriz_translacao = np.float32([[1, 0, desloca[0]], [0, 1, desloca[1]]])

                # Aplica a matriz de rotação e de translação à imagem
                frame = cv2.warpAffine(frame, matriz_translacao, (largura, altura))
                frame_rotacionado = cv2.warpAffine(frame, matriz_rotacao, (largura, altura), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                cv2.imshow('Hawk eye', frame_rotacionado)

            if plot_polygon:
                # Create a mask for the polygon
                polygon_mask = np.zeros_like(frame_lines)

                # Convert polygon coordinates to the format accepted by fillPoly
                bounding_polygon_np = np.array(bounding_polygon, dtype=np.int32)
                bounding_polygon_np = bounding_polygon_np.reshape((-1, 1, 2))

                # Rotate
                # reference_point = np.array([bounding_polygon[0][0][0], bounding_polygon[0][0][1]])
                # angle_degrees = x_mouse*0.2
                # rotated_polygon = rotate_points(bounding_polygon_np.reshape((-1, 2)), reference_point, angle_degrees)
                # bounding_polygon_np = rotated_polygon.reshape((-1, 1, 2)).astype(np.int32)

                # Fill the polygon in the mask
                clr = (0, 230, 230) if fiber_detect else (0, 190, 0)
                cv2.fillPoly(polygon_mask, [bounding_polygon_np], clr)

                # Blend the original image with the mask using 75% transparency
                tpc = 0.75 if not plot_mask else 1
                frame_lines = cv2.addWeighted(frame_lines, 1, polygon_mask, tpc, 0)

        # Plot contours if enabled
        if plot_contours:
            cv2.drawContours(frame_lines, contours, -1, (255, 0, 0), thickness=2)
        
        if fiber_finded and fiber_detect:
            CS.show(frame_lines, plot_mask=(plot_mask and plot_contours), show_reference=not video_play)
            # OpenCV text
            point_down = find_LEFTDOWN_Point(compact_list(bounding_polygon))
            bottomLeftCornerOfText = (int(point_down[0] + 30), int(point_down[1] - 30))
            fontColorSelected = fontColorDark if plot_mask else fontColor
            # second = (bottomLeftCornerOfText[0] + int(150*np.cos(angle*np.pi/180.0)), bottomLeftCornerOfText[1] + int(150*np.sin(angle*np.pi/180.0)))
            # cv2.line(frame_lines, bottomLeftCornerOfText, second, (10, 100, 100), 3)
            cv2.putText(frame_lines,'Fiber size: {:3.0f} pixels'.format(CS.size), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColorSelected,
                thickness,
                lineType)
            point_up = find_LEFTUP_Point(compact_list(bounding_polygon))
            bottomLeftCornerOfText = (int(point_up[0] + 30), int(point_up[1] + 30))
            cv2.putText(frame_lines,'Percentual: {:3.2f} %'.format(CS.percentual), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColorSelected,
                thickness,
                lineType)
            # print(-angle)


        # Play and pause controllers
        PC.setFrame(frame_lines)
        PC.is_on((x_mouse, y_mouse))
        PC.show()

        # Text camp
        if plot_text_camp:
            # TI.background(frame_lines)
            TI.show(frame_lines, x_mouse, y_mouse)

        # Show frame
        cv2.imshow('OptiMeasure Pro', frame_lines)

        # To boolean flags
        RIGHT_BUTTON_CLICKED = False
        LEFT_BUTTON_CLICKED = False
        SPACE_KEY_CLICKED = False

        # Press the 'q' key to exit the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            plot_polygon = not plot_polygon
        elif key == ord('c'):
            plot_contours = not plot_contours
        elif key == ord('1'):
            plot_mask = False
        elif key == ord('2'):
            plot_mask = True
            plot_polygon = False
        elif key == ord('h'):
            plot_hawk_eye = not plot_hawk_eye
            if not plot_hawk_eye:
                cv2.destroyWindow("Hawk eye")
        elif key == ord('r'):
            plot_rotated_frame = not plot_rotated_frame
        elif key == ord('f'):
            plot_filtered_mask = not plot_filtered_mask
        elif key == ord('t'):
            plot_text_camp = not plot_text_camp
        elif key == ord('s'):
            # Obter a data e hora atual
            now = datetime.datetime.now()
            # Formatar a data e hora para usar no nome do arquivo
            filename = now.strftime("%Y-%m-%d_%H-%M-%S") + '.jpg'
            # Salva o frame atual em um arquivo com a data e hora no nome
            cv2.imwrite("./data/images/"+filename, frame_lines)
            print(f"Frame salvo como {filename}!")
        elif key == ord(' '):
            SPACE_KEY_CLICKED = not SPACE_KEY_CLICKED
            PC.play = not PC.play
            video_play = PC.play
            if not PC.play:
                PC.time_pause = time.time()
            else:
                PC.time_play = time.time()
        elif key == 27:
            break
        # Close window button
        if cv2.getWindowProperty('OptiMeasure Pro', cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        break

# Release the video capture, close the windows, and exit the program
cap.release()
cv2.destroyAllWindows()
