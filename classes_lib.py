#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import time
from shapely.geometry import Polygon, LineString, MultiPoint, Point

class TextInput:
    def __init__(self):
        self.text = ""
        # Definir a posição e a cor do retângulo
        self.size = 100
        self.top_left     = (50, 50)
        self.bottom_right = (self.size + self.top_left[0], int(self.size*0.5) + self.top_left[1])
        self.radius       = int(self.size*0.16)
        self.color        = (255, 0, 0)  # Cor azul

    def setPose(self, pose):
        half = int(self.size*0.5)
        self.top_left     = (pose[0]-half, pose[1]-half)
        self.bottom_right = (pose[0]+half, pose[1]+half)
    
    def draw_rounded_rectangle(self, img, top_left, bottom_right, radius, color, thickness):
        """ Desenha um retângulo com bordas arredondadas.
            img: imagem onde desenhar
            top_left: coordenada superior esquerda do retângulo (x, y)
            bottom_right: coordenada inferior direita do retângulo (x, y)
            radius: raio das bordas arredondadas
            color: cor do retângulo em BGR (blue, green, red)
            thickness: espessura da borda; use -1 para preenchimento sólido
        """
        # Pontos para os cantos do retângulo interno
        top_left_inner = (top_left[0] + radius, top_left[1] + radius)
        bottom_right_inner = (bottom_right[0] - radius, bottom_right[1] - radius)

        # Desenha o retângulo interno (área central)
        cv2.rectangle(img, top_left_inner, bottom_right_inner, color, thickness)

        # Desenha as quatro bordas
        cv2.rectangle(img, (top_left[0], top_left_inner[1]), (top_left_inner[0], bottom_right_inner[1]), color, thickness)
        cv2.rectangle(img, (bottom_right_inner[0], top_left_inner[1]), (bottom_right[0], bottom_right_inner[1]), color, thickness)
        cv2.rectangle(img, (top_left_inner[0], top_left[1]), (bottom_right_inner[0], top_left_inner[1]), color, thickness)
        cv2.rectangle(img, (top_left_inner[0], bottom_right_inner[1]), (bottom_right_inner[0], bottom_right[1]), color, thickness)

        # Desenha os quatro cantos com círculos
        cv2.circle(img, top_left_inner, radius, color, thickness)
        cv2.circle(img, (bottom_right_inner[0], top_left_inner[1]), radius, color, thickness)
        cv2.circle(img, (top_left_inner[0], bottom_right_inner[1]), radius, color, thickness)
        cv2.circle(img, bottom_right_inner, radius, color, thickness)
    
    def background(self, img):
        self.draw_rounded_rectangle(img, self.top_left, self.bottom_right, self.radius, self.color, -1)

class PlayPause:
    def __init__(self, frame=None, pose=(0, 0), size=100, color = (40, 40, 40), thickness = cv2.FILLED):
        self.pose = pose
        self.size = size
        self.frame = frame
        self.play = True
        self.isShow = False
        self.isAnimation = False
        self.color = color
        self.thickness = thickness
        self.time_pause = 0
        self.time_play = 0
    
    def setFrame(self, frame):
        self.frame = frame
    
    def is_on(self, point):
        # pl = [[self.pose[0], self.pose[1]], [self.pose[0], self.pose[1]+self.size], [self.pose[0]+self.size, self.pose[1]], [self.pose[0]+self.size, self.pose[1]+self.size]]
        # io = point_inside_polygon(point, pl)
        dist = np.linalg.norm(np.array(point) - np.array(self.pose))
        io = dist < self.size*1.41
        self.isShow = dist < self.size*5
        self.color = (0, 0, 255) if io else (50+5*dist/5.0, 50+5*dist/5.0, 50+5*dist/5.0)
        return io
    
    def is_play(self):
        return self.play
    
    def animation(self):
        self.isAnimation = False
        t_anima = 0.3
        tm = time.time()-self.time_play if self.play else time.time()-self.time_pause
        if tm > t_anima:
            return
        self.isAnimation = True
        if self.play:
            nv = 3
            anima = int(0.14*self.size*np.sin(nv*3.141592653589793238462643383279502884197169*(t_anima-tm)/t_anima))
            size = int(self.size / 2)
            cv2.circle(self.frame, (self.pose[0]+size, self.pose[1]+size), np.abs(anima+size), (0, 0, 255), self.thickness)
        else:
            anima = int(0.6*self.size*(tm)/t_anima)
            add = int(self.size/2)
            cv2.rectangle(self.frame, (self.pose[0]+add-anima, self.pose[1]+add-anima), (self.pose[0]+add+anima, self.pose[1]+add+anima), (80, 80, 80), self.thickness)
    
    def show(self):
        self.animation()
        if not self.isShow or self.isAnimation:
            return
        if self.play:
            size = int(self.size / 2)
            cv2.circle(self.frame, (self.pose[0]+size, self.pose[1]+size), size, self.color, self.thickness)
        else:
            cv2.rectangle(self.frame, self.pose, (self.pose[0]+self.size, self.pose[1]+self.size), self.color, self.thickness)

class Cursor:
    def __init__(self, size = 100, pose = (0, 0), ref = 100):
        self.size = size
        self.pose = pose
        self.angle = 0
        self.p0 = (0, 0)
        self.p1 = (0, 100)
        self.reference_size = ref
        self.p0_reference = self.p0
        self.p1_reference = self.p1
        self.started_reference = False
        self.percentual = 100

    def update_parameters(self, polygon, mouse_x=0, mouse_y=0, angle=0, set_reference=False, mask=None):
        polygon = compact_list(polygon)
        self.angle = angle
        if mask is not None:
            pts = find_line_endpoints((mouse_x, mouse_y), angle, mask)
            # pts = find_intersection_points((mouse_x, mouse_y), polygon, angle=angle, line_length=1500)
        else:
            pts = find_intersection_points((mouse_x, mouse_y), polygon, angle=angle, line_length=1500)
        if pts is not (0, 0) and len(pts) == 2:
            if not set_reference:
                self.p0, self.p1 = pts
                # print(self.p0, self.p1)
                delta_x = self.p1[0] - self.p0[0]
                delta_y = self.p1[1] - self.p0[1]
                self.size = math.sqrt(delta_x**2 + delta_y**2)
                self.percentual = 100 * self.size/self.reference_size
            if set_reference or not self.started_reference:
                self.p0_reference, self.p1_reference = pts
                # print(self.p0, self.p1)
                delta_x = self.p1_reference[0] - self.p0_reference[0]
                delta_y = self.p1_reference[1] - self.p0_reference[1]
                self.reference_size = math.sqrt(delta_x**2 + delta_y**2)
                self.started_reference = True

    def show(self, frame, plot_mask=False, show_reference=False):
        p0 = (int(self.p0[0]), int(self.p0[1]))
        p1 = (int(self.p1[0]), int(self.p1[1]))
        clr = (30, 120, 30) if not plot_mask else (165, 255, 165)
        cv2.line(frame, p0, p1, clr, 4)
        if show_reference:
            self.showReference(frame)
    
    def showReference(self, frame):
        if self.started_reference:
            p0 = (int(self.p0_reference[0]), int(self.p0_reference[1]))
            p1 = (int(self.p1_reference[0]), int(self.p1_reference[1]))
            clr = (0, 0, 255)
            cv2.line(frame, p0, p1, clr, 6)

def find_line_intersection(m, n, point, angle):
    x, y = point

    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)

    # Calculate the slope of the line
    slope = np.tan(angle_rad)

    # Calculate the intersection points with the matrix boundaries
    if angle % 180 != 90:  # Avoid vertical lines
        x1 = 0
        y1 = int(y - slope * x)
        x2 = n - 1
        y2 = int(y + slope * (x2 - x))
    else:
        x1 = x
        y1 = 0
        x2 = x
        y2 = m - 1

    return (x1, y1), (x2, y2)

def signum(f):
    if f > 0:
        return 1
    elif f < 0:
        return -1
    else:
        return 0

def bresenham_line(x0, y0, x1, y1, largura = 1000, altura = 1000):
    points = []

    x, y = x0, y0

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    # Determine the sign of the direction
    sx = int(signum(x1 - x0))
    sy = int(signum(y1 - y0))

    inter = False

    if dy > dx:
        t = dx
        dx = dy
        dy = t
        inter = True

    # Initial decision parameter
    e = 2 * dy - dx
    a = 2 * dy
    b = 2 * dy - 2 * dx

    for i in range(1, dx + 1):
        # Add the current point

        if e < 0:
            if inter:
                y += sy
            else:
                x += sx
            e += a
        else:
            y += sy
            x += sx
            e += b
        if x < largura and y < altura:
            points.append((x, y))

    return points

def find_bounding_polygon(contours, nL=4, epsilon_factor=0.02):
    if not contours or nL < 4:
        return None

    # Find the convex hull of the contours
    hull = cv2.convexHull(np.vstack(contours))

    # Calculate the perimeter of the convex hull
    perimeter = cv2.arcLength(hull, True)

    # Initialize epsilon based on the desired number of sides
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Keep reducing epsilon until we get a valid polygon with exactly 4 sides
    while len(approx) != nL and epsilon > 1:
        epsilon *= 0.9  # Reduce epsilon by 10%
        approx = cv2.approxPolyDP(hull, epsilon, True)

    # Return the coordinates of the vertices as a list if it has exactly 4 sides
    return approx.tolist() if len(approx) == nL else None


def point_inside_polygon(point, polygon):
    # Convert the point and polygon to the appropriate format
    point = tuple(point)
    polygon = np.array(polygon, dtype=np.int32)

    # Use the pointPolygonTest function to check the position of the point relative to the polygon
    # Returns > 0 if the point is inside, 0 if it's on the edge, and < 0 if it's outside
    result = cv2.pointPolygonTest(polygon, point, measureDist=False)

    # If the result is greater than 0, the point is inside the polygon
    return result > 0

def rotate_points(points, reference_point, angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Translate the points so that the reference point is at the origin
    translated_points = points - reference_point

    # Perform the rotation
    rotated_x = translated_points[:, 0] * np.cos(angle_radians) - translated_points[:, 1] * np.sin(angle_radians)
    rotated_y = translated_points[:, 0] * np.sin(angle_radians) + translated_points[:, 1] * np.cos(angle_radians)

    # Translate the points back to their original position
    rotated_points = np.column_stack((rotated_x, rotated_y)) + reference_point

    return rotated_points

def truncate_list(lst, value, mask):
    # Encontrar o índice do primeiro valor maior ou igual a 'value'
    start_index = next((i for i, point in enumerate(lst) if mask[point[1], point[0]] >= value), None)

    # Encontrar o índice do último valor maior ou igual a 'value' de trás para frente
    end_index = next((len(lst) - i - 1 for i, point in enumerate(reversed(lst)) if mask[point[1], point[0]] >= value), None)

    # Se não houver nenhum valor maior ou igual a 'value', retornar uma lista vazia
    if start_index is None or end_index is None:
        return []

    # Retornar a lista truncada
    return lst[start_index:end_index + 1]

def find_line_endpoints(point, angle, mask):

    altura, largura = mask.shape[:2]

    polygon_coords = [(0, 0), (largura, 0), (largura, altura), (0, altura)]
    x_value = (point[0], point[1])
    fip = find_intersection_points(x_value, polygon_coords, angle=angle, line_length=(2*max(altura, largura)))
    x0, y0 = 0, 0
    x1, y1 = 0, 0
    if len(fip) == 2:
        x0, y0 = int(fip[0][0]), int(fip[0][1])
        x1, y1 = int(fip[1][0]), int(fip[1][1])
    # print(x0, y0, x1, y1)
    values = bresenham_line(x0, y0, x1, y1, largura=largura, altura=altura)
    values = truncate_list(values, 200, mask)
    # mouse_x, mouse_y = point[0], point[1]

    # if 0 <= mouse_x < largura or 0 <= mouse_y < altura:
    #     print(mask[mouse_y][mouse_x])

    # print(len(values))
    if len(values) > 1:
        return values[0], values[-1]
    return fip


def find_intersection_points(center_point, polygon_coords, angle=90, line_length=1000):
    if len(polygon_coords) < 3:
        return (0, 0)

    angle_radians = math.radians(angle)

    x1 = center_point[0] - line_length * math.cos(angle_radians)
    y1 = center_point[1] - line_length * math.sin(angle_radians)
    x2 = center_point[0] + line_length * math.cos(angle_radians)
    y2 = center_point[1] + line_length * math.sin(angle_radians)

    line_coordinates = [(x1, y1), (x2, y2)]
    oriented_line = LineString(line_coordinates)

    polygon = Polygon(polygon_coords)

    intersection = oriented_line.intersection(polygon)

    if intersection.is_empty:
        return (0, 0)

    if isinstance(intersection, Point):
        return [(intersection.x, intersection.y)]
    elif isinstance(intersection, LineString):
        return list(intersection.coords[:2])
    elif isinstance(intersection, Polygon):
        return list(intersection.exterior.coords[:2])
    elif intersection.geom_type == 'MultiPoint':
        return [(point.x, point.y) for point in intersection]

    return (0, 0)


def find_LEFT_Points(polygon, toler=320):
    return [point for point in polygon if point[0] < toler]

def find_RIGHT_Points(polygon, toler=320):
    return [point for point in polygon if point[0] >= toler]

def find_LEFTUP_Point(polygon):
    pts = find_LEFT_Points(polygon)
    if len(pts) == 2:
        min1 = pts[0]
        min2 = pts[1]
        return min1 if min1[1] > min2[1] else min2
    return (0, 0)

def find_RIGHTUP_Point(polygon):
    pts = find_RIGHT_Points(polygon)
    if len(pts) == 2:
        max1 = pts[0]
        max2 = pts[1]
        return max1 if max1[1] > max2[1] else max2
    return (0, 0)

def find_LEFTDOWN_Point(polygon):
    pts = find_LEFT_Points(polygon)
    if len(pts) == 2:
        min1 = pts[0]
        min2 = pts[1]
        return min1 if min1[1] < min2[1] else min2
    return (0, 0)

def find_RIGHTDOWN_Point(polygon):
    pts = find_RIGHT_Points(polygon)
    if len(pts) == 2:
        max1 = pts[0]
        max2 = pts[1]
        return max1 if max1[1] < max2[1] else max2
    return (0, 0)

def find_rect_polygon(polygon):
    return [find_LEFTDOWN_Point(polygon), find_LEFTUP_Point(polygon), find_RIGHTUP_Point(polygon), find_RIGHTDOWN_Point(polygon)]

def calculate_angle_and_distance(point1, point2, radians=False):
    # Calcular as diferenças nas coordenadas x e y
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]

    # Calcular a distância euclidiana
    distance = math.sqrt(delta_x**2 + delta_y**2)

    # Calcular o ângulo em radianos
    angle_radians = math.atan2(delta_y, delta_x)

    # Converter para graus se necessário
    if not radians:
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees, distance
    else:
        return angle_radians, distance
    
def find_rect_medium_angle(polygon):
    angle0, dist0 = calculate_angle_and_distance(find_LEFTUP_Point(polygon), find_RIGHTUP_Point(polygon))
    angle1, dist1 = calculate_angle_and_distance(find_LEFTDOWN_Point(polygon), find_RIGHTDOWN_Point(polygon))
    return (angle0+angle1)/2.0

def compact_list(lst):
    return [point[0] for point in lst]


