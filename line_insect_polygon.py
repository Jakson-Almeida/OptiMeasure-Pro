import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry

def plot_polygon(ax, polygon, fc='blue', alpha=0.5):
    if isinstance(polygon, shapely.geometry.MultiPolygon):
        for poly in polygon:
            plot_polygon(ax, poly, fc, alpha)
    elif isinstance(polygon, shapely.geometry.Polygon):
        patch = plt.Polygon(np.array(polygon.exterior.xy).T, fc=fc, alpha=alpha)
        ax.add_patch(patch)

circle = shapely.geometry.Point(5.0, 0.0).buffer(10.0)
clip_poly = shapely.geometry.Polygon([[-9.5, -2], [2, 2], [3, 4], [-1, 3]])
clipped_shape = circle.difference(clip_poly)

line = shapely.geometry.LineString([[-10, -5], [15, 5]])
line2 = shapely.geometry.LineString([[-10, -5], [-5, 0], [2, 3]])

print('Blue line intersects clipped shape:', line.intersects(clipped_shape))
print('Green line intersects clipped shape:', line2.intersects(clipped_shape))

fig, ax = plt.subplots()

ax.plot(*np.array(line.xy), color='blue', linewidth=3, solid_capstyle='round')
ax.plot(*np.array(line2.xy), color='green', linewidth=3, solid_capstyle='round')

plot_polygon(ax, clipped_shape, fc='blue', alpha=0.5)

ax.axis('equal')
plt.show()
