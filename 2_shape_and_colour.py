import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PolyCollection
from scipy.spatial import Voronoi

NUM_DOTS = 10
BOX_SIZE = 1.0
SPEED = 0.01

positions = np.random.rand(NUM_DOTS, 2)
velocities = (np.random.rand(NUM_DOTS, 2) - 0.5) * SPEED
colors = plt.cm.tab10(np.arange(NUM_DOTS) % 10)


def clip_polygon_to_box(polygon):
    def clip(points, inside, intersection):
        if len(points) == 0:
            return []

        clipped = []
        previous = points[-1]
        previous_inside = inside(previous)

        for current in points:
            current_inside = inside(current)

            if current_inside:
                if not previous_inside:
                    clipped.append(intersection(previous, current))
                clipped.append(current)
            elif previous_inside:
                clipped.append(intersection(previous, current))

            previous = current
            previous_inside = current_inside

        return clipped

    def vertical_intersection(x_value):
        def intersection(a, b):
            if b[0] == a[0]:
                return a
            t = (x_value - a[0]) / (b[0] - a[0])
            return a + t * (b - a)
        return intersection

    def horizontal_intersection(y_value):
        def intersection(a, b):
            if b[1] == a[1]:
                return a
            t = (y_value - a[1]) / (b[1] - a[1])
            return a + t * (b - a)
        return intersection

    points = [np.array(point, dtype=float) for point in polygon]
    points = clip(points, lambda p: p[0] >= 0, vertical_intersection(0))
    points = clip(points, lambda p: p[0] <= BOX_SIZE, vertical_intersection(BOX_SIZE))
    points = clip(points, lambda p: p[1] >= 0, horizontal_intersection(0))
    points = clip(points, lambda p: p[1] <= BOX_SIZE, horizontal_intersection(BOX_SIZE))
    return points


def finite_voronoi_polygons(vor):
    regions = []
    vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    radius = BOX_SIZE * 4

    ridges = {}
    for (point_a, point_b), (vertex_a, vertex_b) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges.setdefault(point_a, []).append((point_b, vertex_a, vertex_b))
        ridges.setdefault(point_b, []).append((point_a, vertex_a, vertex_b))

    for point_index, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]

        if all(vertex >= 0 for vertex in region):
            regions.append(region)
            continue

        new_region = [vertex for vertex in region if vertex >= 0]

        for neighbor, vertex_a, vertex_b in ridges[point_index]:
            if vertex_b < 0:
                vertex_a, vertex_b = vertex_b, vertex_a
            if vertex_a >= 0:
                continue

            tangent = vor.points[neighbor] - vor.points[point_index]
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[point_index, neighbor]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[vertex_b] + direction * radius

            new_region.append(len(vertices))
            vertices.append(far_point.tolist())

        polygon_vertices = np.asarray([vertices[vertex] for vertex in new_region])
        polygon_center = polygon_vertices.mean(axis=0)
        angles = np.arctan2(
            polygon_vertices[:, 1] - polygon_center[1],
            polygon_vertices[:, 0] - polygon_center[0],
        )
        regions.append(np.asarray(new_region)[np.argsort(angles)].tolist())

    return regions, np.asarray(vertices)


def voronoi_cells(points):
    vor = Voronoi(points)
    regions, vertices = finite_voronoi_polygons(vor)
    polygons = []
    color_indices = []
    segments = []

    for point_index, region in enumerate(regions):
        polygon = clip_polygon_to_box(vertices[region])
        if len(polygon) < 3:
            continue

        polygon = np.asarray(polygon)
        polygons.append(polygon)
        color_indices.append(point_index)

        closed = np.vstack([polygon, polygon[0]])
        for start, end in zip(closed[:-1], closed[1:]):
            if np.linalg.norm(end - start) > 0.0001:
                segments.append([start, end])

    return polygons, color_indices, segments


fig, ax = plt.subplots()
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
ax.set_aspect("equal")
ax.set_title("Condition 2: Shape and Colour")
ax.grid(True, alpha=0.25)

polygons, color_indices, segments = voronoi_cells(positions)
blocks = PolyCollection(polygons, facecolors=colors[color_indices], edgecolors="none", alpha=0.25)
lines = LineCollection(segments, colors="black", linewidths=1.2, alpha=0.65)
dots = ax.scatter(positions[:, 0], positions[:, 1], s=90, c=colors, edgecolors="black", linewidths=0.7, zorder=3)
ax.add_collection(blocks)
ax.add_collection(lines)


def update(frame):
    global positions, velocities

    positions += velocities

    hit_x = (positions[:, 0] <= 0) | (positions[:, 0] >= BOX_SIZE)
    hit_y = (positions[:, 1] <= 0) | (positions[:, 1] >= BOX_SIZE)

    velocities[hit_x, 0] *= -1
    velocities[hit_y, 1] *= -1

    positions = np.clip(positions, 0, BOX_SIZE)
    polygons, color_indices, segments = voronoi_cells(positions)
    blocks.set_verts(polygons)
    blocks.set_facecolors(colors[color_indices])
    blocks.set_alpha(0.25)
    lines.set_segments(segments)
    dots.set_offsets(positions)

    return blocks, lines, dots


animation = FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)
plt.show()
