import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NUM_DOTS = 10
BOX_SIZE = 1.0
SPEED = 0.01

positions = np.random.rand(NUM_DOTS, 2)
velocities = (np.random.rand(NUM_DOTS, 2) - 0.5) * SPEED
colors = plt.cm.tab10(np.arange(NUM_DOTS) % 10)

fig, ax = plt.subplots()
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
ax.set_aspect("equal")
ax.set_title("Condition 1: Colour Only")
ax.grid(True, alpha=0.25)

dots = ax.scatter(
    positions[:, 0],
    positions[:, 1],
    s=90,
    c=colors,
    edgecolors="black",
    linewidths=0.7,
)


def update(frame):
    global positions, velocities

    positions += velocities

    hit_x = (positions[:, 0] <= 0) | (positions[:, 0] >= BOX_SIZE)
    hit_y = (positions[:, 1] <= 0) | (positions[:, 1] >= BOX_SIZE)

    velocities[hit_x, 0] *= -1
    velocities[hit_y, 1] *= -1

    positions = np.clip(positions, 0, BOX_SIZE)
    dots.set_offsets(positions)

    return (dots,)


animation = FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)
plt.show()
