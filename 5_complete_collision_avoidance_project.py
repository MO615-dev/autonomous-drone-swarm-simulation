import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import Voronoi, voronoi_plot_2d

# ============================================================
# Dynamic Voronoi-Based Path Planning for Autonomous Drone Swarms
# 10th-grade mathematics project simulation
#
# Repeating demonstration sequence:
# charging station outside the box -> takeoff -> Phase 1 -> Phase 2
# -> Phase 3 wind -> Phase 4 landing -> windy return to the station
# ============================================================

# ---------------- PARAMETERS ----------------
NUM_DRONES = 10
MAX_X = 10
MAX_Y = 10
DT = 0.08

MAX_SPEED = 0.65
GOAL_FORCE = 0.85
RETURN_FORCE = 1.20
AVOID_FORCE = 1.35
WIND_FORCE = 1.00
DAMPING = 0.96

SAFE_DISTANCE = 1.15
LANDING_RADIUS = 0.35
TRAIL_LENGTH = 45
FRAME_INTERVAL_MS = 45

STATION_START_END = 40
TAKEOFF_END = 140
PHASE_1_END = 260
PHASE_2_END = 400
PHASE_3_END = 540
PHASE_4_END = 740
RETURN_WIND_END = PHASE_4_END + 180
CHARGING_END = RETURN_WIND_END + 100
CYCLE_FRAMES = CHARGING_END + 40
TOTAL_FRAMES = CYCLE_FRAMES

np.random.seed(8)

# ---------------- INITIALIZATION ----------------
initial_positions = np.random.rand(NUM_DRONES, 2) * [MAX_X, MAX_Y]
initial_velocities = (np.random.rand(NUM_DRONES, 2) - 0.5) * 0.8
positions = initial_positions.copy()
velocities = initial_velocities.copy()

# Destinations are placed on the opposite side of the box.
destinations = np.column_stack(
    [
        MAX_X - initial_positions[:, 0],
        MAX_Y - initial_positions[:, 1],
    ]
)

# The simulation box is 0..10 on both axes. The station is deliberately below it.
CHARGING_STATION = np.array([MAX_X / 2, -2.5])
slot_angles = np.linspace(0, 2 * np.pi, NUM_DRONES, endpoint=False)
charging_slots = CHARGING_STATION + 0.7 * np.column_stack(
    [np.cos(slot_angles), np.sin(slot_angles)]
)

landed = np.zeros(NUM_DRONES, dtype=bool)
at_charging_station = np.zeros(NUM_DRONES, dtype=bool)
landing_order = list(range(NUM_DRONES))
next_landing_index = 0
trails = [[positions[i].copy()] for i in range(NUM_DRONES)]
last_cycle_frame = None

colors = plt.cm.tab10(np.arange(NUM_DRONES) % 10)

PHASE_EXPLANATIONS = {
    0: "Why: all drones begin together below the 0-10 simulation box.",
    1: "Why: the drones leave the station and move into the simulation box.",
    2: "Why: the goal force guides each drone to its destination; no Voronoi yet.",
    3: "Why: Voronoi cells and safe zones now help nearby drones avoid collisions.",
    4: "Why: a strong wind moves the drones away from their original tracks.",
    5: "Why: the wind stops, so the drones return to their destinations and land.",
    6: "Why: after landing, drones immediately return with Voronoi safety and wind.",
    7: "Why: all drones have reached the station below the simulation box.",
    8: "Why: the next cycle begins again with takeoff.",
}


def limit_speed(vector, max_speed):
    speed = np.linalg.norm(vector)
    if speed > max_speed:
        return vector / speed * max_speed
    return vector


def reset_simulation():
    global positions, velocities, next_landing_index, trails

    positions = charging_slots.copy()
    velocities = np.zeros_like(charging_slots)
    landed[:] = False
    at_charging_station[:] = True
    next_landing_index = 0
    trails = [[positions[i].copy()] for i in range(NUM_DRONES)]


def current_phase(frame):
    cycle_frame = frame % CYCLE_FRAMES

    if cycle_frame < STATION_START_END:
        return 0, "Charging station: all drones ready"
    if cycle_frame < TAKEOFF_END:
        return 1, "Takeoff: drones leave the charging station"
    if cycle_frame < PHASE_1_END:
        return 2, "Phase 1: Direct path planning"
    if cycle_frame < PHASE_2_END:
        return 3, "Phase 2: Voronoi diagram + safe zones"
    if cycle_frame < PHASE_3_END:
        return 4, "Phase 3: Strong wind disturbance"
    if cycle_frame < PHASE_4_END:
        return 5, "Phase 4: Wind stops; safe landing"
    if cycle_frame < RETURN_WIND_END:
        return 6, "Return flight: Voronoi + wind toward charging station"
    if cycle_frame < CHARGING_END:
        return 7, "Charging station: drones recharge"
    return 8, "Cycle restart: ready for takeoff"


def wind_vector(frame):
    # Outbound wind is active only during Phase 3.
    cycle_frame = frame % CYCLE_FRAMES

    if cycle_frame < PHASE_2_END or cycle_frame >= PHASE_3_END:
        return np.array([0.0, 0.0])

    wind_progress = (cycle_frame - PHASE_2_END) / (PHASE_3_END - PHASE_2_END)
    gust_envelope = np.sin(np.pi * wind_progress)
    wind_angle = 0.9 + 0.55 * np.sin(cycle_frame * 0.07)
    direction = np.array([np.cos(wind_angle), np.sin(wind_angle)])
    return WIND_FORCE * gust_envelope * direction


def return_wind_vector(frame):
    # A second wind disturbance affects the flight back to the station.
    cycle_frame = frame % CYCLE_FRAMES

    if cycle_frame < PHASE_4_END or cycle_frame >= RETURN_WIND_END:
        return np.array([0.0, 0.0])

    wind_progress = (cycle_frame - PHASE_4_END) / (
        RETURN_WIND_END - PHASE_4_END
    )
    gust_envelope = np.sin(np.pi * wind_progress)
    wind_angle = -0.8 + 0.6 * np.sin(cycle_frame * 0.08)
    direction = np.array([np.cos(wind_angle), np.sin(wind_angle)])
    return 0.85 * gust_envelope * direction


def goal_acceleration(drone_id, force=GOAL_FORCE):
    direction = destinations[drone_id] - positions[drone_id]
    distance = np.linalg.norm(direction)

    if distance < 0.001:
        return np.array([0.0, 0.0])

    return force * direction / distance


def avoidance_acceleration(drone_id):
    push = np.array([0.0, 0.0])

    for other_id in range(NUM_DRONES):
        if other_id == drone_id or landed[other_id]:
            continue

        difference = positions[drone_id] - positions[other_id]
        distance = np.linalg.norm(difference)

        if 0.001 < distance < SAFE_DISTANCE:
            # The closer two drones are, the stronger the push apart.
            strength = (SAFE_DISTANCE - distance) / SAFE_DISTANCE
            push += (difference / distance) * strength

    return AVOID_FORCE * push


def append_trails():
    for drone_id in range(NUM_DRONES):
        trails[drone_id].append(positions[drone_id].copy())
        if len(trails[drone_id]) > TRAIL_LENGTH:
            trails[drone_id].pop(0)


def update_preflight(frame, phase_number):
    cycle_frame = frame % CYCLE_FRAMES

    if phase_number == 0:
        positions[:] = charging_slots
        velocities[:] = 0.0
        at_charging_station[:] = True
    else:
        progress = (cycle_frame - STATION_START_END) / (TAKEOFF_END - STATION_START_END)
        progress = np.clip(progress, 0.0, 1.0)
        positions[:] = charging_slots + (initial_positions - charging_slots) * progress
        velocities[:] = 0.0
        at_charging_station[:] = False

    append_trails()


def update_post_landing(frame, phase_number):
    cycle_frame = frame % CYCLE_FRAMES

    if phase_number == 6:
        progress = (cycle_frame - PHASE_4_END) / (RETURN_WIND_END - PHASE_4_END)
        progress = np.clip(progress, 0.0, 1.0)
        base_positions = destinations + (charging_slots - destinations) * progress
        positions[:] = base_positions + 0.55 * return_wind_vector(frame)
        velocities[:] = 0.0
        landed[:] = False
        at_charging_station[:] = False
    else:
        positions[:] = charging_slots
        velocities[:] = 0.0
        at_charging_station[:] = True

    append_trails()


def update_physics(frame):
    global positions, velocities, next_landing_index

    cycle_frame = frame % CYCLE_FRAMES
    phase_number, _ = current_phase(frame)

    if phase_number in (0, 1):
        update_preflight(frame, phase_number)
        return

    if phase_number >= 6:
        update_post_landing(frame, phase_number)
        return

    active_wind = wind_vector(frame)

    for drone_id in range(NUM_DRONES):
        if landed[drone_id]:
            velocities[drone_id] = np.array([0.0, 0.0])
            positions[drone_id] = destinations[drone_id]
            continue

        goal_force = RETURN_FORCE if phase_number >= 5 else GOAL_FORCE
        acceleration = goal_acceleration(drone_id, goal_force)

        if phase_number >= 3:
            acceleration += avoidance_acceleration(drone_id)

        acceleration += active_wind

        velocities[drone_id] += acceleration * DT
        velocities[drone_id] *= DAMPING
        velocities[drone_id] = limit_speed(velocities[drone_id], MAX_SPEED)
        positions[drone_id] += velocities[drone_id] * DT

        # Bounce away from walls.
        if positions[drone_id, 0] < 0 or positions[drone_id, 0] > MAX_X:
            velocities[drone_id, 0] *= -0.75
            positions[drone_id, 0] = np.clip(positions[drone_id, 0], 0, MAX_X)

        if positions[drone_id, 1] < 0 or positions[drone_id, 1] > MAX_Y:
            velocities[drone_id, 1] *= -0.75
            positions[drone_id, 1] = np.clip(positions[drone_id, 1], 0, MAX_Y)

    append_trails()

    # Landing starts only after the wind phase, and only one drone can land at a time.
    if phase_number >= 5 and next_landing_index < NUM_DRONES:
        drone_to_land = landing_order[next_landing_index]
        distance_to_goal = np.linalg.norm(positions[drone_to_land] - destinations[drone_to_land])

        if distance_to_goal < LANDING_RADIUS:
            landed[drone_to_land] = True
            positions[drone_to_land] = destinations[drone_to_land]
            velocities[drone_to_land] = np.array([0.0, 0.0])
            next_landing_index += 1


def minimum_drone_distance():
    smallest_distance = float("inf")

    for i in range(NUM_DRONES):
        for j in range(i + 1, NUM_DRONES):
            distance = np.linalg.norm(positions[i] - positions[j])
            smallest_distance = min(smallest_distance, distance)

    return smallest_distance


def draw_voronoi(ax):
    # Voronoi needs unique points. Tiny jitter prevents rare duplicate-point errors.
    jitter = np.random.normal(scale=1e-6, size=positions.shape)
    vor = Voronoi(positions + jitter)
    voronoi_plot_2d(
        vor,
        ax=ax,
        show_vertices=False,
        show_points=False,
        line_colors="royalblue",
        line_width=1.1,
        line_alpha=0.65,
    )


def draw_scene(frame):
    ax.clear()
    phase_number, phase_title = current_phase(frame)
    cycle_frame = frame % CYCLE_FRAMES

    ax.set_xlim(-1.0, MAX_X + 1.0)
    ax.set_ylim(CHARGING_STATION[1] - 1.0, MAX_Y + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(phase_title)

    simulation_box = plt.Rectangle(
        (0, 0),
        MAX_X,
        MAX_Y,
        fill=False,
        edgecolor="black",
        linewidth=1.2,
        linestyle="--",
        alpha=0.65,
    )
    ax.add_patch(simulation_box)
    ax.text(0.2, 0.25, "Simulation box: x,y = 0 to 10", fontsize=8, alpha=0.7)

    if phase_number in (3, 4, 5, 6):
        draw_voronoi(ax)

    # Safe-distance circles
    if phase_number in (3, 4, 5, 6):
        for drone_id in range(NUM_DRONES):
            circle = plt.Circle(
                positions[drone_id],
                SAFE_DISTANCE / 2,
                color=colors[drone_id],
                fill=False,
                alpha=0.18,
                linestyle="--",
            )
            ax.add_patch(circle)

    # Destination markers
    ax.scatter(
        destinations[:, 0],
        destinations[:, 1],
        marker="*",
        s=170,
        c=colors,
        edgecolors="black",
        linewidths=0.7,
        label="Destinations",
    )

    # The central charging station is shown at the start and during the return flight.
    if phase_number in (0, 1, 6, 7, 8):
        station_circle = plt.Circle(
            CHARGING_STATION,
            0.95,
            color="limegreen",
            alpha=0.18,
            fill=True,
            linestyle="--",
        )
        ax.add_patch(station_circle)
        ax.scatter(
            CHARGING_STATION[0],
            CHARGING_STATION[1],
            marker="P",
            s=260,
            c="limegreen",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
        ax.text(
            CHARGING_STATION[0] - 0.9,
            CHARGING_STATION[1] - 1.25,
            "Charging station (outside box)",
            color="darkgreen",
            fontsize=9,
            weight="bold",
        )

    # Trails
    for drone_id in range(NUM_DRONES):
        trail = np.array(trails[drone_id])
        ax.plot(trail[:, 0], trail[:, 1], color=colors[drone_id], alpha=0.35, linewidth=1.5)

    # Drones
    for drone_id in range(NUM_DRONES):
        if phase_number in (1, 6):
            marker = "^"
        elif phase_number in (0, 7, 8):
            marker = "s"
        elif phase_number == 5 and landed[drone_id]:
            marker = "s"
        else:
            marker = "o"

        size = 120 if marker == "s" else 90
        label = f"D{drone_id}"

        ax.scatter(
            positions[drone_id, 0],
            positions[drone_id, 1],
            marker=marker,
            s=size,
            c=[colors[drone_id]],
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
        )
        ax.text(
            positions[drone_id, 0] + 0.08,
            positions[drone_id, 1] + 0.08,
            label,
            fontsize=8,
            weight="bold",
        )

    # Wind arrow and return message
    outbound_wind = wind_vector(frame)
    return_wind = return_wind_vector(frame)
    wind = outbound_wind if np.linalg.norm(outbound_wind) > 0.001 else return_wind
    if np.linalg.norm(wind) > 0.001:
        ax.arrow(
            7.4,
            8.9,
            wind[0] * 1.4,
            wind[1] * 1.4,
            head_width=0.18,
            color="darkorange",
            length_includes_head=True,
        )
        wind_label = "Outbound wind" if phase_number == 4 else "Return wind"
        ax.text(7.4, 9.55, wind_label, color="darkorange", fontsize=10, weight="bold")
    elif phase_number == 5:
        ax.text(7.4, 9.55, "Wind stopped: landing", color="forestgreen", fontsize=9, weight="bold")

    min_distance = minimum_drone_distance()
    landed_count = int(np.sum(landed))
    charging_count = int(np.sum(at_charging_station))
    info = (
        f"Cycle frame: {cycle_frame}/{CYCLE_FRAMES - 1}\n"
        f"{PHASE_EXPLANATIONS[phase_number]}\n"
        f"Minimum drone distance: {min_distance:.2f}\n"
        f"Safe distance: {SAFE_DISTANCE:.2f}\n"
        f"Landed: {landed_count}/{NUM_DRONES}\n"
        f"At charging station: {charging_count}/{NUM_DRONES}"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
    )


def update(frame):
    global last_cycle_frame

    cycle_frame = frame % CYCLE_FRAMES
    if cycle_frame == 0 and last_cycle_frame not in (None, 0):
        reset_simulation()

    last_cycle_frame = cycle_frame
    update_physics(cycle_frame)
    draw_scene(cycle_frame)


reset_simulation()
fig, ax = plt.subplots(figsize=(8.5, 8.5))
ani = animation.FuncAnimation(
    fig,
    update,
    frames=TOTAL_FRAMES,
    interval=FRAME_INTERVAL_MS,
    repeat=True,
)

plt.tight_layout()
plt.show()
