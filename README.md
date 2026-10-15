# Automated Drone Swarm Simulation

A 2D educational Python simulation exploring how a small swarm can move toward destinations while responding to nearby drones, wind, and a shared charging station. The project builds from simple movement experiments to a repeating ten-drone demonstration.

This is a visual learning model, not flight-control software. It does not control real aircraft, model physical drone dynamics, or guarantee collision-free movement.

## What the project demonstrates

The five scripts show the project developing in stages:

1. **Colour only** — ten moving agents with distinct colours.
2. **Shape and colour** — Voronoi geometry and safe-zone visualisation.
3. **Constant force** — how an applied force changes movement.
4. **Comparison** — side-by-side movement with and without the force.
5. **Complete simulation** — ten simulated drones travel from a charging station to individual destinations, experience two wind phases, land in sequence, and return to recharge.

The final simulation is [`5_complete_collision_avoidance_project.py`](5_complete_collision_avoidance_project.py). It uses direct goal steering, local separation forces, Voronoi visualisation, speed limits, and discrete position and velocity updates. The Matplotlib animation displays trails, flight phases, wind direction, landing progress, and minimum drone separation.

## Mathematical ideas in the current code

- **Voronoi geometry:** partitions the 2D plane according to the nearest drone position and is used to visualise local regions.
- **Local separation:** nearby drones apply a repulsive steering force based on their distance.
- **Discrete motion updates:** acceleration changes velocity, then velocity changes position at each time step.
- **Wind disturbance:** time-varying forces perturb outbound and return flights.
- **Bounded speed and area:** agents are speed-limited and reflected at the simulation boundary.

The separation rule is a simple educational heuristic. The code does not constrain drones to buffered Voronoi cells, so the diagram does not provide a mathematical collision-avoidance guarantee. The simulation also does not currently implement Mean Field Game theory, HJB or Fokker–Planck equations, Nash-equilibrium solving, Dijkstra pathfinding, static-obstacle avoidance, or 3D flight planning. These could be explored as future extensions.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib
- SciPy

## Run the final simulation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the animation:

```bash
python -m pip install -r requirements.txt
python 5_complete_collision_avoidance_project.py
```

Run any numbered script to explore an earlier stage.

## Project files

| File | Purpose |
| --- | --- |
| `1_colour_only.py` | Initial movement and colour experiment |
| `2_shape_and_colour.py` | Shape, colour, and Voronoi visualisation |
| `3_constant_force.py` | Movement under a constant force |
| `4_comparison.py` | Compare movement with and without force |
| `5_complete_collision_avoidance_project.py` | Full repeating swarm demonstration |
| `requirements.txt` | Python package dependencies |

