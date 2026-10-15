# Automated Drone Swarm Simulation

A visual Python project that explores how a group of simulated drones can move toward goals while responding to nearby drones, changing wind, and a shared charging station.

This is a 2D educational simulation. It does not connect to, control, or certify real aircraft, and its avoidance rules do not guarantee collision-free flight.

## What it demonstrates

The project grows through five experiments:

1. **Colour only** — ten moving agents with distinct colours.
2. **Shape and colour** — Voronoi regions and safe-zone visualisation.
3. **Constant force** — how an applied force changes movement.
4. **Comparison** — side-by-side movement with and without the force.
5. **Complete simulation** — ten drones travel from a charging station to individual destinations, use Voronoi-based separation, experience wind, land in sequence, and return to recharge.

The final simulation is [`5_complete_collision_avoidance_project.py`](5_complete_collision_avoidance_project.py). It displays an animated Matplotlib window and repeats its demonstration cycle.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib
- SciPy

## Run it

Create and activate a virtual environment, then install the dependencies:

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

Install the packages and launch the final simulation:

```bash
python -m pip install -r requirements.txt
python 5_complete_collision_avoidance_project.py
```

To explore an earlier experiment, run its numbered Python file with the same environment.

## How the final simulation works

- Ten agents move in a bounded 2D area toward destinations opposite their starting positions.
- Goal-directed acceleration moves each agent toward its destination.
- Voronoi geometry and a safe-distance rule influence separation between nearby agents.
- Two wind phases perturb outbound and return flights.
- The animation shows flight phases, trails, wind direction, landing progress, charging, and minimum separation.

The parameters near the top of the final script control the number of drones, force strengths, safe distance, timing, and animation speed. The model uses simplified point-like agents and should be treated as a learning visualisation rather than a validated flight-planning system.

## Project structure

| File | Purpose |
| --- | --- |
| `1_colour_only.py` | Initial movement and colour experiment |
| `2_shape_and_colour.py` | Shape, colour, and Voronoi visualisation |
| `3_constant_force.py` | Movement under a constant force |
| `4_comparison.py` | Compare movement with and without force |
| `5_complete_collision_avoidance_project.py` | Full repeating swarm demonstration |
| `requirements.txt` | Python package dependencies |

## Development

The numbered scripts preserve the project's progression. Changes to the final demonstration should be made in the fifth script, with earlier experiments kept as reference steps.
