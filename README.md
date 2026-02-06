# quad_trajectories

A ROS 2 Python library of quadrotor trajectory definitions built on JAX. Trajectories return position-level outputs `[x, y, z, yaw]` — all higher-order derivatives are computed on demand using JAX's forward-mode autodiff (`jacfwd`).

## Available Trajectories

| Type | CLI value | Description |
|------|-----------|-------------|
| Hover | `hover` | Stationary hover with 8 sub-modes (altitude, position combos) |
| Yaw Only | `yaw_only` | Hold position while spinning in yaw |
| Circle (Horizontal) | `circle_horz` | Circular path in the XY plane |
| Circle (Vertical) | `circle_vert` | Circular path in the XZ plane |
| Figure-8 (Horizontal) | `fig8_horz` | Lemniscate in the XY plane |
| Figure-8 (Vertical) | `fig8_vert` | Lemniscate in the XZ plane (supports `--short` variant) |
| Helix | `helix` | Spiral ascending and descending |
| Sawtooth | `sawtooth` | Waypoint-based sawtooth pattern |
| Triangle | `triangle` | Waypoint-based triangular pattern |

## Key Features

- **Position-only design** — trajectories define only `[x, y, z, yaw]`; controllers compute velocity, acceleration, jerk, etc. via `jacfwd()`
- **JAX JIT-compiled** — all trajectory functions are JIT-compiled for real-time performance
- **Registry pattern** — `TRAJ_REGISTRY` maps `TrajectoryType` enum values to trajectory callables
- **Context-aware** — `TrajContext` controls sim/hardware mode, hover mode, spin enable, double speed, and short variants

## Usage

```python
from quad_trajectories import TRAJ_REGISTRY, TrajectoryType, TrajContext

ctx = TrajContext(sim=True, spin=True, double_speed=False)
traj_fn = TRAJ_REGISTRY[TrajectoryType.HELIX]

# Get [x, y, z, yaw] at time t
pos = traj_fn(t, ctx)
```

Derivatives are typically computed by the controller using utility functions in `quad_trajectories.utils`, which wrap `jax.jacfwd` to produce velocity, acceleration, and lookahead horizons.

## Package Structure

```
quad_trajectories/
├── __init__.py          # Public API exports
├── core.py              # All trajectory implementations
├── registry.py          # TrajectoryType enum → function mapping
├── types.py             # TrajContext dataclass, TrajectoryType enum
├── utils.py             # Derivative helpers and horizon generation
└── jax_utils.py         # JAX JIT configuration
```

## Installation

```bash
# Inside a ROS 2 workspace src/ directory
git clone git@github.com:evannsm/quad_trajectories.git
cd .. && colcon build --symlink-install
```

## License

MIT
