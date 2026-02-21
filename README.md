# Resident Problem

## Problem Definition

The resident problem is a scheduling problem where we have a set of residents. We need to find a solution which satisfies the following constraints:

1. There are always at least **N** residents in the hospital
2. Between shifts, there must be at least **M** hours of rest
3. A resident cannot work more than **K** hours in a row
4. A resident cannot work more than **L** hours in a week
5. There are teaching hours between **T₁** and **T₂**; each resident must work at least **P** hours during this time
6. Each shift must be at least **S** hours long (no micro-shifts)
7. Each resident must have at least **D** full day(s) off (24 consecutive hours without work) per week

Fairness can be yielded by cycling the residents through the different schedules.

## Solution Approach

### Model

Time is discretized into hourly slots over a planning horizon of `n_hours` hours.
The core decision variable is:

```
x[r][t] ∈ {0, 1}   — 1 if resident r is working at hour t, else 0
```

Each problem constraint maps directly to a linear inequality over these binary variables:

| Constraint | Linear form |
|---|---|
| ≥ N on-duty at all times | `Σ_r x[r][t] ≥ N` for every hour `t` |
| ≥ M rest hours between shifts | `x[r][t+s] + x[r][t] − x[r][t+1] ≤ 1` for `s ∈ [2, M]` |
| ≤ K consecutive working hours | `Σ_{s=0}^{K} x[r][t+s] ≤ K` for every `t` |
| ≤ L hours per 168-hour week | `Σ_{week} x[r][t] ≤ L` |
| ≥ P teaching hours in [T₁, T₂] | `Σ_{t=T₁}^{T₂} x[r][t] ≥ P` for every resident |
| ≥ S hours per shift | `Σ_{s=0}^{S-1} x[r][t+s] ≥ S · (x[r][t] − x[r][t−1])` on every shift start `t` |
| ≥ D full days off per week | boolean `day_off[r][t]` indicators; `Σ_t day_off[r][t] ≥ D`, `day_off[r][t]=1 ⟹ x[r][t+s]=0` for `s ∈ [0,23]` |

**Shift-start detection:** the expression `x[r][t] − x[r][t−1]` equals 1 exactly
when a new shift begins at hour `t`, 0 during a continuing shift, and −1 when a
shift ends.  Multiplying the minimum-length requirement by this expression means
the constraint is active only on shift starts and is trivially satisfied otherwise.
New shifts may not start within `S` hours of the horizon end (a shift there could
not be completed), so those positions are blocked with `x[r][t] ≤ x[r][t−1]`.

**Day-off encoding:** one boolean variable `day_off[r][t]` is introduced for every
valid 24-hour window start `t` within each 168-hour week.  Setting it to 1 forces
all 24 following hours to be free via CP-SAT implication constraints.  Requiring
at least `D` of these variables to be 1 per (resident, week) pair enforces the
minimum rest guarantee.

The rest constraint derivation: when a shift ends at hour `t`
(`x[r][t]=1, x[r][t+1]=0`), the expression `x[r][t] − x[r][t+1]` equals 1,
which forces `x[r][t+s] = 0` for each `s` in the range. Hour `t+1` is already
guaranteed non-working by the shift-end condition, so only `s ∈ [2, M]` need
explicit constraints.

**Objective:** minimize total working hours (produces the most relaxed feasible schedule).

### Parameters summary

| Parameter | Symbol | Meaning |
|---|---|---|
| `min_on_duty` | N | Minimum residents on duty at every hour |
| `min_rest` | M | Minimum rest hours between any two shifts |
| `max_consecutive` | K | Maximum consecutive working hours |
| `max_weekly` | L | Maximum working hours in any 168-hour week |
| `teaching_hours` / `min_teaching` | T₁,T₂,P | Teaching window and minimum attendance |
| `min_shift_length` | S | Minimum duration of any single shift |
| `min_days_off_per_week` | D | Minimum full 24-hour rest days per resident per week |

### Solver

The model is solved with **Google OR-Tools CP-SAT**, a state-of-the-art
constraint-programming / SAT-based integer solver that handles binary variables
and linear constraints natively.

### File Structure

```
residentProblem/
├── model.py            # SchedulingParams and Schedule data classes
├── solver.py           # CP-SAT constraint model and solve() entry point
├── visualizer.py       # Matplotlib heatmap + constraint-check summary
├── main.py             # Example run (10 residents, 1 week)
├── requirements.txt
└── tests/
    └── test_solver.py  # Pytest suite covering all seven constraints
```

## Usage

### Install dependencies

```
pip install -r requirements.txt
```

### Run the example

```
python main.py
```

### Custom problem

```python
from model import SchedulingParams
from solver import solve
from visualizer import print_summary, plot_schedule

params = SchedulingParams(
    n_residents=10,
    n_hours=336,        # 2 weeks
    min_on_duty=3,
    min_rest=8,
    max_consecutive=12,
    max_weekly=60,
    teaching_hours=tuple(range(50, 91)),
    min_teaching=5,
    min_shift_length=6,
    min_days_off_per_week=1,
)

schedule = solve(params, time_limit_seconds=120.0)
if schedule:
    print_summary(schedule)
    plot_schedule(schedule)
```

### Run tests

```
pytest tests/
```
```

Now let me verify the final file structure:
