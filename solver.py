from model import Schedule, SchedulingParams
from ortools.sat.python import cp_model

_HOURS_PER_WEEK = 168
_HOURS_PER_DAY = 24


def _add_min_coverage_constraint(
    model: cp_model.CpModel,
    x: list[list[cp_model.IntVar]],
    n_residents: int,
    n_hours: int,
    min_on_duty: int,
) -> None:
    """At least min_on_duty residents must be on duty at every hour."""
    for t in range(n_hours):
        model.add(sum(x[r][t] for r in range(n_residents)) >= min_on_duty)


def _add_min_rest_constraint(
    model: cp_model.CpModel,
    x: list[list[cp_model.IntVar]],
    n_residents: int,
    n_hours: int,
    min_rest: int,
) -> None:
    """
    Enforce at least min_rest hours of rest between any two consecutive shifts.

    When a shift ends at hour t (x[r][t]=1, x[r][t+1]=0), hours t+1 through t+min_rest
    must all be non-working. Hour t+1 is already non-working by the shift-end condition,
    so we only need to enforce hours t+2 through t+min_rest explicitly.

    Derived from: x[r][t+s] + x[r][t] - x[r][t+1] <= 1  for s in [2, min_rest]
    which activates (forces x[r][t+s]=0) exactly when x[r][t]=1 and x[r][t+1]=0.
    """
    for r in range(n_residents):
        for t in range(n_hours - 1):
            for s in range(2, min_rest + 1):
                if t + s < n_hours:
                    model.add(x[r][t + s] + x[r][t] - x[r][t + 1] <= 1)


def _add_max_consecutive_constraint(
    model: cp_model.CpModel,
    x: list[list[cp_model.IntVar]],
    n_residents: int,
    n_hours: int,
    max_consecutive: int,
) -> None:
    """
    No resident may work more than max_consecutive hours in a row.

    Enforced by requiring that in every window of (max_consecutive + 1) hours,
    at most max_consecutive of them are working hours.
    """
    window = max_consecutive + 1
    for r in range(n_residents):
        for t in range(n_hours - max_consecutive):
            model.add(sum(x[r][t + s] for s in range(window)) <= max_consecutive)


def _add_max_weekly_constraint(
    model: cp_model.CpModel,
    x: list[list[cp_model.IntVar]],
    n_residents: int,
    n_hours: int,
    max_weekly: int,
) -> None:
    """No resident may work more than max_weekly hours in any 168-hour week."""
    for r in range(n_residents):
        for week_start in range(0, n_hours, _HOURS_PER_WEEK):
            week_end = min(week_start + _HOURS_PER_WEEK, n_hours)
            model.add(sum(x[r][t] for t in range(week_start, week_end)) <= max_weekly)


def _add_min_teaching_constraint(
    model: cp_model.CpModel,
    x: list[list[cp_model.IntVar]],
    n_residents: int,
    n_hours: int,
    teaching_hours: tuple[int, ...],
    min_teaching: int,
) -> None:
    """Each resident must work at least min_teaching hours among the designated teaching hours."""
    valid_teaching_hours = [t for t in teaching_hours if t < n_hours]
    for r in range(n_residents):
        model.add(sum(x[r][t] for t in valid_teaching_hours) >= min_teaching)


def _add_min_shift_length_constraint(
    model: cp_model.CpModel,
    x: list[list[cp_model.IntVar]],
    n_residents: int,
    n_hours: int,
    min_shift_length: int,
) -> None:
    """
    Every working shift must be at least min_shift_length hours long.

    A shift starts at hour t when x[r][t]=1 and x[r][t-1]=0 (or t=0).
    On a shift start the constraint fires as:

        sum(x[r][t .. t+min_shift_length-1]) >= min_shift_length

    encoded linearly as:

        sum(window) >= min_shift_length * x[r][t] - min_shift_length * x[r][t-1]

    When x[r][t-1]=1 (mid-shift) or x[r][t]=0 the RHS is <= 0 and the
    constraint is trivially satisfied.

    A new shift may not start within min_shift_length hours of the horizon end,
    because it cannot be completed. Those hours are only reachable by continuing
    an already-running shift (x[r][t] <= x[r][t-1]).
    """
    for r in range(n_residents):
        for t in range(n_hours):
            if t + min_shift_length > n_hours:
                # Cannot complete a full shift from here: block new starts.
                if t > 0:
                    model.add(x[r][t] <= x[r][t - 1])
                else:
                    model.add(x[r][0] == 0)
            else:
                window_sum = sum(x[r][t + s] for s in range(min_shift_length))
                if t > 0:
                    model.add(
                        window_sum
                        >= min_shift_length * x[r][t] - min_shift_length * x[r][t - 1]
                    )
                else:
                    model.add(window_sum >= min_shift_length * x[r][0])


def _add_min_days_off_constraint(
    model: cp_model.CpModel,
    x: list[list[cp_model.IntVar]],
    n_residents: int,
    n_hours: int,
    min_days_off_per_week: int,
) -> None:
    """
    Each resident must have at least min_days_off_per_week full rest days per week.

    A full rest day is any 24-consecutive-hour window in which the resident does
    not work at all.

    For each (resident, week) pair a boolean indicator day_off[t] is introduced
    for every valid 24-hour window start t within that week.  The implications

        day_off[t] = 1  =>  x[r][t+s] = 0   for s in 0..23

    are enforced directly, and at least min_days_off_per_week indicators must be
    set to 1.  Weeks shorter than 24 hours are skipped.
    """
    for r in range(n_residents):
        for week_start in range(0, n_hours, _HOURS_PER_WEEK):
            week_end = min(week_start + _HOURS_PER_WEEK, n_hours)
            if week_end - week_start < _HOURS_PER_DAY:
                continue

            valid_starts = range(week_start, week_end - _HOURS_PER_DAY + 1)
            day_off_vars = [
                model.new_bool_var(f"day_off_{r}_{t}") for t in valid_starts
            ]

            model.add(sum(day_off_vars) >= min_days_off_per_week)

            for day_off_var, t in zip(day_off_vars, valid_starts):
                for s in range(_HOURS_PER_DAY):
                    model.add_implication(day_off_var, x[r][t + s].negated())


def solve(
    params: SchedulingParams, time_limit_seconds: float = 60.0
) -> Schedule | None:
    """
    Solve the resident scheduling problem with CP-SAT.

    Minimizes total working hours across all residents (most relaxed feasible schedule).
    Returns a Schedule if one exists within the time limit, or None if infeasible.
    """
    model = cp_model.CpModel()
    R = params.n_residents
    T = params.n_hours

    x: list[list[cp_model.IntVar]] = [
        [model.new_bool_var(f"x_{r}_{t}") for t in range(T)] for r in range(R)
    ]

    _add_min_coverage_constraint(model, x, R, T, params.min_on_duty)
    _add_min_rest_constraint(model, x, R, T, params.min_rest)
    _add_max_consecutive_constraint(model, x, R, T, params.max_consecutive)
    _add_max_weekly_constraint(model, x, R, T, params.max_weekly)
    _add_min_teaching_constraint(
        model, x, R, T, params.teaching_hours, params.min_teaching
    )
    _add_min_shift_length_constraint(model, x, R, T, params.min_shift_length)
    _add_min_days_off_constraint(model, x, R, T, params.min_days_off_per_week)

    model.minimize(sum(x[r][t] for r in range(R) for t in range(T)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds

    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    assignments: list[list[int]] = [
        [int(solver.value(x[r][t])) for t in range(T)] for r in range(R)
    ]
    return Schedule(params=params, assignments=assignments)
