import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from model import Schedule, SchedulingParams
from solver import solve

_HOURS_PER_DAY = 24
_HOURS_PER_WEEK = 168

# ---------------------------------------------------------------------------
# Helpers – constraint checkers (independent of the solver)
# ---------------------------------------------------------------------------


def check_min_coverage(schedule: Schedule) -> list[str]:
    params = schedule.params
    return [
        f"Hour {t}: {schedule.on_duty_count(t)} on duty (need ≥{params.min_on_duty})"
        for t in range(params.n_hours)
        if schedule.on_duty_count(t) < params.min_on_duty
    ]


def check_min_rest(schedule: Schedule) -> list[str]:
    params = schedule.params
    violations: list[str] = []
    for r in range(params.n_residents):
        row = schedule.assignments[r]
        for t in range(params.n_hours - 1):
            if row[t] == 1 and row[t + 1] == 0:
                for s in range(1, params.min_rest + 1):
                    if t + s < params.n_hours and row[t + s] == 1:
                        violations.append(
                            f"R{r}: works at hour {t + s}, only {s}h after shift ended at {t}"
                        )
    return violations


def check_max_consecutive(schedule: Schedule) -> list[str]:
    params = schedule.params
    violations: list[str] = []
    for r in range(params.n_residents):
        run = 0
        for t, slot in enumerate(schedule.assignments[r]):
            run = run + 1 if slot else 0
            if run > params.max_consecutive:
                violations.append(
                    f"R{r}: {run} consecutive hours at hour {t} (max {params.max_consecutive})"
                )
    return violations


def check_max_weekly(schedule: Schedule) -> list[str]:
    params = schedule.params
    violations: list[str] = []
    for r in range(params.n_residents):
        for week_start in range(0, params.n_hours, _HOURS_PER_WEEK):
            week_end = min(week_start + _HOURS_PER_WEEK, params.n_hours)
            weekly = sum(
                schedule.assignments[r][t] for t in range(week_start, week_end)
            )
            if weekly > params.max_weekly:
                violations.append(
                    f"R{r}: {weekly}h in week @{week_start} (max {params.max_weekly})"
                )
    return violations


def check_min_teaching(schedule: Schedule) -> list[str]:
    params = schedule.params
    return [
        f"R{r}: {schedule.teaching_hours_count(r)} teaching hours (need ≥{params.min_teaching})"
        for r in range(params.n_residents)
        if schedule.teaching_hours_count(r) < params.min_teaching
    ]


def check_min_shift_length(schedule: Schedule) -> list[str]:
    """
    Every working shift must be at least min_shift_length hours long.

    A shift that runs to the end of the horizon is also checked; the solver
    ensures such shifts cannot be shorter than the minimum.
    """
    params = schedule.params
    violations: list[str] = []
    for r in range(params.n_residents):
        row = schedule.assignments[r]
        in_shift = False
        shift_start = 0
        for t in range(params.n_hours):
            if row[t] == 1 and not in_shift:
                in_shift = True
                shift_start = t
            elif row[t] == 0 and in_shift:
                length = t - shift_start
                if length < params.min_shift_length:
                    violations.append(
                        f"R{r}: shift from hour {shift_start} to {t - 1} "
                        f"is {length}h (min {params.min_shift_length})"
                    )
                in_shift = False
        if in_shift:
            length = params.n_hours - shift_start
            if length < params.min_shift_length:
                violations.append(
                    f"R{r}: shift from hour {shift_start} to {params.n_hours - 1} "
                    f"is {length}h (min {params.min_shift_length})"
                )
    return violations


def check_min_days_off(schedule: Schedule) -> list[str]:
    """
    Each resident must have at least min_days_off_per_week full 24-hour rest
    windows in every 168-hour week.  Weeks shorter than 24 hours are skipped.
    """
    params = schedule.params
    violations: list[str] = []
    for r in range(params.n_residents):
        row = schedule.assignments[r]
        for week_start in range(0, params.n_hours, _HOURS_PER_WEEK):
            week_end = min(week_start + _HOURS_PER_WEEK, params.n_hours)
            if week_end - week_start < _HOURS_PER_DAY:
                continue
            days_off = sum(
                1
                for t in range(week_start, week_end - _HOURS_PER_DAY + 1)
                if all(row[t + s] == 0 for s in range(_HOURS_PER_DAY))
            )
            if days_off < params.min_days_off_per_week:
                violations.append(
                    f"R{r}: {days_off} full day(s) off in week @{week_start} "
                    f"(need ≥{params.min_days_off_per_week})"
                )
    return violations


def assert_all_constraints(schedule: Schedule) -> None:
    """Raise AssertionError listing every violated constraint."""
    all_violations = (
        check_min_coverage(schedule)
        + check_min_rest(schedule)
        + check_max_consecutive(schedule)
        + check_max_weekly(schedule)
        + check_min_teaching(schedule)
        + check_min_shift_length(schedule)
        + check_min_days_off(schedule)
    )
    assert not all_violations, "Constraint violations:\n" + "\n".join(all_violations)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_params(**overrides) -> SchedulingParams:
    """Return a small, fast-to-solve parameter set with optional field overrides."""
    defaults = dict(
        n_residents=6,
        n_hours=48,
        min_on_duty=2,
        min_rest=4,
        max_consecutive=8,
        max_weekly=80,
        teaching_hours=tuple(range(8, 25)),  # hours 8–24 inclusive
        min_teaching=2,
        min_shift_length=4,
        # Constraint 7 is disabled by default because the 48-hour horizon makes
        # it infeasible when combined with teaching_hours that span hours 8–24:
        # every valid 24-hour off window that still leaves 2 free teaching hours
        # contains hour 24, so no two residents can be on duty there.
        # Tests for constraint 7 use n_hours=168 where the overlap cannot occur.
        min_days_off_per_week=0,
    )
    defaults.update(overrides)
    return SchedulingParams(**defaults)


# ---------------------------------------------------------------------------
# Feasibility tests
# ---------------------------------------------------------------------------


def test_simple_feasible():
    """A straightforward problem instance must return a schedule."""
    schedule = solve(_base_params())
    assert schedule is not None


def test_infeasible_too_few_residents():
    """Requiring more on-duty residents than exist must be infeasible."""
    schedule = solve(_base_params(n_residents=2, min_on_duty=3))
    assert schedule is None


def test_infeasible_teaching_exceeds_available_hours():
    """Requiring more teaching hours than the list contains must be infeasible."""
    # Only 8 teaching hours available; require 10
    schedule = solve(_base_params(teaching_hours=tuple(range(10, 18)), min_teaching=10))
    assert schedule is None


def test_infeasible_max_weekly_too_tight():
    """
    If min_on_duty * n_hours > n_residents * max_weekly the workload cannot be
    distributed without exceeding the weekly cap.

    4 residents × max 1h/week = 4 total hours available,
    but we need ≥2 on duty for 48 hours = 96 resident-hours minimum.
    """
    schedule = solve(_base_params(n_residents=4, min_on_duty=2, max_weekly=1))
    assert schedule is None


def test_infeasible_min_shift_length_blocks_all_work():
    """
    When min_shift_length equals the entire horizon every resident's single shift
    fills the planning window entirely – combined with min_on_duty this is only
    feasible when there are enough residents to sustain coverage while each works
    the full horizon.  Here we force infeasibility by demanding that min_on_duty
    exceeds available residents.
    """
    # horizon = 24h, min_shift_length = 24 → each resident either works all 24h or none
    # With 2 residents and min_on_duty = 3 this is impossible
    schedule = solve(
        _base_params(n_hours=24, n_residents=2, min_on_duty=3, min_shift_length=24)
    )
    assert schedule is None


def test_infeasible_no_room_for_full_day_off():
    """
    When every hour must be covered and there are not enough residents to give
    anyone a 24-hour off window, the schedule is infeasible.

    2 residents, both required on duty at every hour → neither can have a day off.
    """
    schedule = solve(
        _base_params(
            n_residents=2,
            n_hours=48,
            min_on_duty=2,
            min_days_off_per_week=1,
        )
    )
    assert schedule is None


def test_minimal_horizon():
    """Single-hour horizon with one resident on duty must be trivially solvable."""
    params = SchedulingParams(
        n_residents=1,
        n_hours=1,
        min_on_duty=1,
        min_rest=0,
        max_consecutive=1,
        max_weekly=168,
        teaching_hours=(0,),
        min_teaching=1,
        min_shift_length=1,
        min_days_off_per_week=0,  # no room for a full day off in a 1-hour horizon
    )
    schedule = solve(params)
    assert schedule is not None
    assert schedule.assignments[0][0] == 1


# ---------------------------------------------------------------------------
# Constraint satisfaction tests (all seven constraints)
# ---------------------------------------------------------------------------


def test_constraint_1_min_coverage():
    schedule = solve(_base_params())
    assert schedule is not None
    assert not check_min_coverage(schedule), check_min_coverage(schedule)


def test_constraint_2_min_rest():
    schedule = solve(_base_params(min_rest=6))
    assert schedule is not None
    assert not check_min_rest(schedule), check_min_rest(schedule)


def test_constraint_3_max_consecutive():
    schedule = solve(_base_params(max_consecutive=6))
    assert schedule is not None
    assert not check_max_consecutive(schedule), check_max_consecutive(schedule)


def test_constraint_4_max_weekly():
    # Use a one-week horizon so the weekly cap is meaningful.
    # 8 residents × 45 h/week = 360 resident-hours ≥ 2 on-duty × 168 h = 336 required.
    schedule = solve(
        _base_params(
            n_hours=168,
            n_residents=8,
            max_weekly=45,
            min_days_off_per_week=1,
        )
    )
    assert schedule is not None
    assert not check_max_weekly(schedule), check_max_weekly(schedule)


def test_constraint_5_min_teaching():
    schedule = solve(_base_params(teaching_hours=tuple(range(10, 31)), min_teaching=4))
    assert schedule is not None
    assert not check_min_teaching(schedule), check_min_teaching(schedule)


def test_constraint_6_min_shift_length():
    """Every shift in the produced schedule must meet the minimum length."""
    schedule = solve(_base_params(min_shift_length=6))
    assert schedule is not None
    assert not check_min_shift_length(schedule), check_min_shift_length(schedule)


def test_constraint_6_shift_length_of_one_hour_permitted():
    """
    When min_shift_length=1 every single-hour assignment is a valid shift.
    The constraint must not block feasible solutions.
    """
    schedule = solve(_base_params(min_shift_length=1))
    assert schedule is not None
    assert not check_min_shift_length(schedule), check_min_shift_length(schedule)


def test_constraint_7_min_days_off():
    """Each resident must have at least one 24-hour off window per week."""
    schedule = solve(
        _base_params(
            n_hours=168,
            n_residents=8,
            max_weekly=100,
            min_days_off_per_week=1,
        )
    )
    assert schedule is not None
    assert not check_min_days_off(schedule), check_min_days_off(schedule)


def test_constraint_7_zero_days_off_is_unconstrained():
    """
    min_days_off_per_week=0 places no restriction on rest days and must always
    yield a feasible schedule (given the other constraints allow one).
    """
    schedule = solve(_base_params(min_days_off_per_week=0))
    assert schedule is not None


def test_all_constraints_simultaneously():
    """Solve a richer instance and verify every constraint at once."""
    params = SchedulingParams(
        n_residents=8,
        n_hours=168,
        min_on_duty=2,
        min_rest=8,
        max_consecutive=12,
        max_weekly=60,
        teaching_hours=tuple(range(40, 81)),  # hours 40–80 inclusive
        min_teaching=4,
        min_shift_length=4,
        min_days_off_per_week=1,
    )
    schedule = solve(params, time_limit_seconds=120.0)
    assert schedule is not None
    assert_all_constraints(schedule)


# ---------------------------------------------------------------------------
# Schedule structure tests
# ---------------------------------------------------------------------------


def test_schedule_shape():
    """assignments must have shape (n_residents, n_hours) with values in {0, 1}."""
    params = _base_params()
    schedule = solve(params)
    assert schedule is not None

    assert len(schedule.assignments) == params.n_residents
    for row in schedule.assignments:
        assert len(row) == params.n_hours
        assert all(v in (0, 1) for v in row)


def test_working_hours_non_negative():
    schedule = solve(_base_params())
    assert schedule is not None
    for r in range(schedule.params.n_residents):
        assert schedule.working_hours(r) >= 0


def test_on_duty_count_consistent():
    """on_duty_count must equal the column-sum of the assignments matrix."""
    schedule = solve(_base_params())
    assert schedule is not None
    for t in range(schedule.params.n_hours):
        expected = sum(
            schedule.assignments[r][t] for r in range(schedule.params.n_residents)
        )
        assert schedule.on_duty_count(t) == expected


def test_teaching_hours_count_matches_list():
    """teaching_hours_count must sum assignments only over the designated teaching hours."""
    schedule = solve(_base_params(teaching_hours=tuple(range(10, 21)), min_teaching=2))
    assert schedule is not None
    for r in range(schedule.params.n_residents):
        manual = sum(schedule.assignments[r][t] for t in schedule.params.teaching_hours)
        assert schedule.teaching_hours_count(r) == manual


def test_teaching_hours_non_contiguous():
    """
    Non-contiguous teaching hours must be handled correctly by both the solver
    and teaching_hours_count.

    Four scattered hours are designated; every resident must cover at least 1.
    """
    teaching_hours = (5, 15, 25, 35)
    schedule = solve(_base_params(teaching_hours=teaching_hours, min_teaching=1))
    assert schedule is not None
    assert not check_min_teaching(schedule), check_min_teaching(schedule)
    for r in range(schedule.params.n_residents):
        manual = sum(schedule.assignments[r][t] for t in teaching_hours)
        assert schedule.teaching_hours_count(r) == manual


def test_params_stored_on_schedule():
    """The schedule must carry back the exact params it was solved with."""
    params = _base_params()
    schedule = solve(params)
    assert schedule is not None
    assert schedule.params == params
