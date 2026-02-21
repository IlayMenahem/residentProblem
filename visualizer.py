import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from model import Schedule

_HOURS_PER_DAY = 24
_HOURS_PER_WEEK = 168


def plot_schedule(schedule: Schedule, title: str = "Resident Schedule") -> None:
    """
    Render the schedule as a heatmap.

    Rows are residents, columns are hours.
    Blue cells are working hours.
    Teaching hours are shaded green (individual columns, not necessarily contiguous).
    Red dashed lines mark week boundaries.
    """
    params = schedule.params
    R = params.n_residents
    T = params.n_hours

    grid = np.array(schedule.assignments, dtype=float)  # shape: (R, T)

    fig, ax = plt.subplots(figsize=(max(12, T // 6), max(4, R // 2 + 2)))

    ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="upper")

    ax.set_xlabel("Hour")
    ax.set_ylabel("Resident")
    ax.set_title(title)

    ax.set_yticks(range(R))
    ax.set_yticklabels([f"R{r}" for r in range(R)])

    for week_start in range(0, T, _HOURS_PER_WEEK):
        ax.axvline(
            x=week_start - 0.5,
            color="red",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
            label="Week boundary" if week_start == 0 else "_nolegend_",
        )

    for t in params.teaching_hours:
        if t < T:
            ax.axvspan(t - 0.5, t + 0.5, alpha=0.15, color="green", zorder=0)

    legend_handles = [
        mpatches.Patch(facecolor="steelblue", label="Working"),
        mpatches.Patch(facecolor="green", alpha=0.4, label="Teaching hour"),
        plt.Line2D(
            [0], [0], color="red", linestyle="--", linewidth=1.0, label="Week boundary"
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def print_summary(schedule: Schedule) -> None:
    """Print a per-resident summary and check whether all constraints are met."""
    params = schedule.params
    R = params.n_residents
    T = params.n_hours

    print(f"Schedule Summary  ({R} residents, {T} hours)")
    print("=" * 60)
    print(f"{'Resident':<12}{'Total hrs':>10}{'Teaching hrs':>14}{'Max consec':>12}")
    print("-" * 60)

    for r in range(R):
        total = schedule.working_hours(r)
        teaching = schedule.teaching_hours_count(r)
        max_consec = _max_consecutive_hours(schedule.assignments[r])
        print(f"  R{r:<10}{total:>10}{teaching:>14}{max_consec:>12}")

    print()
    _print_constraint_check(schedule)


def _max_consecutive_hours(row: list[int]) -> int:
    """Return the longest run of consecutive working hours in a single resident row."""
    max_run = current_run = 0
    for slot in row:
        current_run = current_run + 1 if slot else 0
        max_run = max(max_run, current_run)
    return max_run


def _shift_lengths(row: list[int]) -> list[int]:
    """Return the length of every working shift in a resident's assignment row."""
    shifts: list[int] = []
    run = 0
    for slot in row:
        if slot:
            run += 1
        elif run > 0:
            shifts.append(run)
            run = 0
    if run > 0:
        shifts.append(run)
    return shifts


def _full_day_off_count(row: list[int], week_start: int, week_end: int) -> int:
    """Return the number of 24-consecutive-hour off windows within [week_start, week_end)."""
    return sum(
        1
        for t in range(week_start, week_end - _HOURS_PER_DAY + 1)
        if all(row[t + s] == 0 for s in range(_HOURS_PER_DAY))
    )


def _print_constraint_check(schedule: Schedule) -> None:
    """Verify and display whether every problem constraint is satisfied."""
    params = schedule.params
    T = params.n_hours

    violations: list[str] = []

    # Constraint 1: minimum coverage
    for t in range(T):
        count = schedule.on_duty_count(t)
        if count < params.min_on_duty:
            violations.append(
                f"Coverage: only {count} on duty at hour {t} (need {params.min_on_duty})"
            )

    # Constraint 2: minimum rest between shifts
    for r in range(params.n_residents):
        row = schedule.assignments[r]
        for t in range(T - 1):
            if row[t] == 1 and row[t + 1] == 0:
                for s in range(1, params.min_rest + 1):
                    if t + s < T and row[t + s] == 1:
                        violations.append(
                            f"Rest: R{r} works at hour {t + s}, only {s}h after shift ended at {t}"
                        )

    # Constraint 3: max consecutive hours
    for r in range(params.n_residents):
        run = 0
        for t, slot in enumerate(schedule.assignments[r]):
            run = run + 1 if slot else 0
            if run > params.max_consecutive:
                violations.append(
                    f"Consecutive: R{r} has >{params.max_consecutive} consecutive hours at hour {t}"
                )

    # Constraint 4: max weekly hours
    for r in range(params.n_residents):
        for week_start in range(0, T, _HOURS_PER_WEEK):
            week_end = min(week_start + _HOURS_PER_WEEK, T)
            weekly = sum(
                schedule.assignments[r][t] for t in range(week_start, week_end)
            )
            if weekly > params.max_weekly:
                violations.append(
                    f"Weekly: R{r} has {weekly}h in week starting {week_start} (max {params.max_weekly})"
                )

    # Constraint 5: minimum teaching hours
    for r in range(params.n_residents):
        count = schedule.teaching_hours_count(r)
        if count < params.min_teaching:
            violations.append(
                f"Teaching: R{r} has {count}h (need {params.min_teaching})"
            )

    # Constraint 6: minimum shift length
    for r in range(params.n_residents):
        for length in _shift_lengths(schedule.assignments[r]):
            if length < params.min_shift_length:
                violations.append(
                    f"Shift length: R{r} has a {length}h shift (min {params.min_shift_length})"
                )

    # Constraint 7: minimum full days off per week
    for r in range(params.n_residents):
        for week_start in range(0, T, _HOURS_PER_WEEK):
            week_end = min(week_start + _HOURS_PER_WEEK, T)
            if week_end - week_start < _HOURS_PER_DAY:
                continue
            count = _full_day_off_count(schedule.assignments[r], week_start, week_end)
            if count < params.min_days_off_per_week:
                violations.append(
                    f"Days off: R{r} has {count} full day(s) off in week @{week_start}"
                    f" (need {params.min_days_off_per_week})"
                )

    if violations:
        print("CONSTRAINT VIOLATIONS:")
        for v in violations:
            print(f"  ✗ {v}")
    else:
        print("All constraints satisfied ✓")
