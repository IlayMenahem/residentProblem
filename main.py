from model import SchedulingParams
from solver import solve
from visualizer import plot_schedule, print_summary

_HOURS_PER_DAY = 24
_DAYS = {
    "sunday": 0,
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
}


def build_teaching_hours(
    days: list[str],
    start_hour: int,
    end_hour: int,
) -> tuple[int, ...]:
    """
    Return a sorted tuple of absolute hour indices for the given days and daily time range.

    start_hour is inclusive, end_hour is exclusive (e.g. 12, 16 → hours 12, 13, 14, 15).
    Hour 0 is Sunday 00:00.
    """
    return tuple(
        sorted(
            _DAYS[day] * _HOURS_PER_DAY + h
            for day in days
            for h in range(start_hour, end_hour)
        )
    )


def main() -> None:
    """
    Example: schedule 8 residents over two weeks (336 hours).

    Shift rules reflect a realistic hospital setting:
      - At least 2 residents on duty at all times
      - 7 hours of rest between shifts
      - No shift longer than 16 hours
      - No more than 60 working hours in a week
      - Teaching rounds run 12:00–16:00 on Sunday through Wednesday;
        every resident must attend at least 10 of those hours
    """
    teaching_hours = build_teaching_hours(
        days=["sunday", "monday", "tuesday", "wednesday", "thursday"],
        start_hour=12,
        end_hour=16,
    )

    params = SchedulingParams(
        n_residents=10,
        n_hours=168,
        min_on_duty=3,
        min_rest=10,
        max_consecutive=12,
        max_weekly=60,
        teaching_hours=teaching_hours,
        min_teaching=10,
        min_shift_length=6,
        min_days_off_per_week=1,
    )

    print("Solving resident scheduling problem …")
    schedule = solve(params, time_limit_seconds=60.0)

    if schedule is None:
        print("No feasible schedule found within the time limit.")
        return

    print_summary(schedule)
    plot_schedule(schedule, title="Resident Schedule – 2 Weeks")


if __name__ == "__main__":
    main()
