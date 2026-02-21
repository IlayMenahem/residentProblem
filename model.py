from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulingParams:
    """
    Parameters defining a resident scheduling problem instance.

    Time is measured in hours from the start of the scheduling horizon.
    Weeks are fixed 168-hour windows starting at hour 0.
    """

    n_residents: int  # total number of residents available
    n_hours: int  # length of the scheduling horizon in hours
    min_on_duty: int  # N: minimum residents on duty at every hour
    min_rest: int  # M: minimum rest hours required between any two shifts
    max_consecutive: int  # K: maximum consecutive working hours per shift
    max_weekly: int  # L: maximum working hours in any 168-hour week
    teaching_hours: tuple[int, ...]  # ordered hour indices that count as teaching time
    min_teaching: int  # P: minimum hours each resident must work during teaching hours
    min_shift_length: int  # C6: minimum duration of any single working shift in hours
    min_days_off_per_week: (
        int  # C7: minimum number of full 24-hour rest windows per resident per week
    )


@dataclass
class Schedule:
    """
    A feasible resident schedule produced by the solver.

    assignments[r][t] == 1 means resident r is working at hour t, else 0.
    """

    params: SchedulingParams
    assignments: list[list[int]]

    def working_hours(self, resident: int) -> int:
        """Total working hours assigned to a resident over the full horizon."""
        return sum(self.assignments[resident])

    def on_duty_count(self, hour: int) -> int:
        """Number of residents on duty at a given hour."""
        return sum(self.assignments[r][hour] for r in range(self.params.n_residents))

    def teaching_hours_count(self, resident: int) -> int:
        """Hours resident worked during any of the designated teaching hours."""
        return sum(
            self.assignments[resident][t]
            for t in self.params.teaching_hours
            if t < self.params.n_hours
        )
