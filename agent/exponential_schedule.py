import math


class ExponentialSchedule:
    def __init__(self, value_from, value_to, num_steps):
        """Exponential schedule from `value_from` to `value_to` in `num_steps` steps.

        $value(t) = a \exp (b t)$

        :param value_from: Initial value
        :param value_to: Final value
        :param num_steps: Number of steps for the exponential schedule
        """
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps

        # YOUR CODE HERE: Determine the `a` and `b` parameters such that the schedule is correct
        self.a = self.value_from
        if num_steps > 1:
            self.b = math.log(value_to / value_from) / (num_steps - 1)
        else:
            self.b = 0  # Avoid division by zero if num_steps is 1

    def value(self, step) -> float:
        """Return exponentially interpolated value between `value_from` and `value_to`interpolated value between.

        Returns {
            `value_from`, if step == 0 or less
            `value_to`, if step == num_steps - 1 or more
            the exponential interpolation between `value_from` and `value_to`, if 0 <= steps < num_steps
        }

        :param step: The step at which to compute the interpolation
        :rtype: Float. The interpolated value
        """

        # YOUR CODE HERE: Implement the schedule rule as described in the docstring,
        # using attributes `self.a` and `self.b`.
        if step <= 0:
            value = self.value_from
        elif step >= self.num_steps - 1:
            value = self.value_to
        else:
            value = self.a * math.exp(self.b * step)

        return value


def _test_schedule(schedule, step, value, ndigits=5):
    """Tests that the schedule returns the correct value."""
    v = schedule.value(step)
    if not round(v, ndigits) == round(value, ndigits):
        raise Exception(
            f"For step {step}, the scheduler returned {v} instead of {value}"
        )


if __name__ == "__main__":
    _schedule = ExponentialSchedule(0.1, 0.2, 3)
    _test_schedule(_schedule, -1, 0.1)
    _test_schedule(_schedule, 0, 0.1)
    _test_schedule(_schedule, 1, 0.141421356237309515)
    _test_schedule(_schedule, 2, 0.2)
    _test_schedule(_schedule, 3, 0.2)
    del _schedule

    _schedule = ExponentialSchedule(0.5, 0.1, 5)
    _test_schedule(_schedule, -1, 0.5)
    _test_schedule(_schedule, 0, 0.5)
    _test_schedule(_schedule, 1, 0.33437015248821106)
    _test_schedule(_schedule, 2, 0.22360679774997905)
    _test_schedule(_schedule, 3, 0.14953487812212207)
    _test_schedule(_schedule, 4, 0.1)
    _test_schedule(_schedule, 5, 0.1)
    del _schedule
