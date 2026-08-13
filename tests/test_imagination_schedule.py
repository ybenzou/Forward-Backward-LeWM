"""Imagination depth schedule contracts."""

import pytest

from policy import compute_imagine_steps, expected_replan_depths


def test_offset_schedules():
    assert expected_replan_depths(25) == [0]
    assert expected_replan_depths(50) == [5, 0]
    assert expected_replan_depths(75) == [10, 5, 0]
    assert expected_replan_depths(100) == [15, 10, 5, 0]


def test_elapsed_past_offset_is_zero():
    assert compute_imagine_steps(75, 100, 25, 5) == 0
    assert compute_imagine_steps(25, 25, 25, 5) == 0


def test_non_divisible_raises():
    with pytest.raises(ValueError):
        compute_imagine_steps(76, 0, 25, 5)
    with pytest.raises(ValueError):
        compute_imagine_steps(75, 0, 24, 5)
    with pytest.raises(ValueError):
        compute_imagine_steps(75, 0, 25, 0)
