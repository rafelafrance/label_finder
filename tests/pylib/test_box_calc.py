"""Test box calculations."""
import unittest

import numpy as np
import numpy.testing as npt

from finder.pylib import box_calc as calc


class TestBoxCalc(unittest.TestCase):
    def test_find_box_groups_01(self):
        """It handles non-overlapping."""
        boxes = np.array(
            [
                [10, 10, 20, 20],
                [30, 30, 40, 40],
                [50, 50, 60, 60],
            ]
        )
        npt.assert_array_equal(calc.find_box_groups(boxes), [3, 2, 1])

    def test_find_box_groups_02(self):
        """It handles one box inside another."""
        boxes = np.array(
            [
                [100, 100, 400, 400],
                [110, 110, 390, 390],
            ]
        )
        npt.assert_array_equal(calc.find_box_groups(boxes), [1, 1])

    def test_find_box_groups_03(self):
        """It handles overlap above the threshold."""
        boxes = np.array(
            [
                [0, 0, 100, 200],
                [0, 1, 102, 203],
            ]
        )
        npt.assert_array_equal(calc.find_box_groups(boxes), [1, 1])

    def test_find_box_groups_04(self):
        """It handles overlap below the threshold."""
        boxes = np.array(
            [
                [0, 0, 1, 2],  # Bigger
                [1, 2, 2, 3],  # Smaller
            ]
        )
        npt.assert_array_equal(calc.find_box_groups(boxes), [1, 2])

    def test_find_box_groups_05(self):
        """It handles multiple groups of overlap."""
        boxes = np.array(
            [
                [100, 100, 400, 400],  # Group 1
                [500, 500, 600, 600],  # ..... 2
                [510, 510, 610, 610],  # ..... 2
                [110, 110, 410, 410],  # ..... 1
                [490, 490, 590, 590],  # ..... 2
            ]
        )
        npt.assert_array_equal(calc.find_box_groups(boxes, 0.5), [1, 2, 2, 1, 2])
