"""Test box calculations."""
import unittest

import numpy as np
import numpy.testing as npt
import torch

from finder.pylib import old_box_calc as calc


class TestBoxCalc(unittest.TestCase):
    def test_iou_01(self):
        """It handles disjoint boxes."""
        box1 = [10, 10, 20, 20]
        box2 = [30, 30, 40, 40]
        self.assertEqual(calc.iou(box1, box2), 0.0)

    def test_iou_02(self):
        """It handles disjoint boxes."""
        box1 = [30, 30, 40, 40]
        box2 = [10, 10, 20, 20]
        self.assertEqual(calc.iou(box1, box2), 0.0)

    def test_iou_03(self):
        """It handles one box inside another box."""
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 5, 5]
        ii1 = 10.0 * 10.0
        ii2 = 5.0 * 5.0
        self.assertEqual(calc.iou(box1, box2), (ii2 / (ii1 + ii2 - ii2)))

    def test_nms_01(self):
        """It handles non-overlapping."""
        boxes = np.array(
            [
                [10, 10, 20, 20],
                [30, 30, 40, 40],
                [50, 50, 60, 60],
            ]
        )
        npt.assert_array_equal(calc.nms(boxes), boxes)

    def test_nms_02(self):
        """It handles one box inside another."""
        boxes = np.array(
            [
                [100, 100, 400, 400],
                [110, 110, 390, 390],
            ]
        )
        npt.assert_array_equal(calc.nms(boxes), [boxes[0]])

    def test_nms_03(self):
        """It handles overlap above the threshold."""
        boxes = np.array(
            [
                [100, 100, 400, 400],
                [110, 110, 410, 410],
            ]
        )
        npt.assert_array_equal(calc.nms(boxes), [boxes[1]])

    def test_nms_04(self):
        """It handles overlap below the threshold."""
        boxes = np.array(
            [
                [100, 100, 400, 400],
                [399, 399, 500, 500],
            ]
        )
        npt.assert_array_equal(calc.nms(boxes), boxes)

    def test_nms_05(self):
        """It handles an empty array."""
        boxes = np.array([])
        npt.assert_array_equal(calc.nms(boxes), boxes)

    def test_overlap_groups_01(self):
        """It handles non-overlapping."""
        boxes = np.array(
            [
                [10, 10, 20, 20],
                [30, 30, 40, 40],
                [50, 50, 60, 60],
            ]
        )
        npt.assert_array_equal(calc.overlapping_boxes(boxes), [3, 2, 1])

    def test_overlap_groups_02(self):
        """It handles one box inside another."""
        boxes = np.array(
            [
                [100, 100, 400, 400],
                [110, 110, 390, 390],
            ]
        )
        npt.assert_array_equal(calc.overlapping_boxes(boxes), [1, 1])

    def test_overlap_groups_03(self):
        """It handles overlap above the threshold."""
        boxes = np.array(
            [
                [0, 0, 100, 200],
                [0, 1, 102, 203],
            ]
        )
        npt.assert_array_equal(calc.overlapping_boxes(boxes), [1, 1])

    def test_overlap_groups_04(self):
        """It handles overlap below the threshold."""
        boxes = np.array(
            [
                [0, 0, 1, 2],  # Bigger
                [1, 2, 2, 3],  # Smaller
            ]
        )
        npt.assert_array_equal(calc.overlapping_boxes(boxes), [1, 2])

    def test_overlap_groups_05(self):
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
        npt.assert_array_equal(calc.overlapping_boxes(boxes), [1, 2, 2, 1, 2])

    def test_small_box_suppression_01(self):
        """It removes a tiny box."""
        boxes = torch.Tensor([[0, 0, 10, 10], [0, 0, 100, 100]])
        npt.assert_array_equal(calc.small_box_suppression(boxes), torch.tensor([1]))

    def test_small_box_suppression_02(self):
        """It removes a different tiny box."""
        boxes = torch.Tensor([[0, 0, 100, 100], [0, 0, 10, 10]])
        npt.assert_array_equal(calc.small_box_suppression(boxes), torch.tensor([0]))

    def test_small_box_suppression_03(self):
        """It does not remove non-overlapping boxes."""
        boxes = torch.Tensor([[0, 0, 100, 100], [200, 200, 210, 210]])
        npt.assert_array_equal(calc.small_box_suppression(boxes), torch.tensor([0, 1]))

    def test_small_box_suppression_04(self):
        """It does not remove overlaps below the threshold."""
        boxes = torch.Tensor([[0, 0, 100, 100], [99, 99, 109, 109]])
        npt.assert_array_equal(calc.small_box_suppression(boxes), torch.tensor([0, 1]))
