#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
import kmeans

import numpy as np


class TestKmeans(unittest.TestCase):
    def test_diff_true(self):
        old_mean = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        new_mean = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        I = kmeans.Kmeans()
        I.mean = new_mean
        expr = I.diff(old_mean)
        self.assertIs(expr, True)

    def test_diff_false(self):
        old_mean = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        new_mean = np.array([[1, 2, 3], [4, 7, 6], [7, 8, 9]])
        I = kmeans.Kmeans()
        I.mean = new_mean
        expr = I.diff(old_mean)
        self.assertIs(expr, False)

    def test_diff_true2(self):
        old_mean = np.array([[1, 1], [1, 1]])
        new_mean = np.array([[0, 0], [0, 0]])
        I = kmeans.Kmeans()
        I.mean = new_mean
        expr = I.diff(old_mean)
        self.assertIs(expr, False)

    def test_diff_false2(self):
        old_mean = np.array([[0.00000001, 2], [4, 5]])
        new_mean = np.array([[0.0000001, 2], [4, 5]])
        I = kmeans.Kmeans()
        I.mean = new_mean
        expr = I.diff(old_mean)
        self.assertIs(expr, True)

    def test_calc_mean(self):
        I = kmeans.Kmeans()
        I.features = np.array(([[10, 20], [14, 24], [100, 120], [110, 120]]))
        I.label = [0, 0, 1, 1]
        ary = I.calc_mean()
        expr = np.array(([[12, 22], [105, 120]]))
        self.assertTrue(np.all(ary == expr))
