import unittest
import csv
import pandas as pd
from clearn import evaluate

class TestZTest(unittest.TestCase):
    def test_with_first_significantly_better(self):
        first_accuracy = 280
        second_accuracy = 120
        total_count = 300

        actual_result = evaluate.run_z_test(first_accuracy, second_accuracy, total_count)
        expected_result = 1
        self.assertEqual(expected_result, actual_result)

    def test_with_second_significantly_better(self):
        first_accuracy = 10
        second_accuracy = 80
        total_count = 100

        actual_result = evaluate.run_z_test(first_accuracy, second_accuracy, total_count)
        expected_result = -1
        self.assertEqual(expected_result, actual_result)

    def test_with_no_significant_difference(self):
        first_accuracy = 130
        second_accuracy = 140
        total_count = 150

        actual_result = evaluate.run_z_test(first_accuracy, second_accuracy, total_count)
        expected_result = 0
        self.assertEqual(expected_result, actual_result)
