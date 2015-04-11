import unittest
import csv
import os
import json
import copy
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

    def test_with_negative_accuracy(self):
        with self.assertRaises(ValueError):
            evaluate.run_z_test(1, -10, 1)

    def test_with_zero_count(self):
        with self.assertRaises(ValueError):
            evaluate.run_z_test(1, 1, 0)

class TestReportRankings(unittest.TestCase):
    def setUp(self):
        first_ranking = evaluate.Ranking()
        first_ranking.ranks = {'Sequential': 1, 'Nonsequential': 2, 'Baseline': 3}
        first_ranking.accuracy = {'Sequential': .50, 'Nonsequential': .30, 'Baseline': .10}

        second_ranking = evaluate.Ranking()
        second_ranking.ranks = {'Sequential': 2, 'Nonsequential': 3, 'Baseline': 1}
        second_ranking.accuracy = {'Sequential': .30, 'Nonsequential': .10, 'Baseline': .50}

        self.rankings = {'Pittsburgh': first_ranking, 'Philidelphia': second_ranking}

    def test_ranking_output(self):
        evaluate.report_rankings(self.rankings)

        with open('results.json') as actual_file:
            actual_results = json.load(actual_file)

        with open('clearn/data/test_files/expected_results.json') as expected_file:
            expected_results = json.load(expected_file)

        self.assertEqual(expected_results, actual_results)
        os.remove('results.json')

    def test_ranking_with_missing_ranking_object(self):
        malformed_rankings = copy.deepcopy(self.rankings)

        malformed_rankings['Pittsburgh'] = None

        with self.assertRaises(ValueError):
            evaluate.report_rankings(malformed_rankings)

    def test_ranking_with_missing_accuracy_attribute(self):
        malformed_rankings = copy.deepcopy(self.rankings)

        malformed_rankings['Pittsburgh'].accuracy = None

        with self.assertRaises(ValueError):
            evaluate.report_rankings(malformed_rankings)
