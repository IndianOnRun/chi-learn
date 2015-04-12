import unittest
from unittest.mock import MagicMock
import os
import json
import copy
from clearn import clearn_path
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

        with open(clearn_path('data/test_files/expected_results.json')) as expected_file:
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


class TestRankingAlgorithm(unittest.TestCase):
    def setUp(self):
        self.original_run_z_test = evaluate.run_z_test

    def test_ranking_no_tie(self):
        evaluate.run_z_test = MagicMock(return_value=1)

        expected_result = {'sequential': 1, 'nonsequential': 2, 'baseline': 3}

        sorted_models = [('sequential', 1), ('nonsequential', 2), ('baseline', 3)]
        actual_ranking = evaluate.Ranking()
        total_count = 10

        actual_ranking.ranks['sequential'] = 1
        evaluate.find_ranking(actual_ranking, sorted_models, total_count, 1)
        evaluate.find_ranking(actual_ranking, sorted_models, total_count, 2)

        self.assertEquals(expected_result, actual_ranking.ranks)

    def test_ranking_with_first_two_tie(self):
        expected_result = {'sequential': 1, 'nonsequential': 1, 'baseline': 2}

        sorted_models = [('sequential', 1), ('nonsequential', 2), ('baseline', 3)]
        actual_ranking = evaluate.Ranking()
        total_count = 10

        actual_ranking.ranks['sequential'] = 1

        evaluate.run_z_test = MagicMock(return_value=0)
        evaluate.find_ranking(actual_ranking, sorted_models, total_count, 1)

        evaluate.run_z_test = MagicMock(return_value=1)
        evaluate.find_ranking(actual_ranking, sorted_models, total_count, 2)

        self.assertEquals(expected_result, actual_ranking.ranks)

    def test_ranking_three_way_tie(self):
        evaluate.run_z_test = MagicMock(return_value=0)

        expected_result = {'sequential': 1, 'nonsequential': 1, 'baseline': 1}

        sorted_models = [('sequential', 1), ('nonsequential', 2), ('baseline', 3)]
        actual_ranking = evaluate.Ranking()
        total_count = 10

        actual_ranking.ranks['sequential'] = 1
        evaluate.find_ranking(actual_ranking, sorted_models, total_count, 1)
        evaluate.find_ranking(actual_ranking, sorted_models, total_count, 2)

        self.assertEquals(expected_result, actual_ranking.ranks)

    def tearDown(self):
        evaluate.run_z_test = self.original_run_z_test


class TestRankingDictCreation(unittest.TestCase):
    pass
