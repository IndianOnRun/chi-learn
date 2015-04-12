from clearn import clearn_path
from clearn import evaluate
from clearn.predict import NonsequentialPredictor
from unittest.mock import MagicMock
from unittest.mock import patch
import copy
import datetime
import json
import os
import unittest
import pandas as pd
import numpy as np

class TestEvaluate(unittest.TestCase):
    pass

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
    def test_basic_rankings_generation(self):
        # The numbers returned for the ratings are not going to be
        # right, but they are correct given what we mock out find_ranking
        # to be.  Changing the return value in the middle of the function
        # is a bit much.
        seq_accuracy = {'Pittsburgh': 230, 'Philidelphia': 100}
        nonseq_accuracy = {'Pittsburgh': 200, 'Philidelphia': 110}
        baseline_accuracy = {'Pittsburgh': 80, 'Philidelphia': 80}
        total_count = 300
        area_to_rankings_map = evaluate.create_rankings(seq_accuracy, nonseq_accuracy, baseline_accuracy, total_count)

        # Checks to make sure each area is accounted for, and that we don't
        # return any empty rankings
        for area in ['Pittsburgh', 'Philidelphia']:
            self.assertTrue(area in area_to_rankings_map)
            self.assertIsNotNone(area_to_rankings_map[area])
            self.assertIsNotNone(area_to_rankings_map[area].ranks)
            self.assertIsNotNone(area_to_rankings_map[area].accuracy)

    def test_ranking_generation_with_differing_length_arrays(self):
        seq_accuracy = {'Pittsburgh': 230}
        nonseq_accuracy = {'Pittsburgh': 200, 'Philidelphia': 110}
        baseline_accuracy = {'Pittsburgh': 80, 'Philidelphia': 80}
        total_count = 300

        with self.assertRaises(ValueError):
            area_to_rankings_map = evaluate.create_rankings(seq_accuracy, nonseq_accuracy, baseline_accuracy, total_count)

    def test_ranking_generation_with_invalid_day_length(self):
        seq_accuracy = {'Pittsburgh': 230, 'Philidelphia': 100}
        nonseq_accuracy = {'Pittsburgh': 200, 'Philidelphia': 110}
        baseline_accuracy = {'Pittsburgh': 80, 'Philidelphia': 80}
        total_count = 20

        with self.assertRaises(ValueError):
            area_to_rankings_map = evaluate.create_rankings(seq_accuracy, nonseq_accuracy, baseline_accuracy, total_count)

    def test_ranking_generation_with_negatives(self):
        seq_accuracy = {'Pittsburgh': -100, 'Philidelphia': 100}
        nonseq_accuracy = {'Pittsburgh': 200, 'Philidelphia': 110}
        baseline_accuracy = {'Pittsburgh': 80, 'Philidelphia': 80}
        total_count = -50

        with self.assertRaises(ValueError):
            area_to_rankings_map = evaluate.create_rankings(seq_accuracy, nonseq_accuracy, baseline_accuracy, total_count)

class TestPredictorAccuracy(unittest.TestCase):
    def setUp(self):
        self.backup_preprocess = NonsequentialPredictor.preprocess
        self.backup_accuracy = evaluate.get_predictor_accuracy_in_area

        # Make preprocessing an identity function
        NonsequentialPredictor.preprocess = lambda dict: dict
        self.predictor = NonsequentialPredictor

        # Have accuracy always return 100 correct predictions
        evaluate.get_predictor_accuracy_in_area = lambda x, y, z: 100

    def test_get_accuracy(self):
        days_to_predict = pd.date_range(datetime.date(2001,1,1), datetime.date(2001, 1, 5))
        dataframe = pd.DataFrame({'Violent Crimes Committed?': True}, index=days_to_predict)
        initial_dict = {'Pittsburgh': dataframe, 'Philidelphia': dataframe}

        resulting_dict = evaluate.get_predictor_accuracy(initial_dict, days_to_predict, self.predictor)

        # Make sure we spit out a dict with the same keys
        self.assertEquals(initial_dict.keys(), resulting_dict.keys())

        # Make sure the values of said keys are the number correctly
        # predicted
        for _, correct_predictions in resulting_dict.items():
            self.assertEquals(correct_predictions, 100)

    def test_invalid_predictor(self):
        with self.assertRaises(ValueError):
            evaluate.get_predictor_accuracy(None, None, evaluate.Ranking)

    def tearDown(self):
        NonsequentialPredictor.preprocess = self.backup_preprocess
        evaluate.get_predictor_accuracy_in_area = self.backup_accuracy

class TestPredictorAreaAccuracy(unittest.TestCase):
    def setUp(self):
        self.backup_predict = NonsequentialPredictor.predict
        self.predictor = NonsequentialPredictor

    def test_predictor_accuracy_in_area_all_correct(self):
        self.predictor.predict = MagicMock(return_value=True)

        expected_true_days = 100
        actual_true_days = self.get_actual_true_days()

        self.assertEquals(expected_true_days, actual_true_days)

    def test_predictor_accuracy_in_area_some_correct(self):
        # Generate a list of values to use as return values for the
        # stubbed function; namely, alternately return True and False
        alternating_list = np.resize([True, False], 100)
        self.predictor.predict = MagicMock(side_effect=alternating_list)

        expected_true_days = 50
        actual_true_days = self.get_actual_true_days()

        self.assertEquals(expected_true_days, actual_true_days)

    def test_predict_too_many_days(self):
        days_to_predict = evaluate.pick_days(100, datetime.date(2007,1,1))
        days_for_dataframe = evaluate.get_all_days(datetime.date(2005,1,1), datetime.date(2005, 3, 1))
        dataframe = pd.DataFrame({'Violent Crime Committed?': True, 'Other data': np.random.randn(60)}, index=days_for_dataframe)

        with self.assertRaises(ValueError):
            evaluate.get_predictor_accuracy_in_area(dataframe, days_to_predict, self.predictor)

    def get_actual_true_days(self):
        # Exactly 100 days
        days_to_predict = evaluate.get_all_days(datetime.date(2005,1,1), datetime.date(2005,4,10))

        # Generate a random dataframe with only the required column
        # 731 = days in the two years specified above
        dataframe = pd.DataFrame({'Violent Crime Committed?': True, 'Other data': np.random.randn(100)}, index=days_to_predict)

        # Generate a range of dates; we've tested this function separately.
        return evaluate.get_predictor_accuracy_in_area(dataframe, days_to_predict, self.predictor)

    def tearDown(self):
        NonsequentialPredictor.predict = self.backup_predict

class TestHelperFunctions(unittest.TestCase):
    def test_get_all_days_in_range(self):
        expected_date_range = [datetime.date(2005, 1, 1), datetime.date(2005, 1, 2), datetime.date(2005, 1, 3)]

        actual_date_range = evaluate.get_all_days(datetime.date(2005, 1, 1), datetime.date(2005, 1, 3))

        for index,timestamp in enumerate( actual_date_range ):
            self.assertEquals(timestamp.date(), expected_date_range[index])

    def test_get_all_days_invalid_range(self):
        with self.assertRaises(ValueError):
            evaluate.get_all_days(datetime.date(2005, 1, 3), datetime.date(2005, 1, 1))

    def test_pick_days_valid(self):
        date_range = evaluate.pick_days(10, datetime.date(2007, 1, 1))
        sorted_dates = sorted(date_range)

        # Make sure we got the right number
        self.assertEquals(len(date_range), 10)

        # Make sure all elements are unique.  If there aren't,
        # the actual list will have fewer elements than the set
        # version of the list (eliminates duplicates)
        self.assertTrue(len(date_range) == len(set(date_range)))
