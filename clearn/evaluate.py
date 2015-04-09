__author__ = 'willengler'

from . import munge, predict

import pandas as pd
import random
import json

"""
How do we do this?

1) Get master mapping from neighborhoods + city to day-indexed time series
2) Determine days to test
3) For each algorithm,
    a) preprocess master mapping
    b) predict for each community area each day in the list
    c) Store summary statistics for each community area
4) For each community area,
    a) Perform statistical test for each community area
    b) Output result file (JSON?) with summary for each neighborhood


"""


def evaluate(num_days, leave_one_out=False):
    """
    Generate a JSON document mapping community area names
        to performance metrics for each algorithm
    """
    time_series_dict = munge.get_master_dict()
    # TODO Better way to get the end date?
    last_day_of_data = time_series.tail(1).index.to_pydate

    # Since we can't evaluate the data from data (predicting tomorrow's violent
    # crimes), we subtract one
    end_date = last_day_of_data - datetime.timedelta(days=1)

    if leave_one_out:
        # Generate list of datetimes from Jan 1, 2005 to latest day in dataset
        days_to_predict = get_all_days(datetime.date(2005, 1, 1), end_date)
    else:
        # Pick random set of num_days days from Jan 1, 2005 to latest day in dataset
        days_to_predict = pick_days(num_days, end_date)

    # Get dicts mapping comm area to accuracy on that area
    seq_accuracy = get_sequential_accuracy(time_series_dict, days_to_predict)
    nonseq_accuracy = get_nonsequential_accuracy(time_series_dict, days_to_predict)
    baseline_accuracy = get_baseline_accuracy(time_series_dict, days_to_predict)

    rankings = create_rankings(seq_accuracy, nonseq_accuracy, baseline_accuracy)
    report_rankings(rankings)


def pick_days(num_days, end_date):
    fullrange = get_all_days(datetime.date(2005, 1, 1), end_date)
    return random.sample(fullrange, num_days)

def get_all_days(start_date, end_date):
    """
    :return: list of datetimes with one datetime for each day between
        the start of evaluation and the end
    """
    # date_range uses inclusive date endpoints.
    return pd.date_range(start_date, end_date).tolist()

"""
get_[sequential, nonsequential, baseline]_accuracy takes:
    days_to_predict: a list of datetimes on which to generate and test predictions
and returns:
    accuracy_by_comm_area: a dict mapping community area names to the percentage of days correctly classified
"""

def get_sequential_accuracy(time_series_dict, days_to_predict):
    sequential_series = predict.sequential_preprocess(time_series)
    area_to_performance_map = {}
    for area, dataframe in time_series_dict:
        number_correct_predictions = 0

        for day in days_to_predict:
            predicted_result, prob = predict.sequential(dataframe, day)
            # Assume that this is store
            actual_result = dataframe['Violent Crime Committed?'][day]
            if actual_result == predicted_result:
                number_correct_predictions += 1

        area_to_performance_map[area] = number_correct_predictions / len(days_to_predict)

    return area_to_performance_map

def get_nonsequential_accuracy(time_series_dict, days_to_predict):
    non_sequential_series = predict.nonsequential_preprocess(time_series)
    area_to_performance_map = {}
    for area, dataframe in time_series_dict:
        number_correct_predictions = 0

        for day in days_to_predict:
            predicted_result, prob = predict.nonsequential(dataframe, day)
            # Assume that prediction is stored in same date rather than the next
            # day (punt this work to pre-processing)
            actual_result = dataframe['Violent Crime Committed?'][day]
            if actual_result == predicted_result:
                number_correct_predictions += 1

        area_to_performance_map[area] = number_correct_predictions / len(days_to_predict)

    return area_to_performance_map

def get_baseline_accuracy(time_series_dict, days_to_predict):
    baseline_series = predict.baseline_preprocess(time_series)

    area_to_performance_map = {}
    for area, dataframe in time_series_dict:
        number_correct_predictions = 0

        for day in days_to_predict:
            predicted_result, prob = predict.baseline(dataframe, day)
            # Assume that prediction is stored in same date rather than the next
            # day (punt this work to pre-processing)
            actual_result = dataframe['Violent Crime Committed?'][day]
            if actual_result == predicted_result:
                number_correct_predictions += 1

        area_to_performance_map[area] = number_correct_predictions / len(days_to_predict)

    return area_to_performance_map

class Ranking:
    def __init__(self):
        self.ranks = {
            # Should be populated with int 1, 2, or 3
            'nonsequential': None,
            'sequential': None,
            'baseline': None
        }
        self.accuracy = {
            # Should be populated with float between 0 and 1
            'nonsequential': None,
            'sequential': None,
            'baseline': None
        }


def create_rankings(seq_accuracy, nonseq_accuracy, baseline_accuracy):
    """

    :return: dictionary mapping comm areas to rankings of each neighborhood
    """

    area_to_ranking_map = {}
    # We sorta pick a random array here to iterate over :P
    for area in seq_accuracy:
        sequential = seq_accuracy[area]
        nonsequential = nonseq_accuracy[area]
        baseline = baseline_accuracy[area]

        area_ranking = Ranking()

        rating_tuples = [('sequential', sequential), ('nonsequential', nonsequential), ('baseline', baseline)]
        sorted_rating_tuples = sorted(array_of_tuples, key=lambda rating: rating[1], reverse=True)

        sorted_models = map(lambda rating: rating[0], sorted_tuples)

        ranks = {}

        for model in ['sequential', 'nonsequential', 'baseline']:
            ranks[model] = sorted_models.index(model) + 1

        area_ranking.ranks = ranks

        area_ranking.accuracy = {'sequential': sequential, 'nonsequential': nonsequential, 'baseline': baseline}

        area_to_ranking[area] = area_ranking

    return area_to_ranking_map

def report_rankings(rankings):
    """
    Outputs a file with easily readable/processable summary of evaluation for each neighborhood

    :param rankings: a mapping from community area names to Ranking objects

    """
    to_json = {}
    for area, rank_obj in rankings:
        # Turn ranking objects into dicts for easy serialization with JSON
        # class
        rank_hash = {}
        rank_hash['ranks'] = rank_obj.ranks
        rank_hash['accuracy'] = rank_obj.accuracy
        to_json[area] = rank_hash

    output_file = open('results.json')
    json.dump(to_json, output_file)
    output_file.close()
