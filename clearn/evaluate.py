from clearn import munge
from clearn import predict

import datetime
import json
import math
import pandas as pd
import random
import sys

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
    time_series_dict = munge.get_master_dict('crimeSample.csv')
    # TODO Better way to get the end date?
    last_day_of_data = time_series_dict.tail(1).index.to_pydate

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

    rankings = create_rankings(seq_accuracy, nonseq_accuracy, baseline_accuracy, len(days_to_predict))
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
    accuracy_by_comm_area: a dict mapping community area names to the number of days correctly classified
"""

def get_sequential_accuracy(time_series_dict, days_to_predict):
    sequential_series = predict.sequential_preprocess(time_series_dict)
    area_to_performance_map = {}
    for area, dataframe in time_series_dict:
        number_correct_predictions = 0

        for day in days_to_predict:
            predicted_result, prob = predict.sequential(dataframe, day)
            # Assume that this is store
            actual_result = dataframe['Violent Crime Committed?'][day]
            if actual_result == predicted_result:
                number_correct_predictions += 1

        area_to_performance_map[area] = number_correct_predictions

    return area_to_performance_map

def get_nonsequential_accuracy(time_series_dict, days_to_predict):
    non_sequential_series = predict.nonsequential_preprocess(time_series_dict)
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

        area_to_performance_map[area] = number_correct_predictions

    return area_to_performance_map

def get_baseline_accuracy(time_series_dict, days_to_predict):
    baseline_series = predict.baseline_preprocess(time_series_dict)

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

        area_to_performance_map[area] = number_correct_predictions

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


def create_rankings(seq_accuracy, nonseq_accuracy, baseline_accuracy, total_count):
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

        models = [('sequential', sequential), ('nonsequential', nonsequential), ('baseline', baseline)]

        # Naively sort models by accuracy.  Reverse it, because smallest are first in array by default; we want
        # to sort by most correct, so we want largest first.
        sorted_models = sorted(models, key=lambda model_tuple: model_tuple[1], reverse=True)

        # Apply rankings to the models
        area_ranking.ranks[sorted_models[0][0]] = 1
        find_ranking(area_ranking, sorted_models, total_count, 1)
        find_ranking(area_ranking, sorted_models, total_count, 2)

        # Finally, update the accuracies in the ranking object:
        for model_tuple in sorted_models:
            area_ranking.accuracy[model_tuple[0]] = model_tuple[1]

        area_to_ranking_map[area] = area_ranking

    return area_to_ranking_map

"""
    Takes in a ranking object, a sorted array of model tuples (see previous function), the number of instances, and the
    index of the first element to rank, and it gives said element the proper ranking.
"""
def find_ranking(ranking, sorted_models, total_count, second_index):
    first_index = second_index - 1

    model_comparison = run_z_test(sorted_models[first_index][1], sorted_models[second_index][1], total_count)

    # ... and assign rankings
    if model_comparison == 1:
        ranking.ranks[sorted_models[second_index][0]] = second_index + 1
    elif model_comparison == 0:
        ranking.ranks[sorted_models[second_index][0]] = second_index
    else:
        sys.exit('Error in sorting algorithm in evaluate.py when calculating ranking for third model')

"""
run_z_test takes:
    first_accuracy: Number of predictions correct from the first model
    second_accuracy: Number of predictions correct from the second
    total_count: Total number of instances tested
and returns:
    significantly_different: -1 if first is significantly worse than second, 0 if there is no significant difference,
    and 1 if first is significantly better than second (using 95% confidence)
"""
def run_z_test(first_accuracy, second_accuracy, total_count):
    if first_accuracy < 0:
        raise ValueError("First accuracy is negative.")

    if second_accuracy < 0:
        raise ValueError("Second accuracy is negative.")

    if total_count < 1:
        raise ValueError("Must have a non-zero count for test")

    first_wrong = total_count - first_accuracy
    second_wrong = total_count - second_accuracy
    first_error = first_wrong / total_count
    second_error = second_wrong / total_count

    error_diff = first_error - second_error

    variance = ((first_error * (1 - first_error)) + (second_error * (1 - second_error)) / total_count)
    stdev = math.sqrt(variance)

    ci_term = 1.96 * stdev

    ci_left_boundary = error_diff - ci_term
    ci_right_boundary = error_diff + ci_term

    # If the interval contains zero, there's no significant difference
    if ci_left_boundary < 0 and ci_right_boundary > 0:
        return 0

    # Difference is significant, return value based on which accuracy is greater
    if first_accuracy < second_accuracy:
        return -1
    else:
        return 1


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
