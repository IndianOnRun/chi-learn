from clearn import munge
from clearn import predict

import datetime
import json
import math
import pandas as pd
import random
import sys
import copy

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
    last_day_of_data = time_series_dict['Edgewater'].index[-1].to_datetime().date()

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
    seq_accuracy = get_predictor_accuracy(copy.deepcopy( time_series_dict ), days_to_predict, predict.SequentialPredictor)
    nonseq_accuracy = get_predictor_accuracy(copy.deepcopy( time_series_dict ), days_to_predict, predict.NonsequentialPredictor)
    baseline_accuracy = get_predictor_accuracy(copy.deepcopy( time_series_dict ), days_to_predict, predict.BaselinePredictor)

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
    if end_date < start_date:
        raise ValueError("End date should be after start date")

    timestamps = pd.date_range(start_date, end_date).tolist()

    return [timestamp for timestamp in timestamps]

"""
get_predictor_accuracy takes:
    days_to_predict: a list of datetimes on which to generate and test predictions
and returns:
    accuracy_by_comm_area: a dict mapping community area names to the number of days correctly classified
"""

def get_predictor_accuracy(time_series_dict, days_to_predict, predictor_to_use):
    if not issubclass(predictor_to_use, predict.Predictor):
        raise ValueError("Please pass in a valid predictor.")

    processed_time_series_dict = predictor_to_use.preprocess(time_series_dict)

    area_to_performance_map = {}
    for area, dataframe in processed_time_series_dict.items():
        area_to_performance_map[area] = get_predictor_accuracy_in_area(dataframe, days_to_predict, predictor_to_use)

    return area_to_performance_map

def get_predictor_accuracy_in_area(dataframe, days_to_predict, predictor_to_use):
    predictor = predictor_to_use(dataframe)

    last_date = dataframe.index[-1].date()

    days_to_predict.sort()
    first_predicted_date = days_to_predict[0].date()
    last_predicted_date = days_to_predict[-1].date()

    # Don't start predicting before 2005
    if first_predicted_date < datetime.date(2005,1,1):
        raise ValueError("Don't predict dates before 2005")

    if last_predicted_date > last_date:
        raise ValueError("Can't predict beyond our last data point")

    number_correct_predictions = 0

    for day in days_to_predict:
        predicted_result = predictor.predict(day)
        actual_result = dataframe['Violent Crime Committed?'].loc[day]
        if actual_result == predicted_result:
            number_correct_predictions += 1

    return number_correct_predictions


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
    if seq_accuracy.keys() != nonseq_accuracy.keys() or nonseq_accuracy.keys() != baseline_accuracy.keys():
        raise ValueError("The cities in your arrays don't match up.")

    if total_count < 1:
        raise ValueError("Can't have negative trials")

    area_to_ranking_map = {}
    # We sorta pick a random array here to iterate over :P
    for area in seq_accuracy:
        sequential = seq_accuracy[area]
        nonsequential = nonseq_accuracy[area]
        baseline = baseline_accuracy[area]

        if sequential < 0 or nonsequential < 0 or baseline < 0:
            raise ValueError("Can't have negative results.")

        if sequential > total_count or nonsequential > total_count or baseline > total_count:
            raise ValueError("Can't have more accurate predictions that trials")

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
        ranking.ranks[sorted_models[second_index][0]] = ranking.ranks[sorted_models[first_index][0]] + 1
    elif model_comparison == 0:
        ranking.ranks[sorted_models[second_index][0]] = ranking.ranks[sorted_models[first_index][0]]
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
    if first_accuracy == second_accuracy:
        return 0

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
    for area, rank_obj in rankings.items():
        if rank_obj is None:
            raise ValueError("Supplied no ranking for " + area)

        if rank_obj.ranks is None or rank_obj.accuracy is None:
            raise ValueError("Ranking() for " + area + " is missing information.")

        rank_hash = {}
        rank_hash['ranks'] = rank_obj.ranks
        rank_hash['accuracy'] = rank_obj.accuracy
        to_json[area] = rank_hash

    output_file = open('results.json', 'w')
    json.dump(to_json, output_file)
    output_file.close()
