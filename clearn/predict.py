import pandas as pd
from . import munge
from .convolve import convolve_by_neighbor
import datetime

DAYS_IN_MONTH = 30

"""
Each of sequential(), nonsequential(), and baseline() take:
    - time_series: a pandas data_frame representing a series of days
        in a particular community area preprocessed for that algorithm
    - day: a datetime specifying the day being predicted

Each returns:
    - classification: True if crime is predicted. Else False.
    - probability: a value between 0 and 1 that represents the predicted probability of violent crime
"""


def sequential(time_series, day):
    pass


def nonsequential(time_series, day):
    pass


def baseline(time_series, day):
    previous_month = get_previous_month(time_series, day)
    # Predict assuming that percentage of days with crime in last month gives us probability of crime the next day
    num_days_with_violent_crime = previous_month['Violent Crime Committed?'].sum()
    probability = num_days_with_violent_crime/DAYS_IN_MONTH
    classification = probability > .5
    return classification, probability

"""
Helpers for sequential(), nonsequential(), baseline()
"""


def get_previous_month(time_series, day):
    """
    Given pandas dataframe indexed by day,
    returns pandas dataframe consisting of the 30 days before day
    """
    thirty_days_ago = day - datetime.timedelta(days=DAYS_IN_MONTH)
    yesterday = day - datetime.timedelta(days=1)
    return time_series.loc[thirty_days_ago: yesterday]


def get_training_examples_up_to(time_series, day):
    # Grab data frame with all days including the last day, and then cut off the last day
    return time_series.loc[:day][:-1]


"""
Each of sequential_preprocess(), nonsequential_preprocess(), baseline_preprocess() take:
    - master_area_dict: the canonical mapping from community area names
        to pandas data frames representing a series of days in that area.
        Also includes data frame for all of Chicago.
        See munge.py for more documentation.
Each returns:
    - [sequential, nonsequential, baseline]_dict: mapping from community area names
        to pandas data frame ready for dumping into the proper algortihm
"""


def sequential_preprocess(master_area_dict):
    pass


def nonsequential_preprocess(master_area_dict, convolve=False):
    # Add windows to every data frame
    with_windows = {area: extract_windows(frame) for area, frame in master_area_dict}
    # Map each community area to a dataframe containing that area's recent history
    #   AND the whole city's recent history
    chicago_frame = with_windows.pop('Chicago')
    with_city_history = {area: frame.join(chicago_frame) for area, frame in with_windows}
    if convolve:
        return convolve_by_neighbor(with_city_history)
    else:
        return with_city_history


def baseline_preprocess(master_area_dict):
    del master_area_dict['Chicago']
    days_by_area = \
        {area: munge.drop_all_columns_but(frame, ['Violent Crime Commited?']) for area, frame in master_area_dict}
    return days_by_area


"""
Helpers for sequential_preprocess(), nonsequential_preprocess(), baseline_preprocess()
"""

def extract_windows(days):
    # Add categories to count types of crimes committed in time windows leading to date we're trying to predict
    for label in ['Violent', 'Severe', 'Minor', 'Petty']:
        days[label + ' Crimes in Last Week'] = pd.rolling_sum(days[label + ' Crimes'], 7)
        days[label + ' Crimes in Last Month'] = pd.rolling_sum(days[label + ' Crimes'], 30)
    # The earliest 30 days in the time series have missing values for their first 30 days. Remove those days.
    return days[:30]


