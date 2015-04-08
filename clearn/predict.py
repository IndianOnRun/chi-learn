from . import munge

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
    pass


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


def nonsequential_preprocess(master_area_dict):
    pass


def baseline_preprocess(master_area_dict):
    pass