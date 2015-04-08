__author__ = 'willengler'

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

    if leave_one_out:
        # Generate list of datetimes from Jan 1, 2005 to latest day in dataset
        days_to_predict = get_all_days(foo, bar)
    else:
        # Pick random set of num_days days from Jan 1, 2005 to latest day in dataset
        days_to_predict = pick_days(num_days, last_day)

    # Get dicts mapping comm area to accuracy on that area
    seq_accuracy = get_sequential_accuracy(days_to_predict)
    nonseq_accuracy = get_nonsequential_accuracy(days_to_predict)
    baseline_accuracy = get_baseline_accuracy(days_to_predict)

    rankings = create_rankings(seq_accuracy, nonseq_accuracy, baseline_accuracy)
    report_rankings(rankings)


def pick_days(num_days, end_date):
    """
    :return: list of datetimes between Jan 1, 2005 and end_date that we want
     to test the algorithms on
    """
    pass


def get_all_days(start_date, end_date):
    """
    :return: list of datetimes with one datetime for each day between
        the start of evaluation and the end
    """
    pass

"""
get_[sequential, nonsequential, baseline]_accuracy takes:
    days_to_predict: a list of datetimes on which to generate and test predictions
and returns:
    accuracy_by_comm_area: a dict mapping community area names to the percentage of days correctly classified
"""


def get_sequential_accuracy(days_to_predict):
    pass


def get_nonsequential_accuracy(days_to_predict):
    pass


def get_baseline_accuracy(days_to_predict):
    pass


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
    pass


def report_rankings(rankings):
    """
    Outputs a file with easily readable/processable summary of evaluation for each neighborhood

    :param rankings: a mapping from community area names to Ranking objects

    """
    pass
