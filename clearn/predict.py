import pandas as pd
from clearn import munge
from hmmlearn.hmm import MultinomialHMM
import numpy as np
from sklearn.naive_bayes import GaussianNB
from clearn.convolve import convolve_by_neighbor
import datetime
from abc import ABCMeta, abstractmethod
from copy import copy

DAYS_IN_MONTH = 30


class Predictor():
    """
    Interface for each prediction algorithm
    """

    # Makes Predictor an abstract class
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, time_series):
        """
        Given time_series (a properly preprocessed pandas data frame with dates as indices),
         initialize the predictor for the community area represented in the time series.
        """
        pass

    @abstractmethod
    def predict(self, day_to_predict):
        """
        Given a day in this predictor's time series (or one past the time series),
        return True if we predict a crime, False otherwise.
        """
        pass

    @staticmethod
    @abstractmethod
    def preprocess(master_dict):
        """
        Given the master_dict (as defined in munge.py),
        return a dict mapping each community area to a time series ready for consumption by this predictor.
        """
        pass


class SequentialPredictor(Predictor):

    def __init__(self, time_series):
        self.time_series = time_series

    def predict(self, day_to_predict):
        # Get records of 30 days before day_to_predict
        previous_thirty_days = get_previous_month(self.time_series, day_to_predict)
        binary_crime_sequence = previous_thirty_days['Violent Crime Committed?'].values.tolist()

        # Unsupervised HMM can't account for string of identical emissions.
        # If we see such a string, just predict the same emission for the following day.
        if binary_crime_sequence == [1]*30:
            return True
        if binary_crime_sequence == [0]*30:
            return False

        votes = []
        # Train nine HMMs. They are initialized randomly, so we take "votes" from nine HMMs.
        #  Why 9? Odd numbers preclude ties.
        #  And nine is a decent tradeoff between performance and getting bad results by chance
        for _ in range(9):
            # Train HMM
            model = MultinomialHMM(n_components=3, n_iter=10000)
            model.fit([np.array(binary_crime_sequence)])

            # Determine the most likely state of the last day in the sequence
            last_state_probs = model.predict_proba(binary_crime_sequence)[-1]
            current_state = self.get_most_likely(last_state_probs)

            # Determine the most likely state of the day we're trying to predict
            transition_probs = model.transmat_[current_state]
            next_state = self.get_most_likely(transition_probs)

            # Determine the most likely emission (crime/no crime) from a day in that state
            emissions = model.emissionprob_[next_state]
            vote = self.get_most_likely(emissions)

            # Record this HMM's vote
            votes.append(vote)

        # Votes are 1 for crime, 0 for no crime. Return True if majority votes for crime.
        return sum(votes) > 4

    @staticmethod
    def get_most_likely(probs):
        """
        probs is a vector of probability weights of a random variable.
        Returns the value with the highest weight.
        """
        return np.where(probs == max(probs))[0][0]

    @staticmethod
    def preprocess(master_dict):
        boolean_dict = BaselinePredictor.preprocess(master_dict)

        def convert_bool_frame_to_binary(df):
            df['Violent Crime Committed?'] = [int(boolean) for boolean in df['Violent Crime Committed?']]
            return df

        sequential_dict = {area: convert_bool_frame_to_binary(frame) for area, frame in boolean_dict.items()}
        return sequential_dict


class NonsequentialPredictor(Predictor):

    def __init__(self, time_series, model=GaussianNB()):
        self.time_series = time_series
        self.model = model

    def predict(self, day_to_predict):

        training_frame = self.get_time_series_up_to(self.time_series, day_to_predict)

        # Grab boolean list of whether a violent crime was committed on each day.
        #  Start with the second day.
        targets = training_frame['Violent Crime Committed?'][1:]

        # Now that we've extracted the targets from the frame, remove them.
        del training_frame['Violent Crime Committed?']

        # Grab list of feature vectors for every day.
        #   End with second to last day in our history.
        #   Now each feature vector is aligned with whether a violent crime was committed on the following day
        feature_vectors = training_frame.values[:-1]

        # Use the Predictor's model as a prototype and train a fresh model for each prediction.
        model = copy(self.model)
        # Train our model on the targets and features
        trained_model = model.fit(feature_vectors, targets)
        feature_vec_to_classify = training_frame.tail(1).features(0)

        # Even though we're only making one prediction, sklearn expects to receive and output list-like data structures
        prediction = trained_model.predict([feature_vec_to_classify])[0]
        return prediction

    @staticmethod
    def preprocess(master_dict, convolve=False):

        # Add windows of recent crime history to every data frame
        with_windows = {area: NonsequentialPredictor.extract_windows(frame) for area, frame in master_dict.items()}

        # Take data for the entire city out of master_dict
        chicago_frame = with_windows.pop('Chicago')
        # and relabel its columns to make clear that it is city data.
        new_column_names = ['Chicago ' + old_name for old_name in list(chicago_frame)]
        chicago_frame.columns = new_column_names

        # Map each community area to a dataframe containing that area's recent history
        #   AND the whole city's recent history
        with_city_history = {area: frame.join(chicago_frame) for area, frame in with_windows.items()}

        if convolve:
            return convolve_by_neighbor(with_city_history)
        else:
            return with_city_history

    @staticmethod
    def extract_windows(days):
        # Add categories to count types of crimes committed in time windows leading to date we're trying to predict
        for label in ['Violent', 'Severe', 'Minor', 'Petty']:
            days[label + ' Crimes in Last Week'] = pd.rolling_sum(days[label + ' Crimes'], 7)
            days[label + ' Crimes in Last Month'] = pd.rolling_sum(days[label + ' Crimes'], 30)
        # The earliest 30 days in the time series have missing values for their first 30 days. Remove those days.
        return days[30:]

    @staticmethod
    def get_time_series_up_to(time_series, day):
        # Grab data frame with all days including the last day, and then cut off the last day
        return time_series.loc[:day][:-1]


class BaselinePredictor(Predictor):

    def __init__(self, time_series):
        self.time_series = time_series

    def predict(self, day_to_predict):
        previous_month = get_previous_month(self.time_series, day_to_predict)

        # Predict assuming that percentage of days with crime in last month gives us probability of crime the next day
        num_days_with_violent_crime = previous_month['Violent Crime Committed?'].sum()
        proportion_of_days_with_violent_crime = num_days_with_violent_crime/DAYS_IN_MONTH
        prediction = proportion_of_days_with_violent_crime > .5
        return prediction

    @staticmethod
    def preprocess(master_area_dict):
        del master_area_dict['Chicago']
        days_by_area = {area: munge.drop_all_columns_but(frame, ['Violent Crime Committed?'])
                        for area, frame in master_area_dict.items()}
        return days_by_area


"""
Helper function used for baseline and sequential
"""


def get_previous_month(time_series, day):
        """
        Given pandas dataframe indexed by day,
        returns pandas dataframe consisting of the 30 days before day
        """
        thirty_days_ago = day - datetime.timedelta(days=DAYS_IN_MONTH)
        yesterday = day - datetime.timedelta(days=1)
        return time_series.loc[thirty_days_ago: yesterday]