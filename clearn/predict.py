import pandas as pd
from . import munge
from hmmlearn import MultinomialHMM
import numpy as np
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
"""


def sequential(time_series, day): 
    #play around with number of components- see what is most accurate
    x = get_previous_month(time_series,day) #x is the timeseries of the three previous days
    x = x[0].values.tolist()
    results = []
    #run this nine (dont have to worry about ties) times to account for the randomness- can also play around with this number
    for ind in range(0,9):
        model = MultinomialHMM(n_components=3,n_iter=10000) #initialize the model
        model.fit([np.array(x)]) #fit the model
        hidden_states = model.predict(x) #determine the hidden states for the sequence
        last_state_probs = model.predict_proba(x)[len(x)-1] #get the most recent hidden state probabilities
        current_state = np.where(last_state_probs == max(last_state_probs))[0][0] #determine the most likely current state from those probs
        transition_probs = model.transmat_[current_state] #get the probabilities of the next state given that state
        next_state = np.where(transition_probs==max(transition_probs))[0][0] #get the next state as the most likely of these probs
        emissions = model.emissionprob_[next_state] #get the emission probabilities of the current state
        output = np.where(emissions==max(emissions))[0][0] #determine the most likely of these emissions
        results.append(output) #add this output to our results array
    if np.count_nonzero(results) >4:
        return 1
    else:
        return 0


def nonsequential(time_series, day, model):
    training_frame = get_time_series_up_to(time_series, day)
    # Grab boolean list of whether a violent crime was committed on each day.
    #  Start with the second day.
    targets = training_frame['Violent Crime Committed?'][1:]
    # Now that we've extracted the targets from the frame, remove them.
    del training_frame['Violent Crime Committed?']
    # Grab list of feature vectors for every day.
    #   End with second to last day in our history.
    #   Now each feature vector is aligned with whether a violent crime was committed on the following day
    feature_vectors = training_frame.values[:-1]
    # Train our model on the targets and features
    trained_model = model.fit(feature_vectors, targets)
    feature_vec_to_classify = training_frame.tail(1).features(0)
    # Even though we're only making one prediction, sklearn expects to receive and output list-like data structures
    classification = trained_model.predict([feature_vec_to_classify])[0]
    return classification


def baseline(time_series, day):
    previous_month = get_previous_month(time_series, day)
    # Predict assuming that percentage of days with crime in last month gives us probability of crime the next day
    num_days_with_violent_crime = previous_month['Violent Crime Committed?'].sum()
    probability = num_days_with_violent_crime/DAYS_IN_MONTH
    classification = probability > .5
    return classification

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


def get_time_series_up_to(time_series, day):
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
    # Drop the unnecessary columns of each neighborhoods data frame
    sequential_dict = {}
    for key in master_area_dict.keys():
        pre_process_df = master_area_dict[key]
        pre_process_df = pre_process_df.drop(['Arrest','Domestic','Severe Crimes','Minor Crimes','Petty Crimes','Month','Weekday'],1)
    # Convert all violent crime numbers greater than 0 to 1
        index = 0
        while index < len(pre_process_df[0]):
            if pre_process_df[0][index] > 0:
                pre_process_df[0][index] = 1
            index = index + 1
        sequential_dict[key]=pre_process_df
    return sequential_dict


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