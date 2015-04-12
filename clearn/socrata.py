import requests
import json
import pandas as pd
import pickle
from . import munge

"""
How am I going to do this?
1) add make_master_dict to munge
 - it should persist the master dict to a pickle file. get_master_dict should retrieve it
2) Create script to update master dict when new data is available
    a) Find out most recent date in master_dict
    b) See if more recent date is available
    c) Make query to grab all new data
    d) Convert to same format as csv
    e) munge it
    f) add it to master_dict
    g) persist master_dict
    h) run analysis
    i) generate prediction for next day
    j) Check prediction for preceding day
    k) Generate HTML
    l) Push HTML to gh_pages branch
"""


PICKLE_PATH = '../data/masterPandasFrame.pickle'


def convert_frame_to_csv_col_names(data_frame):
    data_frame = munge.drop_all_columns_but(data_frame, ['date', 'primary_type', 'community_area', 'arrest', 'domestic'])


def pickle_data_frame(data_frame):
    with open(PICKLE_PATH, 'wb') as file:
        pickle.dump(data_frame, file, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_data_frame():
    try:
        with open(PICKLE_PATH, 'rb') as file:
            return pickle.load(file)
    except IOError:
        print('Unable to open pickled dataframe.')
        return None