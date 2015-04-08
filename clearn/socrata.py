import requests
import json
import pandas as pd
import pickle
from . import munge

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