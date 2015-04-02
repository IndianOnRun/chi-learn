import pandas as pd
import csv

USE_CONVOLUTION = True

def make_raw_data_target_pairs(data_csv_path):
    frames_by_area = make_training_set(data_csv_path)
    pairs = {}
    for area in frames_by_area:
        # Chop off the first target value (was a violent crime committed today?)
        # and the last feature vector
        # so that day x's feature value is aligned with day x+1's classification
        frame = frames_by_area[area]
        raw_targets = frame['Violent Crime Committed?'][1:]
        del frame['Violent Crime Committed?']
        raw_data = frame.values[:-1]
        pairs[area] = (raw_data, raw_targets)
    return pairs


def make_training_set(data_csv_path):
    data_frame = pd.read_csv(data_csv_path)
    timestamps = make_clean_timestamps(data_frame)
    return make_feature_vectors(timestamps)


""" Used in make_clean_timestamps() """


def make_clean_timestamps(data_frame):
    data_frame = drop_all_columns_but(data_frame, ['Date', 'Primary Type', 'Community Area', 'Arrest', 'Domestic'])
    data_frame = convert_comm_area_nums_to_names(data_frame)
    data_frame = transform_from_csv(data_frame, 'Primary Type', '../config/crime_bins.csv')
    timestamps = reindex_by_date(data_frame)
    timestamps = make_cols_categorical(timestamps, ['Primary Type', 'Community Area'])
    return timestamps


def drop_all_columns_but(data_frame, columns):
    return data_frame.reindex(columns=columns)


def convert_comm_area_nums_to_names(data_frame):
    # Replace floats with ints. If no translation, mark it as 0 (an invalid community number)
    def clean_comm_area_value(val):
        try:
            return int(val)
        except ValueError:
            return 0
    data_frame['Community Area'] = data_frame['Community Area'].map(clean_comm_area_value)
    # Remove rows with invalid community area numbers
    data_frame = data_frame[data_frame['Community Area'] > 0]
    # Convert numbers to strings for easy binning
    data_frame['Community Area'] = data_frame['Community Area'].map(lambda num: str(int(num)))
    data_frame = transform_from_csv(data_frame, 'Community Area', '../config/community_areas.csv')
    return data_frame


def transform_from_csv(data_frame, col_name, csv_name):
    with open(csv_name, 'r') as file:
        unbinned_to_binned = {}
        reader = csv.reader(file)
        for line in reader:
            unbinned_to_binned[line[0]] = line[1]

    data_frame[col_name] = data_frame[col_name].map(lambda unbinned: unbinned_to_binned[unbinned])
    return data_frame


def reindex_by_date(data_frame):
    data_frame.index = pd.to_datetime(data_frame['Date'])
    return data_frame.drop('Date', 1)


def make_cols_categorical(data_frame, col_names):
    for name in col_names:
        data_frame[name] = data_frame[name].astype('category')
    return data_frame


""" Used in make_feature_vectors() """


def make_feature_vectors(timestamps):
    days_by_area = get_days_by_area(timestamps)
    days_pan_city = make_series_of_days_from_timestamps(timestamps)
    concatenated_days_by_area = concat_areas_with_city(days_pan_city, days_by_area)
    concatenated_days_by_area = remove_days_without_rolling_sum(concatenated_days_by_area)
    # Do the convolution here, using concatenated_days_by_area as our
    # data
    if USE_CONVOLUTION:
        concatenated_days_by_area = convolve_by_neighbor(concatenated_days_by_area)

    return concatenated_days_by_area

def get_days_by_area(timestamps):
    days_by_area = {}
    grouped = timestamps.groupby('Community Area')
    for name, frame in grouped:
        area_days = make_series_of_days_from_timestamps(frame, include_time_features=True)
        area_days['Violent Crime Committed?'] = area_days['Violent Crimes'].map(lambda num_crimes: num_crimes > 0)
        days_by_area[name] = area_days
    return days_by_area


def concat_areas_with_city(days_pan_city, days_by_area):
    new_column_names = ['Chicago ' + old_name for old_name in list(days_pan_city)]
    days_pan_city.columns = new_column_names
    concatenated_days_by_area = {}
    for area in days_by_area:
        area_days = days_by_area[area]
        concatenated_days_by_area[area] = area_days.join(days_pan_city)
    return concatenated_days_by_area


def extract_features_from_data_frame(crimes_by_area):
    feature_vectors_by_area = {}
    for area in crimes_by_area:
        feature_vectors_by_area[area] = crimes_by_area[area].values
    return feature_vectors_by_area


def remove_days_without_rolling_sum(concatenated_days_by_area):
    for area in concatenated_days_by_area:
        concatenated_days_by_area[area] = concatenated_days_by_area[area][30:]
    return concatenated_days_by_area

def convolve_by_neighbor(concatenated_days_by_area):
    neighbors_of_area = read_in_neighbors_csv('../config/community_area_neighbors.csv')

    for area in concatenated_days_by_area:
        convolution, convolution_week, convolution_month = generate_convolved_columns(concatenated_days_by_area, area, neighbors_of_area)

        dataframe = concatenated_days_by_area[area]

        # Add these columns to our data
        dataframe['Violent Crimes in Neighbors'] = convolution
        dataframe['Violent Crimes in Neighbors in Last Week'] = convolution_week
        dataframe['Violent Crimes in Neighbors in Last Month'] = convolution_month

    return concatenated_days_by_area

def read_in_neighbors_csv(pathname):
    neighbors_of_area = {}

    with open(pathname, 'r') as neighbor_file:
        reader = csv.reader(neighbor_file)
        for line in reader:
            neighbors_of_area[line[0]] = line[1:]

    return neighbors_of_area

def generate_convolved_columns(dataframes, area, neighbors_of_area):
    list_of_neighbors_for_area = neighbors_of_area[area]

    # Initialize our column to be the violent crimes in this area 
    convolution = dataframes[area]['Violent Crimes']
    convolution_week = dataframes[area]['Violent Crimes in Last Week']
    convolution_month = dataframes[area]['Violent Crimes in Last Month']

    # Go through all of the neighbors, and add their sets of violent crimes to our totals
    for actual_neighbor in list_of_neighbors_for_area:
        convolution = convolution + dataframes[actual_neighbor]['Violent Crimes']
        convolution_week = convolution_week + dataframes[actual_neighbor]['Violent Crimes in Last Week']
        convolution_month = convolution_month + dataframes[actual_neighbor]['Violent Crimes in Last Month']

    return [convolution, convolution_week, convolution_month]

""" Used in make_series_of_days_from_timestamps() """


def make_series_of_days_from_timestamps(timestamps, include_time_features=False):
    timestamps = extract_severity_counts(timestamps)
    days = resample_by_day(timestamps)
    if include_time_features:
        days = extract_time_features(days)
    days = extract_windows(days)
    return days


def extract_severity_counts(timestamps):
    for label in ['Violent', 'Severe', 'Minor', 'Petty']:
        timestamps[label + ' Crimes'] = [int(classification == label) for classification in timestamps['Primary Type']]
    return timestamps


def resample_by_day(timestamps):
    days = timestamps.resample('D', how='sum')
    days = days.fillna(0)
    return days


def extract_time_features(days):
    days['Month'] = days.index.map(lambda stamp: stamp.month)
    days['Weekday'] = days.index.map(lambda stamp: stamp.weekday())
    days = make_cols_categorical(days, ['Month', 'Weekday'])
    return days


def extract_windows(days):
    for label in ['Violent', 'Severe', 'Minor', 'Petty']:
        days[label + ' Crimes in Last Week'] = pd.rolling_sum(days[label + ' Crimes'], 7)
        days[label + ' Crimes in Last Month'] = pd.rolling_sum(days[label + ' Crimes'], 30)
    return days
