import pandas as pd
import csv
from clearn import clearn_path


def get_master_dict(csv_path):
    """

    Returns dictionary mapping each community area name and the city of chicago (key='Chicago')
    to pandas dataframes. THe dataframes are indexed by day and have the following columns:
    ['Arrest', 'Domestic', 'Violent Crimes', 'Severe Crimes', 'Minor Crimes', 'Petty Crimes', 'Violent Crime Committed?', 'Month', 'Weekday']
    There is one exception. The Chicago dataframe does not have the 'Month' and 'Weekday' column.
    """
    # Transform csv to Pandas data frame
    data_frame = pd.read_csv(csv_path)
    # Drop unnecessary columns and reidex crimes by date
    timestamps = make_clean_timestamps(data_frame)
    # From crime timestamps, create dictionary mapping community area names
    #   to pandas data frames resampled by day
    days_by_area = get_days_by_area(timestamps)
    # Add an extra mapping to include what happened in all of Chicago on each day
    days_by_area['Chicago'] = make_series_of_days_from_timestamps(timestamps)
    return days_by_area


""" Used in make_clean_timestamps() """


def make_clean_timestamps(data_frame):
    data_frame = drop_all_columns_but(data_frame, ['Date', 'Primary Type', 'Community Area', 'Arrest', 'Domestic'])
    data_frame = convert_comm_area_nums_to_names(data_frame)
    data_frame = transform_from_csv(data_frame, 'Primary Type', clearn_path('config/crime_bins.csv'))
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
    data_frame = transform_from_csv(data_frame, 'Community Area', clearn_path('config/community_areas.csv'))
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


""" Used in get_days_by_area() """


def get_days_by_area(timestamps):
    days_by_area = {}
    grouped = timestamps.groupby('Community Area')
    for name, frame in grouped:
        area_days = make_series_of_days_from_timestamps(frame)
        area_days['Violent Crime Committed?'] = area_days['Violent Crimes'].map(lambda num_crimes: num_crimes > 0)
        area_days = extract_time_features(area_days)
        days_by_area[name] = area_days
    return days_by_area


def extract_time_features(days):
    days['Month'] = days.index.map(lambda stamp: stamp.month)
    days['Weekday'] = days.index.map(lambda stamp: stamp.weekday())
    days = make_cols_categorical(days, ['Month', 'Weekday'])
    return days

""" Used in make_series_of_days_from_timestamps() """


def make_series_of_days_from_timestamps(timestamps):
    timestamps = extract_severity_counts(timestamps)
    days = resample_by_day(timestamps)
    return days


def extract_severity_counts(timestamps):
    for label in ['Violent', 'Severe', 'Minor', 'Petty']:
        timestamps[label + ' Crimes'] = [int(classification == label) for classification in timestamps['Primary Type']]
    return timestamps


def resample_by_day(timestamps):
    days = timestamps.resample('D', how='sum')
    days = days.fillna(0)
    return days