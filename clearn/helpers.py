import csv
import pandas as pd


def bin_from_csv(csv_name, series_to_bin):
    with open(csv_name, 'rb') as bin_file:
        unbinned_to_binned = {}
        reader = csv.reader(bin_file)
        for line in reader:
            unbinned_to_binned[line[0]] = line[1]

    return series_to_bin.map(lambda unbinned: unbinned_to_binned[unbinned])


def make_col_categorical(data_frame, col_name):
    data_frame[col_name] = data_frame[col_name].astype('category')
    return data_frame


def transform_from_csv(data_frame, col_name, csv_name):
    data_frame[col_name] = bin_from_csv(csv_name, data_frame[col_name])
    return data_frame


def reindex_by_date(data_frame):
    data_frame.index = pd.to_datetime(data_frame['Date'])
    data_frame.drop('Date', 1, inplace=True)
    return data_frame


def drop_all_columns_but(data_frame, relevant_columns):
    return data_frame.reindex(columns=relevant_columns)


def convert_comm_area_nums_to_names(data_frame):
    # Remove rows with invalid community area numbers
    data_frame = data_frame[data_frame['Community Area'] > 0]
    # Convert numbers to strings for easy binning
    data_frame['Community Area'] = data_frame['Community Area'].map(lambda num: str(num))
    data_frame = transform_from_csv(data_frame, 'Community Area', '../config/community_areas.csv')
    return data_frame


def drop_nonviolent_crimes(data_frame):
    data_frame = transform_from_csv(data_frame, 'Primary Type', '../config/crime_bins.csv')
    data_frame = data_frame[data_frame['Primary Type'] == 'Violent']
    return data_frame


def extract_time_features(time_series):
    time_series['Year'] = time_series.index.map(lambda stamp: stamp.year)
    time_series['Month'] = time_series.index.map(lambda stamp: stamp.month)
    time_series['Weekday'] = time_series.index.map(lambda stamp: stamp.weekday())
    time_series = make_col_categorical(time_series, 'Month')
    time_series = make_col_categorical(time_series, 'Weekday')
    return time_series


def extract_severity_counts(data_frame):
    for severity in ['Violent', 'Severe', 'Minor', 'Petty']:
        data_frame[severity + ' Crimes'] = [int(category == severity) for category in data_frame['Primary Type']]
    return data_frame