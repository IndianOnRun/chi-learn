def make_raw_data_target_pairs(data_csv_path, use_convolution=True):
    frames_by_area = make_training_set(data_csv_path, use_convolution)
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


def make_training_set(data_csv_path, use_convolution):
    data_frame = pd.read_csv(data_csv_path)
    timestamps = make_clean_timestamps(data_frame)
    return make_feature_vectors(timestamps, use_convolution)

def make_feature_vectors(timestamps, use_convolution):
    days_by_area = get_days_by_area(timestamps)
    # days_pan_city = make_series_of_days_from_timestamps(timestamps)
    # concatenated_days_by_area = concat_areas_with_city(days_pan_city, days_by_area)
    # concatenated_days_by_area = remove_days_without_rolling_sum(concatenated_days_by_area)
    # Do the convolution here, using concatenated_days_by_area as our
    # data
    # if use_convolution:
    #     concatenated_days_by_area = convolve_by_neighbor(concatenated_days_by_area)

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