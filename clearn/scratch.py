
def make_feature_vectors(timestamps, use_convolution):
    days_by_area = get_days_by_area(timestamps)
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


"""
For adding history to individual training examples
"""





"""
For extracting raw data for training
"""


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


def extract_features_from_data_frame(crimes_by_area):
    feature_vectors_by_area = {}
    for area in crimes_by_area:
        feature_vectors_by_area[area] = crimes_by_area[area].values
    return feature_vectors_by_area

def get_target_classification(time_series, day):
    # Return True if violent crime was committed on day specified
    return time_series.loc[day]['Violent Crime Committed?']