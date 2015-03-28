import pandas as pd
import csv


class TrainingExample:

    def __init__(self):
        pass


class DailyCrimesFactory:
    """
        Given a pandas dataframe of "clean" timestamps,
        Create a dataframe indexed by day with
        extra rolling window features,
    """

    def __init__(self, stamps):
        self.crimes = stamps
        self.extract_severity_counts()
        self.resample()
        self.extract_time_features()
        #TODO: Extract time window features

    def extract_time_features(self):
        self.crimes['Year'] = self.crimes.index.map(lambda stamp: stamp.year)
        self.crimes['Month'] = self.crimes.index.map(lambda stamp: stamp.month)
        self.crimes['Weekday'] = self.crimes.index.map(lambda stamp: stamp.weekday())
        self.crimes = make_cols_categorical(self.crimes, ['Year', 'Month', 'Weekday'])

    def extract_severity_counts(self):
        for severity in ['Violent', 'Severe', 'Minor', 'Petty']:
            self.crimes[severity + ' Crimes'] = [int(category == severity) for category in self.crimes['Primary Type']]

    def resample(self):
        self.crimes.resample('D', how='sum')

    def extract_windows(self):
        for severity in ['Violent', 'Severe', 'Minor', 'Petty']:
            self.crimes[severity + ' Crimes in Last Week'] = \
                pd.rolling_sum(self.crimes[severity + ' Crimes'], 7)
            self.crimes[severity + ' Crimes in Last Month'] = \
                pd.rolling_sum(self.crimes[severity + ' Crimes'], 30)


class TrainingSetFactory:
    """Given a csv of Chicago crime data,
    Creates a training set with feature and target vectors"""

    def __init__(self, data_csv_path):
        self.data_frame = pd.read_csv(data_csv_path)
        self.make_clean_timestamps()
        area_to_daily_crimes = self.get_daily_crimes_by_area()
        pan_chicago_daily_crimes = DailyCrimesFactory(self.crimes).crimes
        self.feature_vectors = self.concat_city_with_areas(pan_chicago_daily_crimes, area_to_daily_crimes)


    def make_clean_timestamps(self):
        self.drop_all_columns_but(['Date', 'Primary Type', 'Community Area', 'Arrest', 'Domestic'])
        self.convert_comm_area_nums_to_names()
        self.transform_from_csv('Primary Type', '../config/crime_bins.csv')
        self.reindex_by_date()
        self.data_frame = make_cols_categorical(self.data_frame, ['Primary Type', 'Community Area'])

    def drop_all_columns_but(self, relevant_columns):
        self.data_frame = self.data_frame.reindex(columns=relevant_columns)

    def convert_comm_area_nums_to_names(self):
        # Remove rows with invalid community area numbers
        self.data_frame = self.data_frame[self.data_frame['Community Area'] > 0]
        # Convert numbers to strings for easy binning
        self.data_frame['Community Area'] = self.data_frame['Community Area'].map(lambda num: str(num))
        self.transform_from_csv('Community Area', '../config/community_areas.csv')

    def transform_from_csv(self, col_name, csv_name):
        with open(csv_name, 'rb') as bin_file:
            unbinned_to_binned = {}
            reader = csv.reader(bin_file)
            for line in reader:
                unbinned_to_binned[line[0]] = line[1]

        self.data_frame[col_name] = self.data_frame[col_name].map(lambda unbinned: unbinned_to_binned[unbinned])

    def reindex_by_date(self):
        self.data_frame.index = pd.to_datetime(self.data_frame['Date'])
        self.data_frame.drop('Date', 1, inplace=True)

    def get_daily_crimes_by_area(self):
        area_to_daily_crimes = {}
        grouped = self.data_frame.groupby('Community Area')
        for name, frame in grouped:
            area_crimes = DailyCrimesFactory(frame).crimes
            # Concatenate the chicago stats to the community area stats
            area_to_daily_crimes[name] = area_crimes.values
        return area_to_daily_crimes

    def concat_city_with_area(self, pan_chicago_crimes, crimes_by_area):
        pass

        for area in crimes_by_area:
            area_crimes = crimes_by_area[area]
            area_crimes

def make_cols_categorical(data_frame, col_names):
        for name in col_names:
            data_frame[name] = data_frame[name].astype('category')
        return data_frame


def get_target_vectors(data_frame):
    """
    Returns dictionary mapping community area names
    to boolean-valued numpy vectors. Each value represents whether
    a violent crime was committed on a day. They are listed in ascending order
    of time
    e.g. Edgewater -> [True, False, ...]

    """

    crimes = make_clean_timestamps(data_frame)
    crimes = hlp.drop_nonviolent_crimes(crimes)

    # Add a new column that we'll use to sum crimes per day
    crimes['Violent Crime'] = 1
    grouped = crimes.groupby('Community Area')
    targets = {}
    for name, frame in grouped:
        # Make one row per day
        area_crimes = frame.resample('D', how='sum')
        # Create new column that is true iff at least one violent crime was committed that day
        area_crimes['Violent Crime Committed'] = area_crimes['Violent Crime'].map(
            lambda num_crimes: num_crimes > 0)
        # Store the raw True/False values in the target vector
        targets[name] = area_crimes['Violent Crime Committed'].values

    return targets


def get_feature_vectors(data_frame):
    crimes = make_clean_timestamps(data_frame)
    pan_chicago_series = get_summary(crimes)
    community_area_features = {}
    grouped_by_community = crimes.groupby('Community Area')
    for name, frame in grouped_by_community:
        area_series = get_summary(frame)
        # Concatenate the chicago stats to the community area stats
        community_area_features[name] = area_series.values

def get_summary(crimes):
    """  """
    pass


def drop_nonviolent_crimes(data_frame):
    data_frame = transform_from_csv(data_frame, 'Primary Type', '../config/crime_bins.csv')
    data_frame = data_frame[data_frame['Primary Type'] == 'Violent']
    return data_frame