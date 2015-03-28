import pandas as pd
import csv


# TODO: Drop examples with missing data because of moving window function
class TrainingSetFactory:
    """Given a csv of Chicago crime data,
    Creates a training set with feature and target vectors"""

    def __init__(self, data_csv_path):
        self.data_frame = pd.read_csv(data_csv_path)
        self.make_clean_timestamps()
        self.target_vectors = TargetVectorsFactory(self.data_frame)
        self.feature_vectors_by_area = self.get_feature_vectors()

    def make_clean_timestamps(self):
        self.drop_all_columns_but(['Date', 'Primary Type', 'Community Area', 'Arrest', 'Domestic'])
        self.convert_comm_area_nums_to_names()
        self.transform_from_csv('Primary Type', '../config/crime_bins.csv')
        self.reindex_by_date()
        self.data_frame = make_cols_categorical(self.data_frame, ['Primary Type', 'Community Area'])

    def get_feature_vectors(self):
        area_to_daily_crimes = self.get_daily_crimes_by_area()
        pan_chicago_daily_crimes = DailyCrimesFactory(self.crimes).crimes
        concatenated_crimes_by_area = self.concat_city_with_areas(pan_chicago_daily_crimes, area_to_daily_crimes)
        return self.extract_features_from_data_frame(concatenated_crimes_by_area)

    @staticmethod
    def extract_features_from_data_frame(crimes_by_area):
        feature_vectors_by_area = {}
        for area in crimes_by_area:
            feature_vectors_by_area[area] = crimes_by_area[area].values
        return feature_vectors_by_area

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
            area_crimes = DailyCrimesFactory(frame, include_time_features=True).crimes
            area_to_daily_crimes[name] = area_crimes
        return area_to_daily_crimes

    @staticmethod
    def concat_city_with_areas(pan_chicago_crimes, crimes_by_area):
        column_names = list(pan_chicago_crimes)
        concatenated_crimes_by_area = {}
        for area in crimes_by_area:
            area_crimes = crimes_by_area[area]
            for col in column_names:
                area_crimes['Chicago ' + col] = pan_chicago_crimes[col]
            concatenated_crimes_by_area[area] = area_crimes
        return concatenated_crimes_by_area


class TargetVectorsFactory:
    def __init__(self, crimes):
        self.crimes = crimes
        self.drop_nonviolent_crimes()
        # Add a new column that we'll use to sum crimes per day
        self.crimes['Violent Crime'] = 1
        grouped = self.crimes.groupby('Community Area')
        self.targets = {}
        for name, frame in grouped:
            # Make one row per day
            area_crimes = frame.resample('D', how='sum')
            # Create new column that is true iff at least one violent crime was committed that day
            area_crimes['Violent Crime Committed'] = area_crimes['Violent Crime'].map(
                lambda num_crimes: num_crimes > 0)
            # Store the raw True/False values in the target vector
            self.targets[name] = area_crimes['Violent Crime Committed'].values

    def drop_nonviolent_crimes(self):
        self.crimes = self.crimes[self.crimes['Primary Type'] == 'Violent']


class DailyCrimesFactory:
    """
        Given a pandas dataframe of "clean" timestamps,
        Create a dataframe indexed by day with
        extra rolling window features,
    """

    def __init__(self, crimes, include_time_features=False):
        self.crimes = crimes
        self.extract_severity_counts()
        self.resample()
        if include_time_features:
            self.extract_time_features()
        self.extract_windows()

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

def make_cols_categorical(data_frame, col_names):
        for name in col_names:
            data_frame[name] = data_frame[name].astype('category')
        return data_frame