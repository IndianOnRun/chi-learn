import unittest
import csv
import pandas as pd
from clearn import munge
from clearn import clearn_path
from datetime import date


class TestTimestampCreation(unittest.TestCase):
    def setUp(self):
        fixture_path = clearn_path('data/fixtures/mediumCrimeSample.csv')
        data_frame = pd.read_csv(fixture_path)
        self.timestamps = munge.make_clean_timestamps(data_frame)

    def test_column_names(self):
        # Timestamps data frames shall have precisely these columns
        expected_column_names = {'Primary Type', 'Community Area', 'Arrest', 'Domestic'}
        actual_column_names = set(self.timestamps.columns)
        self.assertEqual(expected_column_names, actual_column_names)

    def test_data_types(self):
        # Timestamps data frames shall have the following pandas data types for each column
        dtypes = self.timestamps.dtypes
        self.assertEqual(dtypes['Primary Type'], 'category')
        self.assertEqual(dtypes['Community Area'], 'category')
        self.assertEqual(dtypes['Arrest'], 'bool')
        self.assertEqual(dtypes['Domestic'], 'bool')

    def test_index(self):
        # Timestamps data frames shall be indexed by time stamps
        # Perform duck typing check to ensure that timestamps are convertible to dates
        try:
            self.timestamps.index[0].date()
        except TypeError:
            self.fail()

    def test_unexpected_crime_type(self):
        # Create data frame with two crimes of type "Cat Rental Fraud",
        #   which is not on record in Chicago
        weird_crimes = pd.DataFrame({
            'Primary Type': 'Cat Rental Fraud',
            'Community Area': [77]*2,
            'Arrest': True,
            'Domestic': True
        })

        # As implemented now, raises a KeyError
        #   But that's an implementation detail.
        #   As long as it raises an exception, I'm happy.
        with self.assertRaises(Exception):
            munge.make_clean_timestamps(weird_crimes)

    def test_known_sample(self):
        # Take first five crimes from small sample
        fixture_path = clearn_path('data/fixtures/tiniestCrimeSample.csv')
        data_frame = pd.read_csv(fixture_path)
        observed_timestamps = munge.make_clean_timestamps(data_frame)

        # Construct expected timestamps
        times = pd.TimeSeries([
            pd.Timestamp('2015-02-27 23:58:00'),
            pd.Timestamp('2015-02-27 23:55:00'),
            pd.Timestamp('2015-02-27 23:53:00'),
            pd.Timestamp('2015-02-27 23:49:00'),
            pd.Timestamp('2015-02-27 23:45:00')
        ])

        expected_timestamps = pd.DataFrame({
            'Primary Type': pd.Categorical(['Petty',
                                            'Violent',
                                            'Violent',
                                            'Petty',
                                            'Minor']),
            'Community Area': pd.Categorical(['South Shore',
                                              'Roseland',
                                              'Humboldt Park',
                                              'Humboldt Park',
                                              'West Englewood']),
            'Arrest': [True, True, True, True, False],
            'Domestic': [False, False, True, False, False],


        }, index=times)
        expected_timestamps.index.name = 'Date'

        # I would have liked to just compare the dataframes
        # But I would have had to specify expected_timestamps in too much detail.
        # Better to just test that the expected content is present in each column.
        self.assertTrue(self.np_arrs_equal(observed_timestamps.index.values, expected_timestamps.index.values))
        for column in expected_timestamps.columns:
            self.assertTrue(self.np_arrs_equal(expected_timestamps[column].values, observed_timestamps[column].values))

    @staticmethod
    def np_arrs_equal(arr1, arr2):
        """
        Equality testing numpy arrays is unintuitive. == performs element-wise comparison and returns a boolean array.
        You then check if the entire boolean array is true.
        Explanation here: http://stackoverflow.com/questions/23949839/valueerror-the-truth-value-of-an-array-with-more-than-one-element-is-ambiguous
        """
        return (arr1 == arr2).all()


class TestMakeDays(unittest.TestCase):
    def setUp(self):
        # This fixture has records of two crimes committed on the same day in Humboldt Park
        fixture_path = clearn_path('data/fixtures/humboldtTwoCrimes.csv')
        data_frame = pd.read_csv(fixture_path)
        timestamps = munge.make_clean_timestamps(data_frame)
        latest_day = timestamps.index[-1]
        self.time_series = munge.make_series_of_days_from_timestamps(timestamps, latest_day)

    def test_index(self):
        # Resampling by day should produce a time series with one element for each day from Jan 1, 2001
        # to Feb 27, 2015, the day the fixture's crimes were committed on
        index = self.time_series.index

        # The Chciago dataset begins on Jan 1, 2001
        expected_start = date(2001, 1, 1)
        self.assertEqual(expected_start, index[0].to_datetime().date())

        # Each time series should extend to the latest available day in the dataset
        expected_end = date(2015, 2, 27)
        self.assertEqual(expected_end, index[-1].to_datetime().date())

        # Every day in between should be present
        self.assertEqual(len(self.time_series.index), (expected_end - expected_start).days + 1)

    def test_column_names(self):
        # Timeseries data frames shall have precisely these columns
        expected_column_names = {'Arrest', 'Domestic',
                                 'Violent Crimes', 'Severe Crimes', 'Minor Crimes', 'Petty Crimes'}
        actual_column_names = set(self.time_series.columns)
        self.assertEqual(expected_column_names, actual_column_names)

    def test_summation(self):
        # When resampling, make_series_of_days_from_timestamps sums types of crimes and boolean values
        sums = {
            'Arrest': 2,
            'Domestic': 1,
            'Violent Crimes': 1,
            'Severe Crimes': 0,
            'Minor Crimes': 0,
            'Petty Crimes': 1
        }

        for column_name, sum in sums.items():
            # Coerce the float value of the last day of the time series to an int
            # and assert that it is equal to the expected sum
            self.assertEqual(sum, int(self.time_series[column_name][-1]), 'Failed on ' + column_name)


class TestMasterDict(unittest.TestCase):
    def setUp(self):
        fixture_path = clearn_path('data/fixtures/mediumCrimeSample.csv')
        self.master_dict = munge.make_master_dict(fixture_path)

    def test_all_community_areas_present(self):
        # The community_areas csv should map community area numbers to names
        comm_areas_path = clearn_path('config/community_areas.csv')
        with open(comm_areas_path, 'r') as comm_file:
            comm_reader = csv.reader(comm_file)
            comm_areas = [row[1] for row in comm_reader]

        # There are 77 community areas in Chicago
        self.assertEqual(77, len(comm_areas))

        # All community areas should have a key in the master_dict
        master_keys = set(self.master_dict.keys())
        for area in comm_areas:
            self.assertIn(area, master_keys)

    def test_chicago_present(self):
        self.assertIn('Chicago', self.master_dict.keys())

    def test_each_time_series_has_same_length(self):
        # Each time series in the master dict should start and end on the same day
        lengths = [len(time_series) for time_series in self.master_dict.values()]
        self.assertTrue(all([length == lengths[0] for length in lengths]))
