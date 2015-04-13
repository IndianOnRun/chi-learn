import unittest
from unittest.mock import MagicMock
from sklearn.base import BaseEstimator
import pandas as pd
import datetime
from clearn.predict import SequentialPredictor, BaselinePredictor, NonsequentialPredictor


class SequentialTests(unittest.TestCase):
    def setUp(self):
        # Create index of 32 dates from arbitrary start point
        date_sequence = pd.date_range('1/1/2011', periods=32, freq='D')
        # Try to predict the last date in the sequence
        self.date_to_predict = date_sequence[-1]
        self.time_series = pd.DataFrame(index=date_sequence)
    
    def test_series_with_all_violent_days(self):
        # 1 means a violent crime was committed
        self.time_series['Violent Crime Committed?'] = [1]*32
        predictor = SequentialPredictor(self.time_series)
        self.assertTrue(predictor.predict(self.date_to_predict))

    def test_series_with_no_violent_days(self):
        # 0 Means no violent crime was committed
        self.time_series['Violent Crime Committed?'] = [0]*32
        predictor = SequentialPredictor(self.time_series)
        self.assertFalse(predictor.predict(self.date_to_predict))

    def test_series_with_one_violent_day(self):
        self.time_series['Violent Crime Committed?'] = [0, 0, 1] + [0]*29
        predictor = SequentialPredictor(self.time_series)
        self.assertFalse(predictor.predict(self.date_to_predict))

    def test_series_with_one_nonviolent_day(self):
        self.time_series['Violent Crime Committed?'] = [1, 1, 0] + [1]*29
        predictor = SequentialPredictor(self.time_series)
        self.assertTrue(predictor.predict(self.date_to_predict))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        # Create index of 30 dates from arbitrary start point
        date_sequence = pd.date_range('1/1/2011', periods=30, freq='D')
        # Try to predict the next day
        self.date_to_predict = date_sequence[-1] + datetime.timedelta(days=1)
        self.time_series = pd.DataFrame(index=date_sequence)

    def test_majority_crime(self):
        # If more than 50% of last 30 days had crime, predict crime
        self.time_series['Violent Crime Committed?'] = [True]*16 + [False]*14
        predictor = BaselinePredictor(self.time_series)
        self.assertTrue(predictor.predict(self.date_to_predict))

    def test_majority_no_crime(self):
        # If less than 50% of last 30 days had crime, predict no crime
        self.time_series['Violent Crime Committed?'] = [True]*14 + [False]*16
        predictor = BaselinePredictor(self.time_series)
        self.assertFalse(predictor.predict(self.date_to_predict))

    def test_even_split(self):
        # If exactly 50% of last 30 days had crime, predict no crime
        self.time_series['Violent Crime Committed?'] = [True]*15 + [False]*15
        predictor = BaselinePredictor(self.time_series)
        self.assertFalse(predictor.predict(self.date_to_predict))


class NonsequentialTests(unittest.TestCase):
    def test_vector_alignment(self):
        # Mock out a generic scikit-learn classifier
        mocked_model = BaseEstimator()
        mocked_model.fit = MagicMock()
        mocked_model.predict = MagicMock(return_value=[True])

        # Create a simple data frame extending to January 15
        date_sequence = pd.date_range('1/1/2011', periods=15, freq='D')
        time_series = pd.DataFrame({
            # This column will be accessed by name to generate the targets vector.
            'Violent Crime Committed?': [True, True] + [False]*13,

            # Actual time series used for nonsequential prediction will contain more than one column.
            # However, we just need to verify that it grabs the correct slices of each column,
            # so one stand-in column will suffice.
            'Other Data': [0]*10 + [1]*5
        }, index=date_sequence)

        # Construct a NonsequentialPredictor with the mock
        predictor = NonsequentialPredictor(time_series, model=mocked_model)

        # The date to predict comes before the end of the time series,
        # so all rows from the 13th on should be discarded
        date_to_predict = datetime.date(2011, 1, 13)

        # The mock always predicts True, so predict() should return True
        self.assertTrue(predictor.predict(date_to_predict))

        # And both fit and predict should have been called
        self.assertTrue(mocked_model.fit.called)
        self.assertTrue(mocked_model.predict.called)

        # When feeding training data to the sklearn model,
        # predict() needs to align each day of the time series with whether a violent crime was committed the NEXT day.
        # Thus, the first element of the Violent Crime Committed? column should have been removed
        #  before being used as the model's targets vector because it has no previous day to partner with.
        expected_targets = [True] + [False]*11

        # Similarly, the last element of any other column (in this case, 'Other Data')
        # should only go up to the day before the day we're trying to predict
        expected_features = [[0]]*10 + [[1]]*2

        # Get the two arguments passed to mocked_model
        fit_args = mocked_model.fit.call_args
        observed_features = fit_args[0][0]
        observed_targets = fit_args[0][1]

        # Equality tests with numpy arrays are wonky, so I convert numpy arrays to Python lists
        self.assertEqual(observed_targets.tolist(), expected_targets)
        self.assertEqual(observed_features.tolist(), expected_features)

        # Confirm the correct argument was passed to predict
        print(mocked_model.predict.call_args)
        observed_day_to_predict = mocked_model.predict.call_args[0][0]
        self.assertEqual(observed_day_to_predict.tolist(), [[1]])


class PreprocessTests(unittest.TestCase):
    def test_baseline_preprocess(self):
        test_dict = {
            'Chicago': 'some_data',
            'Edgewater': pd.DataFrame({
                'Violent Crime Committed?': [True, False],
                'Irrelevant': ['right', 'meow']
            })
        }
        processed = BaselinePredictor.preprocess(test_dict)

        # Baseline's preprocess should strip out city-wide data...
        self.assertNotIn('Chicago', processed)
        # but it should leave in community area data.
        self.assertIn('Edgewater', processed)

        # It should strip out every column from the community areas' data frames...
        self.assertNotIn('Irrelevant', processed['Edgewater'])
        # except for 'Violent Crime Committed?'.
        self.assertIn('Violent Crime Committed?', processed['Edgewater'])

    def test_sequential_preprocess(self):
        test_dict = {
            # The Violent Crime Committed? column should be converted to ints
            'Edgewater': pd.DataFrame({'Violent Crime Committed?': [True, False]}),
            # preprocess() deletes a Chicago key. It's allowed to expect it, so we'll add it here.
            'Chicago': None
        }
        processed_dict = SequentialPredictor.preprocess(test_dict)
        processed_column = processed_dict['Edgewater']['Violent Crime Committed?'].values
        # [True, False] should become [1, 0]
        self.assertEqual(list(processed_column), [1, 0])


class NonsequentialPreprocessTests(unittest.TestCase):
    def setUp(self):
        # Make data frames with length of 31 to allow for window function that aggregates previous 30 days.
        test_dict = {
            'Edgewater': pd.DataFrame({
                'Violent Crimes': [1]*31,
                'Severe Crimes': [2]*31,
                'Minor Crimes': [3]*31,
                'Petty Crimes': [2]*31
            }),
            'Chicago': pd.DataFrame({
                'Violent Crimes': [5]*31,
                'Severe Crimes': [2]*31,
                'Minor Crimes': [10]*31,
                'Petty Crimes': [5]*31
            })
        }
        self.processed = NonsequentialPredictor.preprocess(test_dict)

    def test_keys(self):
        # There should no longer be a data frame for Chicago
        self.assertNotIn('Chicago', self.processed)
        # But every community area should be there
        self.assertIn('Edgewater', self.processed)

    def test_columns(self):
        # There shall be columns for the four bins of crime that day, the preceding week, and the preceding month
        crime_labels = ['Violent', 'Severe', 'Minor', 'Petty']
        column_names = set()
        for label in crime_labels:
            for frequency in ['' ' in Last Week', ' in Last Month']:
                column_names.add(label + ' Crimes' + frequency)

        time_series = self.processed['Edgewater']
        for name in column_names:
            # There shall be a column for each combination of crime type and frequency in each community area
            self.assertIn(name, time_series)
            # and in the city at large.
            self.assertIn('Chicago ' + name, time_series)

    def test_row_values(self):
        # Because the window function requires 30 days of history,
        # our 31 day dataframe should have just one day after processing
        time_series = self.processed['Edgewater']
        self.assertEqual(len(time_series), 1)

        # The 'in Last Week' columns should sum over the preceding 7 days,
        # the 'in Last Month' columns should sum over the last 30 days,
        # and the original columns should be unchanged.
        for col_name, expected_val in {
            'Violent Crimes': 1,
            'Violent Crimes in Last Week': 7,
            'Violent Crimes in Last Month': 30,
            'Severe Crimes': 2,
            'Severe Crimes in Last Week': 14,
            'Severe Crimes in Last Month': 60,
            'Minor Crimes': 3,
            'Minor Crimes in Last Week': 21,
            'Minor Crimes in Last Month': 90,
            'Petty Crimes': 2,
            'Petty Crimes in Last Week': 14,
            'Petty Crimes in Last Month': 60
        }.items():
            self.assertEqual(time_series.iloc[0][col_name], expected_val)