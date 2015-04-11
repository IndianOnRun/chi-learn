import unittest
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
    pass


class PreprocessTests(unittest.TestCase):
    def test_nonsequential_preprocess(self):
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

    def test_baseline_preprocess(self):
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