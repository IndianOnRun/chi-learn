import unittest
import pandas as pd
from clearn import predict


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
        predictor = predict.SequentialPredictor(self.time_series)
        self.assertTrue(predictor.predict(self.date_to_predict))

    def test_series_with_no_violent_days(self):
        # 0 Means no violent crime was committed
        self.time_series['Violent Crime Committed?'] = [0]*32
        predictor = predict.SequentialPredictor(self.time_series)
        self.assertFalse(predictor.predict(self.date_to_predict))

    def test_series_with_one_violent_day(self):
        self.time_series['Violent Crime Committed?'] = [0, 0, 1] + [0]*29
        predictor = predict.SequentialPredictor(self.time_series)
        self.assertFalse(predictor.predict(self.date_to_predict))

    def test_series_with_one_nonviolent_day(self):
        self.time_series['Violent Crime Committed?'] = [1, 1, 0] + [1]*29
        predictor = predict.SequentialPredictor(self.time_series)
        self.assertTrue(predictor.predict(self.date_to_predict))
