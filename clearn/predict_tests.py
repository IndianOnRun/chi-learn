import unittest
import pandas as pd
from clearn import predict


class MarkovTests(unittest.TestCase):
    def setUp(self):
        # Create index of 32 dates from arbitrary start point
        date_sequence = pd.date_range('1/1/2011', periods=32, freq='D')
        # Try to predict the last date in the sequence
        self.date_to_predict = date_sequence[-1]
        self.df = pd.DataFrame(index=date_sequence)
    
    def test_series_with_all_violent_days(self):
        # 1 means a violent crime was committed
        self.df['Violent Crime Committed?'] = [1]*32
        prediction = predict.sequential(self.df, self.date_to_predict)
        self.assertEqual(prediction, 1)

    def test_series_with_no_violent_days(self):
        # 0 Means no violent crime was committed
        self.df['Violent Crime Committed?'] = [0]*32
        prediction = predict.sequential(self.df, self.date_to_predict)
        self.assertEqual(prediction, 0)

    def test_series_with_one_violent_day(self):
        self.df['Violent Crime Committed?'] = [0, 0, 1] + [0]*29
        prediction = predict.sequential(self.df, self.date_to_predict)
        self.assertEqual(prediction, 0)

    def test_series_with_one_nonviolent_day(self):
        self.df['Violent Crime Committed?'] = [1, 1, 0] + [1]*29
        prediction = predict.sequential(self.df, self.date_to_predict)
        self.assertEqual(prediction, 1)
