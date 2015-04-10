import unittest
import pandas as pd
import datetime
from . import predict

class MarkovTests(unittest.TestCase):
    
    def test_series_with_all_violent_days(self):
        crime_sequence = [1]*32
        date = datetime.date.today()
        date_sequence = [date - datetime.timedelta(days=32-x) for x in range(0, 32)]
        df = pd.DataFrame(crime_sequence,index=date_sequence)
        df.columns=['Violent Crime Committed?']
        result = predict.sequential(df,date)
        self.assertEqual(result, 1)

    def test_series_with_no_violent_days(self):
        crime_sequence = [0]*32
        date = datetime.date.today()
        date_sequence = [date - datetime.timedelta(days=32-x) for x in range(0,32)]
        df = pd.DataFrame(crime_sequence,index=date_sequence)
        df.columns=['Violent Crime Committed?']
        result = predict.sequential(df,date)
        self.assertEqual(result, 0)

    def test_series_with_one_violent_day(self):
        crime_sequence = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        date = datetime.date.today()
        date_sequence = [date - datetime.timedelta(days=32-x) for x in range(0, 32)]
        df = pd.DataFrame(crime_sequence,index=date_sequence)
        df.columns=['Violent Crime Committed?']
        result = predict.sequential(df,date)
        self.assertEqual(result, 0)

    def test_series_with_one_nonviolent_day(self):
        crime_sequence = [1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        date = datetime.date.today()
        date_sequence = [date - datetime.timedelta(days=32-x) for x in range(0, 32)]
        df = pd.DataFrame(crime_sequence,index=date_sequence)
        df.columns=['Violent Crime Committed?']
        result = predict.sequential(df,date)
        self.assertEqual(result, 1)
