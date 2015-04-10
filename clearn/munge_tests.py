import unittest
import csv
from clearn import munge
from clearn import clearn_path

class TestTimestampCreation(unittest.TestCase):
    def test_column_names(self):
        pass

    def test_data_types(self):
        pass

    def test_index(self):
        pass

    def test_unexpected_crime_type(self):
        pass

    def test_known_sample(self):
        pass


class TestMakeDays(unittest.TestCase):
    def test_index(self):
        pass

    def test_column_names(self):
        pass

    def test_known_sample(self):
        pass


class TestMasterDict(unittest.TestCase):
    def setUp(self):
        fixture_path = clearn_path('data/fixtures/mediumCrimeSample.csv')
        self.master_dict = munge.get_master_dict(fixture_path)

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
        pass

    def test_every_frame_has_same_days(self):
        pass