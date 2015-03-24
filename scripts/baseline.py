from __future__ import division
import pandas as pd
import csv
from helpers import bin_from_csv
from datetime import date

BEGIN_DATE = date(2001, 1, 1)


def create_baseline(end_date):

    # Load all crimes and trim the columns we won't use
    raw_crimes = pd.read_csv('data/Crimes_-_2001_to_present.csv')
    relevant_columns = ['Primary Type','Community Area']
    trimmed_crimes = raw_crimes.reindex(columns=relevant_columns)

    # Filter out nonviolent crimes
    trimmed_crimes['Primary Type'] = bin_from_csv('config/crime_bins.csv',
                                                  trimmed_crimes['Primary Type'])
    violent_crimes = trimmed_crimes[trimmed_crimes['Primary Type'] == "1"]

    # Create a mapping from community area names to dataframes
    areas_with_crimes = violent_crimes.groupby('Community Area')
    crimes_by_area = {}
    with open('config/community_areas.csv', 'rb') as area_file:
        area_dict = {}
        reader = csv.reader(area_file)
        for line in reader:
            area_dict[int(line[0])] = line[1]
        for name, group in areas_with_crimes:
            # Only include valid community area numbers
            if int(name) in range(1, 78):
                crimes_by_area[area_dict[int(name)]] = group

    # Create mapping from community area names
    # to average number of violent crimes per day
    avg_crimes_by_area = {}
    num_days = days_since_2001(end_date)
    for area in crimes_by_area:
        avg_crimes_by_area[area] = len(crimes_by_area[area].index)/num_days

    # Write neighborhood to crime/day mapping to csv
    with open('config/baseline.csv', 'wb') as baseline_file:
        writer = csv.writer(baseline_file)
        for area in avg_crimes_by_area:
            writer.writerow([area, avg_crimes_by_area[area]])
        baseline_file.close()


def days_since_2001(end_date):
    start_date = date(2001, 1, 1)
    delta = end_date - start_date
    return delta.days