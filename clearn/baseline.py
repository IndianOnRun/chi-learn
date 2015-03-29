from __future__ import division
import csv

import pandas as pd


# Function to bin crimes using mappings from a file.
def bin_from_csv(csv_name, series_to_bin):
    with open(csv_name, 'rb') as bin_file:
        unbinned_to_binned = {}
        reader = csv.reader(bin_file)
        for line in reader:
            unbinned_to_binned[line[0]] = line[1]

    return series_to_bin.map(lambda unbinned: unbinned_to_binned[unbinned])


# Function to bin crimes using the previous function and the specific mapping
# file we created.
def bin_crimes(series):
    return bin_from_csv('../config/crime_bins.csv', series)


# Function to reindex the data using the date column converted to an actual date
# format.
def reindex_by_date(data_frame):
    data_frame.index = pd.to_datetime(data_frame['Date'])

# Read in the crimes
raw_crimes = pd.read_csv('../data/crimeSample.csv')

# Get relevant features for the baseline
relevant_columns = ['Primary Type', 'Community Area', 'Date']
trimmed_crimes = raw_crimes.reindex(columns=relevant_columns)

# Bin crimes based on severity
trimmed_crimes['Primary Type'] = bin_crimes(raw_crimes['Primary Type'])

# Extract only the violent crimes, and delete the type column as it is no longer
# necessary
violent_crimes = trimmed_crimes[trimmed_crimes['Primary Type'] == "Violent"]
del violent_crimes['Primary Type']

# Reindex our data using the date, and delete the extraneous date column
reindex_by_date(violent_crimes)
del violent_crimes['Date']

# Group violent crimes by area, and resample by summing the crimes each day in
# each area
violent_crimes['Crimes'] = 1
grouped_violent = violent_crimes.groupby('Community Area')
violent_crimes_by_location = {}
for location, data_for_location in grouped_violent:
    violent_crimes_by_location[location] = data_for_location.resample('D', how='sum')
    del violent_crimes_by_location[location]['Community Area']

# Convert number of crimes per day into boolean: presence of crimes per day
for location in violent_crimes_by_location:
    violent_crimes_by_location[location]['Crime?'] = violent_crimes_by_location[location]['Crimes'].map(
        lambda num_crimes: num_crimes > 0)
    del violent_crimes_by_location[location]['Crimes']

# Generate a baseline object that calculates the count of days with crime versus
# total number of days
baseline = {}
for location in range(1, 78):
    df = violent_crimes_by_location[location]
    total_days = len(df.index)
    days_with_crime = len(df[df['Crime?'] == True])
    baseline[str(location)] = days_with_crime / total_days

# Write the new baseline to a file
writer = csv.writer(open('../config/new_baseline.csv', 'w'))
for key in baseline:
    writer.writerow([key, str(baseline[key])])
