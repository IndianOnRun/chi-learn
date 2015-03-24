import sys
import csv
import time
import pandas as pd


def main():
    if len(sys.argv) < 2:
        sys.exit("Please enter the name of the csv you want to change to the chi-learn format.")
    reformat(sys.argv[1])


def reformat(csv_name):
    raw_crimes = pd.read_csv(csv_name)
    columns_we_want = ['ID', 'Date', 'Primary Type', 'Description', 'Location Description', 'Arrest', 'Domestic',
                       'Community Area', 'Year']
    raw_crimes = raw_crimes.reindex(columns=columns_we_want)
    raw_crimes['Date'] = raw_crimes['Date'].apply(lambda time_str: time_from_string(time_str))
    raw_crimes['Location Description'] = bin_locations(raw_crimes['Location Description'])
    return raw_crimes


def time_from_string(crime_str):
    return time.strptime(crime_str, '%m/%d/%Y %I:%M:%S %p')


def bin_locations(locations):
    with open('locations.csv', 'rb') as location_file:
        location_dict = {}
        reader = csv.reader(location_file)
        for line in reader:
            location_dict[line[0]] = line[1]

    return locations.map(lambda loc: location_dict[loc])


def bin_from_csv(csv_name, series_to_bin):
    with open(csv_name, 'rb') as bin_file:
        unbinned_to_binned = {}
        reader = csv.reader(bin_file)
        for line in reader:
            unbinned_to_binned[line[0]] = line[1]

    return series_to_bin.map(lambda unbinned: unbinned_to_binned[unbinned])


def bin_crimes(series):
    return bin_from_csv('crime_bins.csv')

if __name__ == '__main__':
    main()
