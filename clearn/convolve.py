import csv

def convolve_by_neighbor(concatenated_days_by_area):
    neighbors_of_area = read_in_neighbors_csv('../config/community_area_neighbors.csv')

    for area in concatenated_days_by_area:
        convolution, convolution_week, convolution_month = generate_convolved_columns(concatenated_days_by_area, area, neighbors_of_area)

        dataframe = concatenated_days_by_area[area]

        # Add these columns to our data
        dataframe['Violent Crimes in Neighbors'] = convolution
        dataframe['Violent Crimes in Neighbors in Last Week'] = convolution_week
        dataframe['Violent Crimes in Neighbors in Last Month'] = convolution_month

    return concatenated_days_by_area


def read_in_neighbors_csv(pathname):
    neighbors_of_area = {}

    with open(pathname, 'r') as neighbor_file:
        reader = csv.reader(neighbor_file)
        for line in reader:
            neighbors_of_area[line[0]] = line[1:]

    return neighbors_of_area


def generate_convolved_columns(dataframes, area, neighbors_of_area):
    list_of_neighbors_for_area = neighbors_of_area[area]

    # Initialize our column to be the violent crimes in this area
    convolution = dataframes[area]['Violent Crimes']
    convolution_week = dataframes[area]['Violent Crimes in Last Week']
    convolution_month = dataframes[area]['Violent Crimes in Last Month']

    # Go through all of the neighbors, and add their sets of violent crimes to our totals
    for actual_neighbor in list_of_neighbors_for_area:
        convolution = convolution + dataframes[actual_neighbor]['Violent Crimes']
        convolution_week = convolution_week + dataframes[actual_neighbor]['Violent Crimes in Last Week']
        convolution_month = convolution_month + dataframes[actual_neighbor]['Violent Crimes in Last Month']

    return [convolution, convolution_week, convolution_month]