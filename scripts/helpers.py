import csv


def bin_from_csv(csv_name, series_to_bin):
    with open(csv_name, 'rb') as bin_file:
        unbinned_to_binned = {}
        reader = csv.reader(bin_file)
        for line in reader:
            unbinned_to_binned[line[0]] = line[1]

    return series_to_bin.map(lambda unbinned: unbinned_to_binned[unbinned])


def make_col_categorical(data_frame, col_name):
    data_frame[col_name] = data_frame[col_name].astype('category')


def transform_from_csv(data_frame, col_name, csv_name):
    data_frame[col_name] = bin_from_csv(csv_name, data_frame[col_name])