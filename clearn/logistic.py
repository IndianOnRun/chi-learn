import pandas as pd
import helpers as hlp


def get_target_vectors(data_frame):
    """
    Returns dictionary mapping community area names
    to boolean-valued numpy vectors. Each value represents whether
    a violent crime was committed on a day. They are listed in ascending order
    of time
    e.g. Edgewater -> [True, False, ...]

    """

    crimes = hlp.drop_all_columns_but(data_frame, ['Date', 'Primary Type', 'Community Area'])
    crimes = hlp.convert_comm_area_nums_to_names(crimes)
    crimes = hlp.drop_nonviolent_crimes(crimes)
    crimes = hlp.reindex_by_date(crimes)

    # Add a new column that we'll use to sum crimes per day
    crimes['Violent Crime'] = 1
    grouped = crimes.groupby('Community Area')
    targets = {}
    for name, frame in grouped:
        # Make one row per day
        area_crimes = frame.resample('D', how='sum')
        # Create new column that is true iff at least one violent crime was committed that day
        area_crimes['Violent Crime Committed'] = area_crimes['Violent Crime'].map(
            lambda num_crimes: num_crimes > 0)
        # Store the raw True/False values in the target vector
        targets[name] = area_crimes['Violent Crime Committed'].values

    return targets