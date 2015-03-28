with open('../data/Crimes_-_2001_to_present.csv', 'rb') as all_crimes:
    with open('../data/mediumCrimeSample.csv', 'w') as some_crimes:
        for line_num in range(10000):
            some_crimes.write(all_crimes.readline())