with open('data/Crimes_-_2001_to_present.csv', 'rb') as all_crimes:
    with open('data/crimeSample.csv', 'w') as some_crimes:
        for line_num in range(110000):
            some_crimes.write(all_crimes.readline())