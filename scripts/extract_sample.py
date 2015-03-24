with open('allCrimes.csv', 'rb') as all_crimes:
    with open('crimeSample.csv', 'w') as some_crimes:
        for line_num in range(110000):
            some_crimes.write(all_crimes.readline())