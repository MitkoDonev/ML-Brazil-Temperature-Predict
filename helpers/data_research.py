import matplotlib.pyplot as plt


def initial_values(df, column):
    years = set()
    for year in df['yr']:
        years.add(year)

    for year in years:
        df_filtered_by_year = df[df.yr.isin([f'{year}'])]
        tota_values = df_filtered_by_year[column].count()

        minimal = min(df_filtered_by_year[column])
        maximum = max(df_filtered_by_year[column])

        print(f'YEAR: {year}')
        print(f'MINIMAL: {minimal}')
        print(f'MAXIMUM: {maximum}')

        plt.figure(figsize=(8, 8))
        plt.plot(df_filtered_by_year[column][:tota_values])
        plt.show()


def data_preparation(df, column, average):
    df_arr = df[column].to_numpy()

    for i in range(0, len(df_arr)):
        if df_arr[i] == 0:
            if df_arr[i - 1] == 0:
                average = (average + df_arr[i + 1]) / 2
                df_arr[i] = round(average, 2)
            elif df_arr[i + 1] == 0:
                df_arr[i] = round(average, 2)
            else:
                average = (df_arr[i - 1] + df_arr[i + 1]) / 2
                df_arr[i] = round(average, 2)

    df[column] = df_arr


def data_research(df, column):
    years = set()
    for year in df['yr']:
        years.add(year)

    for year in years:
        df_filtered_by_year = df[df.yr.isin([f'{year}'])]
        tota_values = df_filtered_by_year[column].count()

        if column != 'prcp':

            average = df_filtered_by_year[column].mean()

            data_preparation(df, column, average)
            data_preparation(df_filtered_by_year, column, average)

        mean = df_filtered_by_year[column].mean()
        standard_deviation = df_filtered_by_year[column].std()

        minimal = min(df_filtered_by_year[column])
        maximum = max(df_filtered_by_year[column])

        print(f'YEAR: {year}')
        print(f'MEAN: {mean}')
        print(f'MINIMAL: {minimal}')
        print(f'MAXIMUM: {maximum}')
        print(f'STANDARD DEVIATION: {standard_deviation}')

        df_filtered_by_year['mean'] = mean
        df_filtered_by_year['plus_three_sigma'] = mean + \
            (3 * standard_deviation)
        df_filtered_by_year['minus_three_sigma'] = mean - \
            (3 * standard_deviation)

        plt.figure(figsize=(8, 8))
        plt.plot(df_filtered_by_year[column][:tota_values])
        plt.plot(df_filtered_by_year['mean'][:tota_values])
        plt.plot(df_filtered_by_year['plus_three_sigma'][:tota_values])
        plt.plot(df_filtered_by_year['minus_three_sigma'][:tota_values])
        plt.show()
