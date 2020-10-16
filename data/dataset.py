import pandas as pd
import numpy as np


def load_dataset():
    dataset = pd.read_csv("Vitoria.csv")

    cities = ['Vitória']
    df_filtered = dataset[dataset.city.isin(cities)]

    df_filtered.drop(['city', 'wsid', 'wsnm', 'elvt', 'lat', 'lon', 'inme', 'prov', 'tmax', 'dmax', 'tmin', 'dmin', 'hmax', 'hmin', 'smax', 'smin'],
                     axis='columns', inplace=True)

    df_filtered = df_filtered.replace(np.nan, 0)

    df_filtered.drop(['entry'], axis='columns', inplace=True)

    return df_filtered
