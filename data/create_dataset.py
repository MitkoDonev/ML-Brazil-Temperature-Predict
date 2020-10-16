import pandas as pd


def create_dataset(dataset, columns):
    return pd.DataFrame(data=dataset, columns=columns)
