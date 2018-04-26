import pandas as pd
import numpy as np

labels = pd.read_csv("../dataset/UNSW-NB15_features.csv")

def import_and_clean(dataset):
    print("importing", dataset, "...")
    data = pd.read_csv("../dataset/" + dataset, low_memory=False)
    data.columns = labels.iloc[:, 1]
    print("filling blank entries...")
    data = data.fillna(0)

    print("converting nominal data to numeric...")
    cols = data.select_dtypes('object').columns
    data[cols] = data[cols].apply(lambda x: x.astype('category').cat.codes)

    return data
