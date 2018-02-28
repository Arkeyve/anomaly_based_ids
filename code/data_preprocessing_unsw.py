import pandas as pd
import numpy as np

def import_and_clean(dataset):
    print("importing", dataset, "...")
    data = pd.read_csv("../dataset/" + dataset)

    print("converting nominal data to numeric...")
    cols = data.select_dtypes('object').columns
    data[cols] = data[cols].apply(lambda x: x.astype('category').cat.codes)

    return data

train = import_and_clean("UNSW_NB15_training-set.csv")
test = import_and_clean("UNSW_NB15_testing-set.csv")
