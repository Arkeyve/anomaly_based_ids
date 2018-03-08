from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

labels = pd.read_csv("../dataset/UNSW-NB15_features.csv")

def import_and_clean(dataset):
    print("importing", dataset, "...")
    data = pd.read_csv("../dataset/" + dataset, low_memory = False)
    data.columns = labels.iloc[:, 1]
    data = data.fillna(0)

    print("converting nominal data to numeric...")
    cols = data.select_dtypes('object').columns
    data[cols] = data[cols].apply(lambda x: x.astype('category').cat.codes)

    return data

train = import_and_clean("UNSW-NB15_1.csv")
test = import_and_clean("UNSW-NB15_2.csv")

print("using 2 clusters (normal/attack)...")
kmeans = KMeans(n_clusters = 2)

print("training using training set...")
kmeans.fit(train.iloc[:, :-2])

print("predicting on test set...")
y_kmeans = kmeans.predict(test.iloc[:, :-2])

print("comparing...")
# in comparison array:
# 0, 0 --> TN
# 0, 1 --> FN
# 1, 0 --> FP
# 1, 1 --> TP
comparison = list(zip(y_kmeans, test.iloc[:, -1]))

result = {
    (0, 0): 0,
    (0, 1): 0,
    (1, 0): 0,
    (1, 1): 0
}

for res in comparison:
    result[res] = result[res] + 1

print("in", y_kmeans.shape[0], "records, the model predicts")
print("True Negative:\t", result[(0, 0)])
print("False Negative:\t", result[(0, 1)])
print("False Positive:\t", result[(1, 0)])
print("True Positive:\t", result[(1, 1)])
accuracy = (result[(1, 1)] + result[(0, 0)]) / y_kmeans.shape[0]
print("accuracy = ", accuracy)
