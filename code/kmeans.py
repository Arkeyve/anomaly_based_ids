from sklearn.cluster import KMeans

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
print(result[(0, 0)], " TN")
print(result[(0, 1)], " FN")
print(result[(1, 0)], " FP")
print(result[(1, 1)], " TP")
accuracy = (result[(1, 1)] + result[(0, 0)]) / y_kmeans.shape[0]
print("accuracy = ", accuracy)
