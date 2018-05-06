import time
import sys
from data_preprocessing_unsw import import_and_clean
from evaluation import evaluate
from sklearn.neighbors import KNeighborsClassifier

train = import_and_clean("UNSW-NB15_1.csv")
test = import_and_clean("UNSW-NB15_2.csv")

knn = KNeighborsClassifier(n_neighbors = int(sys.argv[1]))

print("training...")
start = time.time()
knn.fit(train.iloc[:, :-2], train.iloc[:, -1])
end = time.time()

ttf = (end - start)

print("testing...")
start = time.time()
y_knn = knn.predict(test.iloc[:, :-2])
end = time.time()

ttp = (end - start)

print("evaluating...")
evaluate(y_knn, test.iloc[:, -1])

print("ttf = \t\t", ttf)
print("ttp = \t\t", ttp)
