import time
from data_preprocessing_unsw import import_and_clean
from evaluation import evaluate
from sklearn.naive_bayes import GaussianNB

train = import_and_clean("UNSW-NB15_1.csv")
test = import_and_clean("UNSW-NB15_2.csv")

nb = GaussianNB()

print("training...")
start = time.time()
nb.fit(train.iloc[:, [9, 36]], train.iloc[:, -1])
end = time.time()

ttf = (end - start)

print("testing...")
start = time.time()
y_nb = nb.predict(test.iloc[:, [9, 36]])
end = time.time()

ttp = (end - start)

print("evaluating...")
evaluate(y_nb, test.iloc[:, -1])

print("ttf = \t\t", ttf)
print("ttp = \t\t", ttp)
