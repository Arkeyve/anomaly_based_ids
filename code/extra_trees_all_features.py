import time
from data_preprocessing_unsw import import_and_clean
from evaluation import evaluate
from sklearn.ensemble import ExtraTreesClassifier

train = import_and_clean("UNSW-NB15_1.csv")
test = import_and_clean("UNSW-NB15_2.csv")

etc = ExtraTreesClassifier(max_depth = 2)

print("training...")
start = time.time()
etc.fit(train.iloc[:, :-2], train.iloc[:, -1])
end = time.time()

ttf = (end - start)

print("testing...")
start = time.time()
y_etc = etc.predict(test.iloc[:, :-2])
end = time.time()

ttp = (end - start)

print("evaluating...")
evaluate(y_etc, test.iloc[:, -1])

print("ttf = \t\t", ttf)
print("ttp = \t\t", ttp)
