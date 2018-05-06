import time
import sys
from data_preprocessing_unsw import import_and_clean
from evaluation import evaluate
from sklearn.ensemble import RandomForestClassifier

train = import_and_clean("UNSW-NB15_1.csv")
test = import_and_clean("UNSW-NB15_2.csv")

rfc = RandomForestClassifier(max_depth = int(sys.argv[1]))

print("training...")
start = time.time()
rfc.fit(train.iloc[:, :-2], train.iloc[:, -1])
end = time.time()

ttf = (end - start)

print("testing...")
start = time.time()
y_rfc = rfc.predict(test.iloc[:, :-2])
end = time.time()

ttp = (end - start)

print("evaluating...")
evaluate(y_rfc, test.iloc[:, -1])

print("ttf = \t\t", ttf)
print("ttp = \t\t", ttp)
