import time
import graphviz
from data_preprocessing_unsw import import_and_clean
from evaluation import evaluate
from sklearn import tree

train = import_and_clean("UNSW-NB15_1.csv")
test = import_and_clean("UNSW-NB15_2.csv")

dtc = tree.DecisionTreeClassifier(max_depth = 2)

print("training...")
start = time.time()
dtc.fit(train.iloc[:, :-2], train.iloc[:, -1])
end = time.time()

ttf = (end - start)

print("testing...")
start = time.time()
y_dtc = dtc.predict(test.iloc[:, :-2])
end = time.time()

ttp = (end - start)

print("evaluating...")
evaluate(y_dtc, test.iloc[:, -1])

print("ttf = \t\t", ttf)
print("ttp = \t\t", ttp)

dot_data = tree.export_graphviz(dtc, out_file=None, feature_names=train.columns[:-2], class_names=['normal', 'attack'], filled=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("./results/dtc_clf_md2_all_features")
