import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from data_preprocessing_unsw import import_and_clean

ds = import_and_clean("UNSW-NB15_3.csv")

model = ExtraTreesClassifier()

model.fit(ds.iloc[:, :-2], ds.iloc[:, -1])

plt.bar(np.arange(47), model.feature_importances_)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
