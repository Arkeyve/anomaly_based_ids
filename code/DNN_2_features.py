from __future__ import absolute_import, division, print_function
from data_preprocessing_unsw import import_and_clean
import pandas as pd
import tensorflow as tf
import time

def train_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

train = import_and_clean("UNSW-NB15_1.csv")
test = import_and_clean("UNSW-NB15_2.csv")

input_features = []
for key in train.iloc[:, [9, 36]].keys():
    input_features.append(tf.feature_column.numeric_column(key = key))

classifier = tf.estimator.DNNClassifier(
        feature_columns = input_features,
        hidden_units = [10, 10],
        n_classes = 2
)

print("Training...")
start = time.time()
classifier.train(
        input_fn = lambda:train_input_fn(train.iloc[:, [9, 36]], train.iloc[:, -1], 100),
        steps = 1000
)
end = time.time()

ttf = (end - start)

print("Evaluating...")
start = time.time()
eval_result = classifier.evaluate(input_fn = lambda:eval_input_fn(test.iloc[:, [9, 36]], test.iloc[:, -1], 100))
end = time.time()

ttp = (end - start)

print("Test set accuracy: {accuracy:0.3f}\n".format(**eval_result))
print("TTF:", ttf)
print("TTP", ttp)

