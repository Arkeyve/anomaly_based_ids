from __future__ import absolute_import, division, print_function
from data_preprocessing_unsw import import_and_clean
import pandas as pd
import tensorflow as tf

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
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
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

train = import_and_clean("UNSW-NB15_1.csv")
test = import_and_clean("UNSW-NB15_2.csv")

input_features = []
for key in train.keys():
    input_features.append(tf.feature_column.numeric_column(key = key))

classifier = tf.estimator.DNNClassifier(
        feature_columns = input_features,
        hidden_units = [4, 4],
        n_classes = 2
)

print("Training...")
classifier.train(
        input_fn = lambda:train_input_fn(train, train.iloc[:, -1], 80),
        steps = 800
)

print("Evaluating...")
eval_result = classifier.evaluate(input_fn = lambda:eval_input_fn(test, test.iloc[:, -1], 100))

print("Test set accuracy: {accuracy:0.3f}\n".format(**eval_result))

