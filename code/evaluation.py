def evaluate(classification_result, labels):
    comparison = list(zip(classification_result, labels))

    result = {
        (0, 0): 0,
        (0, 1): 0,
        (1, 0): 0,
        (1, 1): 0
    }

    for res in comparison:
        result[res] = result[res] + 1

    print("in", classification_result.shape[0], "records, the model predicts")
    print("True Negative:\t", result[(0, 0)])
    print("False Negative:\t", result[(0, 1)])
    print("False Positive:\t", result[(1, 0)])
    print("True Positive:\t", result[(1, 1)])
    accuracy = (result[(1, 1)] + result[(0, 0)]) / (result[(0, 0)] + result[(0, 1)] + result[(1, 0)] + result[(1, 1)])
    print("accuracy = \t", accuracy)
