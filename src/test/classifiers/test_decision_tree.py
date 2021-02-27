import numpy as np

from src.models.supervised.classifier.decision_tree.decision_tree import DecisionTree


def simple_test():
    tree = DecisionTree(data=np.array([[1, 2, 3, 4],
                                       [1, 3, 5, 6],
                                       [1, 4, 7, 8],
                                       [10, 10, 10, 9]]),
                        labels=np.array([0, 0, 1, 1]))
    tree.train()
    print(tree.predict([[4, 6, 7, 9],
                        [1, 6, 7, 10]]))


def height_test():
    # Height
    data_ = np.array([[5.9, 5.0, 6.2, 5.9, 5.4, 5.3, 6.6, 5.5],
                      # Weight
                      [122.2, 105.2, 180.0, 122.2, 120.1, 110.0, 190.0, 150.1],
                      # Plays basketball?
                      [1, 0, 1, 1, 0, 0, 1, 0]])
    labels_ = np.array([1, 2, 1, 2, 1, 2, 2, 2])

    tree = DecisionTree(data=data_, labels=labels_)
    tree.train()
    class_dict = {1: "man", 2: "woman"}
    pred = tree.predict([[5.7, 160, 1]])
    for val in pred:
        assert (class_dict[val] == 'man')


def synth_test():
    import pandas as pd  # Local b/c only test
    dataset = '/Users/concord/Documents/Projects/MLAlgs/SyntheticDataset.csv'
    df = pd.read_csv(dataset)

    labels = df['Gender'].to_numpy()
    features = df[['Height(Inches)', 'Weight(Pounds)']].to_numpy()

    tree = DecisionTree(data=features, labels=labels)
    tree.train()
    pred = tree.predict([[65, 180]])
    print(pred)
    return tree