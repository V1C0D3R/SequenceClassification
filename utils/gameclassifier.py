import numpy as np
import csv
from sklearn import tree

class GameClassifier(object):
    """GameClassifier doc should be here"""

    def __init__(self):
        # Init default classifier
        self.init_decision_tree_classifier()
        
    # Classifiers
    def init_decision_tree_classifier(self):
        # Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression
        self.classifier = tree.DecisionTreeClassifier()

    def learn(self, features, names):
        # Fit regression model
        self.classifier = self.classifier.fit(features, names)

    def predict(self, test_features):
        return self.classifier.predict(test_features)

    def get_training_data(training_file_path, all_actions, skip_header):
        features = np.ndarray((0, len(all_actions)))
        classes = np.ndarray((0,1))

        training_file = open(training_file_path, 'r')
        reader = csv.reader(training_file)

        # if skip_header:
        next(reader)

        for row in reader:
            classes = np.append(classes, [[row[0].split(";")[0]]], axis=0)

            feature_row_count = list()
            nb_of_actions = len(row) - 1
            for action in all_actions:
                count = row.count(action)
                if nb_of_actions > 0 :
                    feature_row_count.append(row.count(action))
                else:
                    feature_row_count.append(0)
            features = np.append(features, [feature_row_count], axis=0)

        training_file.close()
        return classes, features

    def get_testing_data(testing_file_path, all_actions, skip_header):
        test_features = np.ndarray((0, len(all_actions)))

        testing_file = open(testing_file_path, 'r')
        reader = csv.reader(testing_file)

        # if skip_header:
        next(reader)

        row_names = []
        for row in reader:
            row_names.append(row[0].split(";")[0])
            feature_row_count = list()
            for action in all_actions:
                feature_row_count.append(row.count(action))
            test_features = np.append(test_features, [feature_row_count], axis=0)

        testing_file.close()
        return row_names, test_features


