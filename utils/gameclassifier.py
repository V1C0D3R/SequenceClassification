from enum import Enum
import numpy as np
import csv
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

class GameClassifier(object):
    """GameClassifier doc should be here"""
        
    # Classifiers
    def init_decision_tree_classifier(self):
        # Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression
        self.classifier = tree.DecisionTreeClassifier()
        print("Decision Tree classifier initialized")

    def init_random_forest_classifier(self, nb_estimators = 100):
        # A random forest is a meta estimator that fits a number of decision tree classifiers on various 
        # sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting
        self.classifier = RandomForestClassifier(n_estimators=nb_estimators)
        print("Random Forest classifier initialized")

    def learn(self, features, names):
        print("Learning...")
        # Fit regression model
        self.learn_rfc(features, names)

    def learn_dtc(self, features, names):
        # Fit regression model
        self.classifier = self.classifier.fit(features, names)

    def learn_rfc(self, features, names):
        # Fit regression model
        self.classifier = self.classifier.fit(features, np.ravel(names))

    def predict(self, test_features):
        print("Predicting...")
        return self.classifier.predict(test_features)

    def get_training_data(training_file_path, all_actions, skip_header):
        print("Getting training data...")
        nb_features = len(all_actions)+2
        features = np.ndarray((0, nb_features))
        classes = np.ndarray((0,1))

        training_file = open(training_file_path, 'r')

        num_lines = len(training_file.readlines())
        print("Number of lines : " + str(num_lines))
        training_file.seek(0)

        reader = csv.reader(training_file)
        if skip_header:
            next(reader)

        for row in reader:
            classes = np.append(classes, [[row[0].split(";")[0]]], axis=0)

            # Actions count
            feature_row_count = list()
            nb_of_actions = len(row) - 1
            for action in all_actions:
                count = row.count(action)
                if nb_of_actions > 0 :
                    feature_row_count.append(row.count(action))
                else:
                    feature_row_count.append(0)

            # Row position
            rowPosition = reader.line_num / (num_lines)

            # Race feature
            race = Race.race_from_string(row[0].split(";")[1])

            line_features = [feature_row_count + [rowPosition, race.value]]
            features = np.append(features, line_features, axis=0)

        training_file.close()
        return classes, features

    def get_testing_data(testing_file_path, all_actions, skip_header):
        print("Getting testing data...")
        nb_features = len(all_actions)+2
        test_features = np.ndarray((0, nb_features))

        testing_file = open(testing_file_path, 'r')

        num_lines = len(testing_file.readlines())
        print("Number of lines : " + str(num_lines))
        testing_file.seek(0)

        reader = csv.reader(testing_file)
        if skip_header:
            next(reader)

        row_names = []
        for row in reader:
            row_names.append(row[0].split(";")[0])

            # Actions count
            feature_row_count = list()
            for action in all_actions:
                feature_row_count.append(row.count(action))

            # Row position
            rowPosition = reader.line_num / (num_lines)

            # Race feature
            race = Race.race_from_string(row[0].split(";")[1])

            line_features = [feature_row_count + [rowPosition, race.value]]
            test_features = np.append(test_features, line_features, axis=0)

        testing_file.close()
        return row_names, test_features

class Race(Enum):
    Protoss = 1
    Terran = 2
    Zerg = 3
    Other = 4

    def __str__(self):
        return self.name

    @classmethod
    def race_from_string(self, race_name):
        return {
            'Protoss': Race.Protoss,
            'Terran': Race.Terran,
            'Zerg': Race.Zerg,
        }.get(race_name, Race.Other)



