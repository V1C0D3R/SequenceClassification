import datetime as dt
from utils.gameclassifier import GameClassifier

# Wanted output file
wanted_file_path = "./outputs/results_{}.csv"
output_file_path = wanted_file_path.format( dt.datetime.now().strftime('%Y%m%d%H%M%S') )
output_header = "row ID,battleneturl\n"

# Starcraft game's training and test file paths
training_file_path = "../train.csv"
testing_file_path = "../test.csv"

# Generate Starcraft actions names
hotkeys_0 = ["hotkey"+str(i)+"0" for i in range(10)]
hotkeys_1 = ["hotkey"+str(i)+"1" for i in range(10)]
hotkeys_2 = ["hotkey"+str(i)+"2" for i in range(10)]
all_actions = ['s', 'sBase', 'sMineral'] + hotkeys_0 + hotkeys_1 + hotkeys_2

# Classify
gc = GameClassifier()

players, features = GameClassifier.get_training_data(training_file_path, all_actions, True)

# Choose classifier and learn
gc.init_random_forest_classifier(len(all_actions))
gc.learn(features, players)

row_names, test_features = GameClassifier.get_testing_data(testing_file_path, all_actions, True)
predictions = gc.predict(test_features)

# Write predictions into file
output_file = open(output_file_path, 'w')
output_file.write(output_header)
for i, name in enumerate(row_names):
    output_file.write(name + "," + predictions[i] + "\n")

output_file.close()
