import os
import data_util
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def preprocess_data(file):

    words = os.listdir(file)

    # Filter out categories that don't belong to a class.
    words.remove('_background_noise_')
    words = [f for f in words if not f.startswith('.')]

    # Specify the target words.
    target_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    unknown_idx = len(target_words)
    silence_idx = unknown_idx + 1

    print(words)

    x, y = [], []
    n = 250

    for word in words:
        # Get class label for current word. If is a target word, 0-9, otherwise 10.
        label = target_words.index(word) if word in target_words else unknown_idx

        sample_size = -1
        if label == 10:
            sample_size = n

        # Get list of image data from current word.
        images = (data_util.getDataSized('spectrograms/train/images/', word, sample_size, rand=True, flatten=True, names_list=False))

        # Create data and labels lists.
        x += images
        y += [label] * len(images)

    # Add silence in
    # silence = data_util.get_silence()
    # x += [silence] * n
    # y += [silence_idx] * n
    #
    # print(x[-n:])
    # print(y[-n:])

    return x, y


def run_model(file, model_name="RandomForest"):
    x, y = preprocess_data(file)

    # Split the data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    # Get model from model name and fit it.
    if model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "AdaBoost":
        model = AdaBoostClassifier()
    else:
        return

    model.fit(x_train, y_train)

    # Check accuracy of model
    predictions = model.predict(x_test)

    accuracy = accuracy_score(predictions, y_test)
    print("accuracy is", accuracy)

    csv_path = '../data/mfcc/out.csv'
    test_path = '../data/spectrograms/test'
    data_util.test_model(csv_path, test_path, model)

    return accuracy




# Specify data path.
spectrogram_data = '../data/spectrograms/train/images'
mfcc_data = '../data/mfcc/train/'
test_data = '../data/mfcc/out.csv'

# test_model(test_data, RandomForestClassifier())

# Run random forest on spectrogram and mfcc data and print results
print("spectrogram:", run_model(spectrogram_data, model_name="Random Forest"))
# print("mfcc:", run_model(mfcc_data, model_name="Random Forest"))

# print("spectrogram:", run_model(spectrogram_data, model_name="AdaBoost"))
# print("mfcc:", run_model(mfcc_data, model_name="AdaBoost"))
