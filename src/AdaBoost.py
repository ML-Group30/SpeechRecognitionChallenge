import os
import data_util
from sklearn.ensemble import AdaBoostClassifier
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

    x, y = [], []
    n = 15

    for word in words:
        # Get class label for current word. If is a target word, 0-9, otherwise 10.
        label = target_words.index(word) if word in target_words else unknown_idx

        # Get list of image data from current word.
        images = (data_util.getDataSized('mfcc/train', word, n, rand=True, flatten=True))

        # Create data and labels lists.
        x += images
        y += [label] * len(images)

    return x, y


def run_model(file):

    x, y = preprocess_data(file)

    # Split the data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    # Fit the model.
    model = AdaBoostClassifier()
    model.fit(x_train, y_train)

    # Check accuracy of model
    predictions = model.predict(x_test)
    accuracy = accuracy_score(predictions, y_test)
    return accuracy


# Specify data path.
spectrogram_data = '../data/spectrograms/train/images'
mfcc_data = '../data/mfcc/train/'

# Run AdaBoost on spectrogram and mfcc data and print results
print("spectrogram:", run_model(spectrogram_data))
print("mfcc:", run_model(mfcc_data))
