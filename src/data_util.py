import os
import spectrogram
import random
import numpy as np
import pandas as pd
from matplotlib.image import imread

data_path = '../data/'


def getData(dataset, featurename=None, n=-1, start=0, rand=False):
    # returns list of n spectrograms
    # list is used due to varying dimensions
    # if n=-1 all spectrograms in the directory are returned
    # start is offset from first file if not random
    base_dir = data_path + dataset + '/'
    img_dir = base_dir + 'images/'
    if (featurename != None): img_dir += (featurename + '/')

    filelist_full = os.listdir(img_dir)
    filelist = []
    if n == -1:
        filelist = filelist_full
    else:
        if rand:
            filelist = random.sample(filelist_full, n)
        else:
            filelist = filelist_full[start:start + n]

    spect_list = []

    for file in filelist:
        image = imread(img_dir + file)
        spect_list.append(image)

    return spect_list


def getDataSized(dataset, featurename=None, n=-1, start=0, rand=False,
                 size=(128, 87), flatten=False, names_list=False):
    # Copy of getData, but specify resize dims and flatten boolean
    # Could replace getData in near future
    base_dir = data_path + dataset + '/'
    # img_dir    = base_dir  + 'images/'
    img_dir = base_dir + '/'
    if (featurename != None): img_dir += (featurename + '/')

    filelist_full = os.listdir(img_dir)
    filelist = []
    if n == -1:
        filelist = filelist_full
    else:
        if rand:
            filelist = random.sample(filelist_full, n)
        else:
            filelist = filelist_full[start:start + n]

    spect_list = []
    filelist = [file for file in filelist if file[0] != '.']

    for file in filelist:
        image = imread(img_dir + file)
        image.resize(size)
        if flatten:
            image = image.flatten()
        spect_list.append(image)

    if names_list:
        return spect_list, filelist
    return spect_list


def getNoise(n, filename='random', flatten=True, dataset='train/images/_background_noise_/'):
    # Returns n clips of noise in the same dimensions of normal data
    # This might produce models more robust to noise than empty windows

    img_dir = data_path + dataset

    noise_list = []

    if (filename == 'random'):
        noise_src = random.choices(os.listdir(img_dir), k=n)
    else:
        noise_src = n * [filename]

    for file in noise_src:
        image = imread(img_dir + file)
        # grab a random 87 consecutive frames
        offset = random.randint(0, image.shape[1] - 87)
        image = image[:, offset:(offset + 87)]
        if flatten:
            image = image.flatten()
        noise_list.append(image)

    return noise_list


# Runs model on test data and outputs results to a csv that is Kaggle formatted.
# csv_path = path where csv is to be saved, test_path = path where test data is at,
# model = model trained on training data.
def test_model(csv_path, test_path, model):
    print("Beginning testing phase")

    # Get the images and the file names of the images. The file names will be used in the csv later.
    images, names = (getDataSized(test_path, '', -1, rand=True, flatten=True, names_list=True))

    # Format the files for the csv submission.
    names = [name.replace('png', 'wav') for name in names]

    # If only want to test out a small portion of data, uncomment these.
    # names = names[:100]
    # images = images[:100]

    # Use the given model to predict labels for all test images. This take approximately
    # 30 minutes on my machine.
    labels = model.predict(images)

    # Map 0-11 values to label name for csv.
    # 0-9 = target word, 10 = unknown, 11 = silence. Might be different for others.
    final_labels = []
    target_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

    for label in labels:
        if label == 10:
            final_labels.append('unknown')
        elif label == 11:
            final_labels.append('silence')
        else:
            final_labels.append(target_words[label])

    # Use the image file names and the label names to make csv.
    df = pd.DataFrame({'fname': names, 'label': final_labels})

    # If you want to print dataframe to console, uncomment this.
    # print(df)

    # Create a csv at the given file path.
    df.to_csv(csv_path, encoding='utf-8', index=False)


# See if this helps silence classification. I made silence.png, which is an all black image.
def get_silence():
    image_path = '../data/spectrograms/train/images/_background_noise_/silence.png'
    image = imread(image_path)
    offset = random.randint(0, image.shape[1] - 87)
    image = image[:, offset:(offset + 87)]
    image = image.flatten()
    return image


def dataSize(no_bg=True):
    # quick func to return max and average spectrogram size
    # background noise is large and distorts reporting
    #   So a boolean is set to disable it
    widths = []
    heights = []

    imgs = getData('test')
    for img in imgs:
        heights.append(img.shape[0])
        widths.append(img.shape[1])

    features = os.listdir('../data/train/images/')
    features.remove('_background_noise_')
    for f in features:
        imgs = getData('train', f)
        for img in imgs:
            heights.append(img.shape[0])
            widths.append(img.shape[1])
    print("MAX")
    print(max(heights))
    print(max(widths))
    print("AVG")
    print(np.mean(heights))
    print(np.mean(widths))


def spect_dir(dataset, featurename=None):
    # Apply the spectrogram function to all items in a folder
    # Dataset = 'train' or 'test'
    base_dir = data_path + dataset + '/'
    audio_dir = base_dir + 'audio/'
    img_dir = base_dir + 'images/'
    if (featurename != None): audio_dir += (featurename + '/')

    wavfiles = os.listdir(audio_dir)

    # Make images/ dir if none
    try:
        os.mkdir(img_dir)
    except:
        pass

    # Make feature specific dir in images/
    if featurename != None:
        try:
            os.mkdir(img_dir + (featurename + '/'))
        except:
            print("Spectrograms already calculated for " + featurename + "!")
            print("Delete the image directory for the feature if you want to regen")
            print("Skipping...")
            return
        img_dir += (featurename + '/')

    index = 1
    for wav_filename in wavfiles:
        print("Generating Spectrogram for " + audio_dir + wav_filename)
        print("File " + str(index) + " of " + str(len(wavfiles)))
        index += 1
        img_filename = wav_filename.split('.')[0] + '.png'
        spectrogram.wav_to_spec(audio_dir + wav_filename, img_dir + img_filename)


def spect_all():
    # Applies spect_dir to all of the data
    features = os.listdir(data_path + 'train/audio/')
    for feature in features:
        print("\nMaking Spectrograms for " + feature)
        print(feature)
        spect_dir('train', featurename=feature)
    spect_dir('test')
