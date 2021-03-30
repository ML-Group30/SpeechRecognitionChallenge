import os
import spectrogram
import random
import numpy as np
from matplotlib.image import imread

data_path = '../data/'

def getData(dataset, featurename=None, n=-1, start=0, rand=False):
    # returns list of n spectrograms
    # list is used due to varying dimensions
    # if n=-1 all spectrograms in the directory are returned
    # start is offset from first file if not random
    base_dir   = data_path + dataset + '/'
    img_dir    = base_dir  + 'images/'
    if (featurename != None): img_dir += (featurename+'/')

    filelist_full = os.listdir(img_dir)
    filelist = []
    if n == -1:
        filelist=filelist_full
    else:
        if rand:
            filelist = random.sample(filelist_full,n)
        else:
            filelist = filelist_full[start:start+n]

    spect_list = []
 
    for file in filelist:
        image = imread(img_dir+file)
        spect_list.append(image)

    return spect_list

def getDataSized(dataset, featurename=None, n=-1, start=0, rand=False,
        size=(128,87), flatten=False):
    # Copy of getData, but specify resize dims and flatten boolean
    # Could replace getData in near future
    base_dir   = data_path + dataset + '/'
    img_dir    = base_dir  + 'images/'
    if (featurename != None): img_dir += (featurename+'/')

    filelist_full = os.listdir(img_dir)
    filelist = []
    if n == -1:
        filelist=filelist_full
    else:
        if rand:
            filelist = random.sample(filelist_full,n)
        else:
            filelist = filelist_full[start:start+n]

    spect_list = []
 
    for file in filelist:
        image = imread(img_dir+file)
        image.resize(size)
        if flatten:
            image = image.flatten()
        spect_list.append(image)

    return spect_list

def getNoise(n, filename='random', flatten=True):
    # Returns n clips of noise in the same dimensions of normal data
    # This might produce models more robust to noise than empty windows

    img_dir = data_path+'train/images/_background_noise_/'

    noise_list = []

    if (filename == 'random'):
        noise_src = random.choices(os.listdir(img_dir), k=n)
    else:
        noise_src = n*[filename]

    for file in noise_src:
        image = imread(img_dir+file)
        # grab a random 87 consecutive frames
        offset = random.randint(0,image.shape[1]-87)
        print(offset)
        image = image[:, offset:(offset+87)]
        if flatten:
            image = image.flatten()
        noise_list.append(image)

    return noise_list

getNoise(1,'running_tap.png')

def dataSize(no_bg=True):
    # quick func to return max and average spectrogram size
    # background noise is large and distorts reporting
    #   So a boolean is set to disable it
    widths = []
    heights = []

    imgs = data_util.getData('test')
    for img in imgs:
        heights.append(img.shape[0])
        widths.append(img.shape[1])

    features = os.listdir('../data/train/images/')
    features.remove('_background_noise_')
    for f in features:
        imgs = data_util.getData('train', f)
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
    base_dir   = data_path + dataset + '/'
    audio_dir  = base_dir  + 'audio/'
    img_dir    = base_dir  + 'images/'
    if (featurename != None): audio_dir += (featurename+'/')

    wavfiles = os.listdir(audio_dir)

    # Make images/ dir if none
    try: os.mkdir(img_dir)
    except: pass

    # Make feature specific dir in images/
    if featurename != None:
        try: os.mkdir(img_dir+(featurename+'/'))
        except:
            print("Spectrograms already calculated for " + featurename + "!")
            print("Delete the image directory for the feature if you want to regen")
            print("Skipping...")
            return
        img_dir += (featurename+'/')

    index = 1
    for wav_filename in wavfiles:
        print("Generating Spectrogram for " + audio_dir+wav_filename)
        print("File " + str(index) + " of " + str(len(wavfiles)))
        index += 1
        img_filename = wav_filename.split('.')[0] + '.png'
        spectrogram.wav_to_spec(audio_dir+wav_filename, img_dir+img_filename)


def spect_all():
    # Applies spect_dir to all of the data
    features = os.listdir(data_path+'train/audio/')
    for feature in features:
        print("\nMaking Spectrograms for "+feature)
        print(feature)
        spect_dir('train', featurename=feature)
    spect_dir('test')

