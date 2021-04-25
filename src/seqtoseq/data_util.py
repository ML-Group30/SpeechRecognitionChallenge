import os
import random
import numpy as np
from matplotlib.image import imread

data_path = '../../data/'

def getDataSized(dataset, featurename=None, n=-1, start=0, rand=False, size=(128,87), flatten=False, s_mode ='spectrogram', transpose=False, combined=False):
    # Copy of getData, but specify resize dims and flatten boolean
    # Could replace getData in near future
    # combined gives all images in same array
    base_dir   = data_path + dataset + '/'
    img_dir   = base_dir  + 'images/'+ s_mode + '/'
    
    # Determine what features to load
    featurelist = []
    if (featurename != None): featurelist = [featurename]
    else:
        featurelist = os.listdir(img_dir)
        featurelist.remove('_background_noise_')

    # Get the relevant files for each feature
    master_filelist = []
    for feat in featurelist:
        filelist_full = [feat+'/'+f for f in os.listdir(img_dir+feat+'/')]
        if n == -1: filelist = filelist_full # Get all
        else:
            if rand:
                filelist = random.sample(filelist_full,n)
            else:
                filelist = filelist_full[start:start+n]
        master_filelist.append(filelist)

    output = []
    for filelist in master_filelist:
        img_list = []
        for filename in filelist:
            image = imread(img_dir + filename)
            image.resize(size)
            if transpose: image = image.T # I believe this puts data in [time][freqbands]
            if flatten: image = image.flatten()
            if combined: output.append(image)
            else: img_list.append(image)
        if not combined: output.append(img_list)
    return output

def getNoise(n, length=87, filename='random', flatten=True, s_mode='spectrogram', transpose=False):
    # Returns n clips of noise in the same dimensions of normal data
    # This might produce models more robust to noise than empty windows

    img_dir = data_path+'train/images/' + s_mode + '/_background_noise_/'

    noise_list = []

    if (filename == 'random'):
        noise_src = random.choices(os.listdir(img_dir), k=n)
    else:
        noise_src = n*[filename]

    for file in noise_src:
        image = imread(img_dir+file)
        # grab a random size[1] consecutive frames
        offset = random.randint(0,image.shape[1]-length)
        image = image[:, offset:(offset+length)]
        if transpose: image = image.T
        if flatten:
            image = image.flatten()
        noise_list.append(image)

    return noise_list
