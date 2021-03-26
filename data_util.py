import os
import spectrogram

data_path = '../data/'

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
            print("Spectrograms already exist for {0}!".format(featurename))
            print("Delete the image dir for the feature if you want to regen")
            print("Skipping...")
            return
        img_dir += (featurename+'/')

    index = 1
    for wav_filename in wavfiles:
        print("Generating Spectrogram for " + audio_dir+wav_filename)
        print("File {0} of {1}".format(index,len(wavfiles)))
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
    spect_dir(test)

spect_all()
