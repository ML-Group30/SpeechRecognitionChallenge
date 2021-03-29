import os
#import matplotlib.pyplot as plt
import numpy as np
import librosa as lr
#import librosa.display as lrd
from PIL import Image as im


#audio_path = "./wow/"
#image_path = "./img/"

#audio = os.listdir(audio_path)
#filename = audio_path+audio[4]


def wav_to_spec(wavfile, imgfile, sr=44100):
    # Convert signal to spectrogram
    x, sr = lr.load(wavfile, sr=44100)
    X = lr.stft(x, n_fft=512)
    Xmag = abs(X)**2.0
    Xmel = lr.feature.melspectrogram(y=x, sr=sr, n_mels=128, fmin=50, fmax=7000)

    # Convert to image
    # Normalize then expand to 255 for pixel intensity
    scale = 1.0/np.amax(Xmel)
    Xmel *= scale
    spec = np.rot90(np.array(Xmel**0.125).T * 255)
    imRep = im.fromarray(spec)
    imRep = imRep.convert('L')
    imRep.save(imgfile)
