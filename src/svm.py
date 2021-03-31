import pandas as pd
import os
from skimage.transform import resize
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import pickle
import data_util
import numpy as np
import matplotlib.pyplot as plt

# Meta Variables
iterations = 100
known_words= ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# Define Unknown words
unknown_words = os.listdir('../data/train/images/')
for k in known_words: unknown_words.remove(k)
unknown_words.remove('_background_noise_')

# build model
# NOTE testing more parameters takes much more time
# Line left in has best performance so far
#param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
#param_grid={'C':[10,100],'gamma':[0.0001,0.001],'kernel':['rbf']}
param_grid={'C':[100],'gamma':[0.001],'kernel':['rbf']}

svc   = svm.SVC(probability=True)
model = GridSearchCV(svc, param_grid, verbose=1, n_jobs=-1) # n_jobs -1 : Use all processors

try:
    #Data Section
    #############

    for i in range(iterations):
        x = [] ; y = []
        #n = 15
        n = 250

    # Silence is 0, give empty or noise arrays
    # Only one unique class 'silence' so no iteration
        sil = data_util.getNoise(n, 'random')
        x += sil
        y += [0]*n
    # Unknown class is 1, like a stack pointer
        for u in range(len(unknown_words)):
            imgs = (data_util.getDataSized('train', unknown_words[u], n, rand=True, flatten=True))
            x += imgs
            y += [1]*len(imgs)
    # enumerate classes grow from 2, away from defaults(silence and unknown)
    # This makes the code extensible for new words to be added
        for k in range(2,len(known_words)):
            imgs = (data_util.getDataSized('train', known_words[k-2], n, rand=True, flatten=True))
            x += imgs
            y += [k]*len(imgs)

    # Model section
    ###############

        print("\nSplitting new dataset...")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        print("Training on {0} graphs, testing {1}".format(len(x_train), len(x_test)))

        print(model.fit(x_train,y_train).best_params_)
        print("Fit complete")

        y_pred = model.predict(x_test)
        print(y_pred)
        print("vs real")
        print(np.array(y_test))

        print("Iteration {0} Accuracy: {1}\n".format(i, accuracy_score(y_pred,y_test)*100))

except:
# Save if cancelled halfway through
    print("Saving model")
    pickle.dump(model, open('svm_model.pickle', 'wb'))
    exit()

# In case of successful iteration, also save
print("Saving model")
pickle.dump(model, open('svm_model.pickle', 'wb'))
