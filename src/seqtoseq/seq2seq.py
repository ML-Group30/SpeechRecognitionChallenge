# Testing batch size = one word input to reset states after
import data_util

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Knobs
###############
epochs = 150
latent_dim = 256
max_word_len = 8
s_mode = 'spectrogram'
num_img = 100 # Number of images per word to load

# Data specifications
alpha_len = 26 # Length of the alphabet used, not including spaces
img_size = (87,128) # TODO: fix data_util to output this


# One-hot formatting functions
##############################
def alpha_index(c):
# Returns a full alphabet-one-hot np array for a character
    # alphabet w/ 0,1 reserved for [,] & 2 for silence
    output = np.zeros([alpha_len + 3]) 
    if c == '[': output[0] = 1 ; return output # Start character
    if c == ']': output[1] = 1 ; return output # End character
    if c == ' ': output[2] = 1 ; return output # End character
    output[ord(c)-ord('a') + 3] = 1 # Normal letter
    return output

def index_to_alpha(c):
# Turns one-hot encoded char back into character
# For interpreting results
# This will likely be one_hot[word_index, time_step, char_index]
    char_id = np.argmax(c) 
    if char_id == 0 : return '['
    if char_id == 1 : return ']'
    if char_id == 2 : return ' '
    return chr(char_id + ord('a') - 3)

def string_to_encoding(string):
    # Converts a string into an acceptable model input
    output = np.zeros((max_word_len, alpha_len+3))
    for c in range(len(string)):
        output[c] = alpha_index(string[c])
    return output

def word_to_decoder_input(word):
    dec_out = word + ']' + ' '*(max_word_len - len(word))
    dec_in = '[' + word + ']'
    dec_in += ' '*(max_word_len - len(word))
    word = list(word)
    inputs = []
    outputs = []
    for t in range(1,max_word_len+1):
        inputs.append(string_to_encoding(dec_in[:t]))
        outputs.append(string_to_encoding(dec_out[t-1])[0])
    return inputs, outputs

# Prepare data
word_list = os.listdir('../../data/train/images/' + s_mode + '/')
word_list.remove('_background_noise_')
print(word_list)

encoder_input_data = []
decoder_input_data = []
decoder_output_data = []

for word in word_list:
    print(f"Processing data for word {word}")
    img_data = data_util.getDataSized('train', featurename=word, n=num_img,
            s_mode=s_mode, transpose=True)[0]
    for i in range(len(img_data)):
        #Pair each image up with an entire sequence of teacher forcing 
        encoder_input_data += ([img_data[i]]*max_word_len)
        dec_in, dec_out = word_to_decoder_input(word)
        decoder_input_data += dec_in
        decoder_output_data += dec_out

#Also silence, n*2 for representation '[', ' ', ']'
print("Processing data for silence")
sil_data = data_util.getNoise(num_img*max_word_len, length=img_size[0], flatten=False, s_mode=s_mode,
        transpose=True)
encoder_input_data += sil_data
for t in range(num_img):
    dec_in, dec_out = word_to_decoder_input(' '*(max_word_len - 2))
    decoder_input_data += dec_in
    decoder_output_data += dec_out

# Data cardinality issue if not np arrays
encoder_input_data = np.array(encoder_input_data)
decoder_input_data = np.array(decoder_input_data)
decoder_output_data = np.array(decoder_output_data)
print(encoder_input_data.shape)
print(decoder_input_data.shape)
print(decoder_output_data.shape)


# Build model
#############
# Autoencoder structure stolen from Robert's LSTM model
encoder_inputs = keras.Input(shape=(87,128))
e = layers.Conv1D(filters=128, kernel_size=7, strides=2)(encoder_inputs)
e = layers.BatchNormalization()(e)
e = layers.Conv1D(filters=128, kernel_size=7, strides=2)(e)
e = layers.BatchNormalization()(e)
e = layers.Activation('relu')(e)
lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(e)
e = layers.Activation('relu')(e)
lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(lstm)
e = layers.Activation('relu')(e)
encoder, fstate_h, fstate_c, bstate_h, bstate_c = layers.Bidirectional(layers.LSTM(latent_dim,
    return_sequences=True, return_state=True))(lstm)
encoder_states = [fstate_h, fstate_c, bstate_h, bstate_c]

#train decoder
decoder_inputs = keras.Input(shape=(max_word_len,alpha_len+3))

'''
# Attempt at creating an attention Layer, TODO: Replace with real attn
tensors  = layers.Bidirectional(layers.LSTM(latent_dim,
        return_sequences=True, return_state=True, dropout=0.1,
        stateful=False))(decoder_inputs, initial_state=encoder_states)
d = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences = False))(tensors)
#d = layers.Concatenate(axis=0)(attn_layers) # Reduce 5 tensors into 1 per step
attn = layers.Dense(max_word_len*(alpha_len+3), activation='sigmoid')(d)
d = layers.Reshape((max_word_len,(alpha_len+3)))(attn)
d = layers.Bidirectional(layers.LSTM(max_word_len, activation='relu', dropout=0.3))(d)
d = layers.Activation('sigmoid')(d)
decoder_outputs = layers.Dense((alpha_len+3),activation="softmax")(d)
'''
# Encoder output is max_word_len not latent_dim
tensors  = layers.Bidirectional(layers.LSTM(latent_dim,
        return_sequences=True, return_state=True, dropout=0.3,
        stateful=False))(decoder_inputs, initial_state=encoder_states)
d = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences = False))(tensors)
# Fully connected layer
d = layers.Dense(max_word_len*(alpha_len+3), activation="relu")(d)
d = layers.Dense(max_word_len*(alpha_len+3), activation="relu")(d)
d = layers.Dropout(0.3)(d)
decoder_outputs = layers.Dense((alpha_len+3),activation="softmax")(d)

model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
opt = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=opt, loss="mse", metrics=["accuracy"])
model.summary()

from keras.utils import plot_model
plot_model(model, to_file='testo.png',show_shapes=True, show_layer_names=True)

print("Training the seq2seq model")
history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_output_data,
    batch_size=max_word_len,
    epochs=epochs,
    shuffle=False,
    validation_split=0.2
)
model.save("test_seq.model")
#plt.plot(range(epochs), history.history['val_loss'])
plt.show()

def make_inference(image):
# Loop predictions in the model to output the guessed word
    text = '[' # start prediction
    output = ''
    image = np.array([image])
    print(image.shape)
    print(np.array([string_to_encoding(text)]).shape)
    for t in range(max_word_len):
        pred = model.predict([image, np.array([string_to_encoding(text)])])
        output = chr(np.argmax(pred) - 3 + ord('a'))
        text += output
        if output == ']': break
    return text

#TODO May not be reinserting internal state, look into manual reinsertion
for word in word_list:
    #Test with unseen image for each word
    img = data_util.getDataSized('train', featurename=word, n=1, start=num_img+1,
            s_mode=s_mode, transpose=True)[0]
    print(f" Predicting on word {word}, result:{make_inference(img[0])}")
exit()
    
