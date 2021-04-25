# Full Sequence to Sequence Trainer
# Uses the decoder in $DECODERFILENAME to encode spectrogram
# Spectrogram and teacher-forcing data to train

import data_util

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Knobs
###############
batch_size = 32
epochs = 150
latent_dim = 256
max_word_len = 16
alpha_len = 26 # Length of the alphabet used, not including spaces
s_mode = 'spectrogram'
img_size = (87,128) # TODO: fix data_util to output this
num_img = 1000 # Number of images per word to load


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
    output = np.zeros((1, max_word_len, alpha_len+3))
    for c in range(len(string)):
        output[0,c] = alpha_index(string[c])
    return output


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
        encoder_input_data += ([img_data[i]]*(len(word)+1))
        start_word = '[' + word
        word_end = word + ']'
        for t in range(0,len(start_word)):
            dec_in = np.zeros((max_word_len, alpha_len+3)) # Word progress
            #Teacher forcing: Each step has a character more than before
            for c in range(0,t+1): 
                dec_in[c] = alpha_index(start_word[c])
            decoder_output_data.append(alpha_index(word_end[t]))
            decoder_input_data.append(dec_in)
#Also silence, n*2 for representation '[', ' ', ']'
print("Processing data for silence")
sil_data = data_util.getNoise(num_img*2, length=img_size[0], flatten=False, s_mode=s_mode,
        transpose=True)
encoder_input_data += sil_data
for t in range(num_img): # This is ugly but was causing data organization issues
    dec_in = np.zeros((max_word_len, alpha_len+3))
    dec_in[0] = alpha_index('[')
    decoder_input_data.append(dec_in)
    dec_in = np.zeros((max_word_len, alpha_len+3))
    dec_in[0] = alpha_index('[') ; dec_in[1] = alpha_index(' ')
    decoder_input_data.append(dec_in)
    decoder_output_data.append(alpha_index(' ')) ; decoder_output_data.append(alpha_index(']'))
# Data cardinality issue if not np arrays
encoder_input_data = np.array(encoder_input_data)
decoder_input_data = np.array(decoder_input_data)
decoder_output_data = np.array(decoder_output_data)
print(encoder_input_data.shape)
print(decoder_input_data.shape)
print(decoder_output_data.shape)
 #Build model
#############
# Autoencoder
encoder_inputs = keras.Input(shape=(87,128))
encoder, state_h, state_c = layers.LSTM(latent_dim, return_sequences=True,
        return_state=True)(encoder_inputs)
encoder_states = [state_h, state_c] # This is transferred to decoder


decoder_inputs = keras.Input(shape=(max_word_len,alpha_len+3))
d = layers.LSTM(latent_dim, return_sequences=True,
        return_state=True)(decoder_inputs, initial_state=encoder_states)
d = layers.LSTM(latent_dim)(d)
decoder_outputs = layers.Dense(alpha_len+3)(d)

model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(opt, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

from keras.utils import plot_model
plot_model(model, to_file='testo.png',show_shapes=True, show_layer_names=True)

model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_output_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2
)
model.save("test_seq.model")
exit()
# TODO Fix this, replace with full inference function and test that too
test = string_to_encoding('[bir')
print(np.argmax(model.predict(test)))
print(chr(np.argmax(model.predict(test)) - 2 + ord('a')))
exit()
    
