# mnist attention
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.activations import relu
 
TIME_STEPS = 1
INPUT_DIM = 72
LSTM_UNITS = 20

def my_relu(x,threshold=0):
    return relu(x,threshold=threshold)

# first way attention
def attention_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS,activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul
 
# build LSTM model with attention
def make_model(lstm_units,dropout_rate=0.3):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
    drop1 = Dropout(dropout_rate)(inputs)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
    attention_mul = attention_block(lstm_out)
    attention_flatten = Flatten()(attention_mul)
    drop2 = Dropout(dropout_rate)(attention_flatten)
    output = Dense(1, activation=my_relu)(drop2)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
