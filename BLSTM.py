import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Add, BatchNormalization, LeakyReLU, LSTM, Bidirectional, Dense
from tensorflow.keras.optimizers import Adam

class BLSTM:
    def __init__(self, num_hiddens, out_dim, lr):
        self.num_hiddens = num_hiddens
        self.out_dim = out_dim
        self.lr = lr

    def build(self, trainable = True):
        
        layers_0 = Input(self.input_size)
        self.output = self.compute_output(layers_0, trainable)
        self.model = Model(inputs = layers_0, outputs = self.output)
        adam = Adam(lr=self.lr)
        self.model.compile(loss = 'mse', optimizer=adam)
        
        return self.model

    def compute_output(self, layers_0, trainable=True):
        num_hiddens = self.num_hiddens
        out_dim = self.out_dim
        
        blstm1 = Bidirectional(LSTM(num_hiddens))(layers_0)
        blstm2 = Bidirectional(LSTM(num_hiddens))(blstm1)
        output = Dense(out_dim)(blstm2)

        return output


