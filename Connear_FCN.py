import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import backend as K
from connear_functions import *

class Connear_FCN:
    def __init__(self, input_size, filter_channel, kernal_sizes, lr):
        self.lr = lr
        self.filter_channel = filter_channel
        self.kernal_size = kernal_sizes
        self.input_size = input_size
        modeldir = 'connear/'
        self.cochlea, self.ihc, self.anf = build_connear(modeldir, poles='', Ncf=201)
        

    def build(self, pretrained_weights = None, trainable = True):
        layers_0 = Input(self.input_size)
        self.output = self.compute_output(layers_0, trainable)
        self.model = Model(inputs = layers_0, outputs = self.output)
        adam = Adam(lr=self.lr)
        self.model.compile(loss = 'mse', optimizer=adam)
        
    def compute_output(self, layers_0, trainable=True):
        filter_channel = self.filter_channel
        kernal_size = self.kernal_size

        output = Conv1D(filter_channel, kernal_size, padding = 'same', trainable=trainable, name='FCN_C1')(layers_0)
        output = BatchNormalization(trainable=trainable, name='FCN_BN1')(output)
        output = LeakyReLU()(output)

        output = Conv1D(filter_channel, kernal_size, padding = 'same', trainable=trainable, name='FCN_C2')(output)
        output = BatchNormalization(trainable=trainable, name='FCN_BN2')(output)
        output = LeakyReLU()(output)

        output = Conv1D(filter_channel, kernal_size, padding = 'same', trainable=trainable, name='FCN_C3')(output)
        output = BatchNormalization(trainable=trainable, name='FCN_BN3')(output)
        output = LeakyReLU()(output)

        output = Conv1D(filter_channel, kernal_size, padding = 'same', trainable=trainable, name='FCN_C4')(output)
        output = BatchNormalization(trainable=trainable, name='FCN_BN4')(output)
        output = LeakyReLU()(output)

        output = Conv1D(1, kernal_size, padding = 'same', trainable=trainable, name='FCN_C5')(output)

        return Activation('tanh')(output)



    
    # Cochlea stage
    vbm = cochlea.predict(stim) # BM vibration
    # IHC-ANF stage
    if Ncf == 201: # use the 201-CF models
        vihc = ihc.predict(vbm) # IHC receptor potential
        ranf = anf.predict(vihc) # ANF firing rate
    else: # use the 1-CF IHC-ANF models
        vihc = np.zeros(vbm.shape)
        ranf = np.zeros((3,vihc.shape[0],vihc.shape[1]-context_left-context_right,vihc.shape[2]))
        for cfi in range (0,CF.size): # simulate one CF at a time to avoid memory issues
            vihc[:,:,[cfi]] = ihc.predict(vbm[:,:,[cfi]])
            ranf[:,:,:,[cfi]] = anf.predict(vihc[:,:,[cfi]])