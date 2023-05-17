from FCN import FCN 
from connear_functions import *
import pdb

def get_model_with_name(MODEL_NAME):
    
    if MODEL_NAME == 'FCN':
        MODEL = FCN(input_size = (None,1), filter_channel=30, kernal_sizes=55, lr=5e-3)
    
    elif MODEL_NAME == 'Connear_FCN':    
        MODEL = Connear_FCN(num_hiddens=300, out_dim=257, lr=self.lr)
    
    elif MODEL_NAME == 'LSTM':
        MODEL = Models.LSTM(num_hiddens=300, out_dim=257, lr=self.lr)
        
    elif MODEL_NAME == 'BLSTM':
        MODEL = Models.BLSTM(num_hiddens=300, out_dim=257, lr=self.lr)    
        
               
    return MODEL
