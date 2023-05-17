import numpy as np 
import pandas as pd
import os, time, random, sys, datetime, csv, pdb
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from scipy.io import wavfile
from tqdm import tqdm
import modelList
import Utils_C
from connear_functions import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"                        ## set GPU number 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth= True
config.gpu_options.per_process_gpu_memory_fraction = 0.9      ## set GPU memory fraction
sess = tf.compat.v1.InteractiveSession(config=config)

FLAGS = tf.compat.v1.flags.FLAGS
############----------Paramemters Setup----------############
tf.compat.v1.flags.DEFINE_string('MODEL', 'FCN', 'model name')
tf.compat.v1.flags.DEFINE_string('MODEL_Layer', '5', 'used to save result name')
tf.compat.v1.flags.DEFINE_integer('Epochs', 100, 'number of Epochs')
tf.compat.v1.flags.DEFINE_integer('FN_num', 30, 'number of filter channel')
tf.compat.v1.flags.DEFINE_integer('KS_num', 55, 'number of kernal size')
tf.compat.v1.flags.DEFINE_float('TRAIN_PERCENTAGE', 0.9, 'epochs for training iterations')
tf.compat.v1.flags.DEFINE_integer('batch_size', 1, 'number of batch size')
############----------Folder Setup----------############
tf.compat.v1.flags.DEFINE_string('train_clean_dir', '/mnt/Intern_SE/Data/clean_trainset_wav', 'set train clean data folder')
tf.compat.v1.flags.DEFINE_string('train_noisy_dir', '/mnt/Intern_SE/Data/noisy_trainset_wav', 'set train noisy data folder')
tf.compat.v1.flags.DEFINE_string('test_clean_dir', '/mnt/Intern_SE/Data/clean_testset_wav', 'set test clean data folder')
tf.compat.v1.flags.DEFINE_string('test_noisy_dir', '/mnt/Intern_SE/Data/noisy_testset_wav', 'set test noisy data folder')
tf.compat.v1.flags.DEFINE_string('RESULTS_DIR', './Result', 'dir for save prediction')
tf.compat.v1.flags.DEFINE_string('SOURCE_DIR', './Source', 'dir for save prediction')
tf.compat.v1.flags.DEFINE_string('MODEL_DIR', './Model', 'set folder for saver')  
tf.compat.v1.flags.DEFINE_string('DATE', 'lr_4_0506', 'date time')  

############----------Others Setup----------############
tf.compat.v1.flags.DEFINE_string('TRAIN', 'False', 'train this model or not')
tf.compat.v1.flags.DEFINE_bool('TEST', True, 'test this model or not')
tf.compat.v1.flags.DEFINE_string('LOAD_WEIGHTS',None, 'whether load multiple weights')
tf.compat.v1.flags.DEFINE_bool('LOAD_MULTIPLE_WEIGHTS', False, 'whether load multiple weights')
tf.compat.v1.flags.DEFINE_bool('ROBUST', False, 'train_with_noisy_file')
tf.compat.v1.flags.DEFINE_bool('SAVE_MODEL', True, 'loading model for retrain')
tf.compat.v1.flags.DEFINE_bool('SAVE_WEIGHTS', True, 'loading model for retrain')
tf.compat.v1.flags.DEFINE_string('SAVE_RESULTS', 'False', 'loading model for retrain')
tf.compat.v1.flags.DEFINE_bool('OVERWRITE_WEIGHT', False, 'overwrite loaded weight or not')
tf.compat.v1.flags.DEFINE_string('MONITOR', 'loss', 'loss or val_loss')
tf.compat.v1.flags.DEFINE_bool('Source', True, 'output the source')



############----------Functions Setup----------############
def get_weight_file(weight_folder):
    weight_files = [os.path.join(weight_folder, x) for x in os.listdir(weight_folder) if x.endswith(".h5")]
    if weight_files!=[]:
        print('Weights Found')
        return max(weight_files , key = os.path.getctime)
    print('Weights Not Found')
    return None

def main():
############----------Paramemters Setup----------############
    MODEL_NAME = '_'.join([FLAGS.MODEL, FLAGS.MODEL_Layer])
    filter_channel = int(FLAGS.FN_num)
    kernal_sizes = int(FLAGS.KS_num)
    Num_traindata_percentage = FLAGS.TRAIN_PERCENTAGE
    epoch = int(FLAGS.Epochs)
    batch_size = FLAGS.batch_size   
    KINDS = '_'.join([str(filter_channel),str(kernal_sizes),str(epoch),str(batch_size)])

############----------Output Folders----------############    
    RESULTS_DIR = FLAGS.RESULTS_DIR
    SOURCE_DIR = FLAGS.SOURCE_DIR
    MODEL_DIR = FLAGS.MODEL_DIR    
    DATE = FLAGS.DATE
    
    model_path = f'{MODEL_DIR}/{MODEL_NAME}_{KINDS}_{DATE}'
    result_path = f'{RESULTS_DIR}/{MODEL_NAME}_{KINDS}_{DATE}'
    score_path = f'{SOURCE_DIR}/{MODEL_NAME}_{KINDS}_{DATE}.csv'
    
############----------Others Input----------############
    # pdb.set_trace()
    SAVE_MODEL = FLAGS.SAVE_MODEL
    WEIGHTS_DIR = os.path.join(model_path,'weights')
    LOG_DIR = os.path.join(model_path,'logs')
    json_file = os.path.join(model_path, KINDS + '.json')
    now = datetime.datetime.now().replace(microsecond=0).isoformat()
    weights_file = os.path.join(WEIGHTS_DIR, KINDS + '.h5')# where weights will be saved


    ######################### Training_set ########################
    Train_Noisy_lists = Utils_C.get_filepaths(FLAGS.train_noisy_dir, True)
    Num_traindata = int(len(Train_Noisy_lists)*Num_traindata_percentage)
    Train_Clean_paths = FLAGS.train_clean_dir

    # data_shuffle
    permute = list(range(len(Train_Noisy_lists)))
    random.shuffle(permute)

    #----------------------- Train_set -----------------------
    Train_Noisy_lists = Utils_C.shuffle_list(Train_Noisy_lists,permute)
    Train_L_Noisy_lists = Train_Noisy_lists[0:Num_traindata]      # Only use subset of training data
    steps_per_epoch = (Num_traindata)//batch_size
    
    #----------------------- Val_set -----------------------
    Val_L_Noisy_lists = Train_Noisy_lists[Num_traindata:]
    Val_Clean_paths = FLAGS.train_clean_dir
    Num_valdata = len(Val_L_Noisy_lists)
    

    ######################### Test_set #########################
    Test_L_Noisy_lists  = Utils_C.get_filepaths(FLAGS.test_noisy_dir, True)
    Test_Clean_paths = FLAGS.test_clean_dir
    Num_testdata = len(Test_L_Noisy_lists)

    g1 = Utils_C.train_data_generator(Train_L_Noisy_lists, Train_Clean_paths,batch_size = batch_size)
    g2 = Utils_C.test_val_data_generator(Val_L_Noisy_lists, Val_Clean_paths)

    MODEL = modelList.get_model_with_name(FLAGS.MODEL)
    
    
    if FLAGS.TRAIN=='True':
        Utils_C.check_folder(WEIGHTS_DIR)
        Utils_C.check_folder(LOG_DIR)
        
        MODEL.build()
        MODEL.model.summary()
        if SAVE_MODEL:
            json_string = MODEL.model.to_json()
            with open(json_file, "w") as f:
                f.write(json_string)   
        
        print('Training...')

        tbCallBack = TensorBoard(log_dir=LOG_DIR)
        callbacks = [tbCallBack, EarlyStopping(monitor='val_loss', min_delta=0.00002, patience=80, verbose=0, mode='min')]

        if FLAGS.SAVE_WEIGHTS:
            if not os.path.exists(WEIGHTS_DIR): 
                os.mkdir(WEIGHTS_DIR)
            callbacks.append(ModelCheckpoint(weights_file, monitor=FLAGS.MONITOR, verbose=0, save_best_only=True, save_freq=2))
        print('Training...')
        hist = MODEL.model.fit(g1,	
                                steps_per_epoch = steps_per_epoch, 
                                epochs = epoch, 
                                verbose=1,
                                validation_data = g2,
                                validation_steps=Num_valdata,
                                callbacks=callbacks,
                                max_queue_size = 30,
                                use_multiprocessing=True,
                                workers=4
                                )
        
        print('Train set: MODEL',MODEL_NAME,'epoch=',epoch,'FN=',filter_channel,'KS=',kernal_sizes,'  is done.')


    if FLAGS.TEST:
############----------Make Sources Folder----------############    
        if FLAGS.Source:
            Utils_C.check_folder(score_path)
            if os.path.exists(score_path):
                os.remove(score_path)              
            with open(score_path, 'a') as f:
                f.write('Filename,Noisy_PESQ,Pred_PESQ,Noisy_STOI,Pred_STOI,Noisy_MSE,Pred_MSE,Noisy_SDI,Pred_SDI\n')

############----------Testing----------############ 
        json_file = open(json_file, 'r')
        json_string = json_file.read()
        json_file.close()
        # pdb.set_trace()
       
        if FLAGS.TRAIN=='True':
            MODEL = MODEL.model
        else:
            MODEL = model_from_json(json_string)
            MODEL.build(get_weight_file(WEIGHTS_DIR))
            print("Using weights:", get_weight_file(WEIGHTS_DIR))
        # else:
        #     print("NO weights info")
        #     pass

        print('De-noising...')            
        Predict_list = Test_L_Noisy_lists
        for path in tqdm(Predict_list): 
            noisy, clean = Utils_C.get_noisy_clean_pair(path, Test_Clean_paths)
            pred = MODEL.predict(noisy)
            prediction = pred.squeeze(2)    
            if FLAGS.Source:
                clean = clean/abs(clean).max()
                noisy = noisy/abs(noisy).max()
                pred_clean_wav = prediction/abs(prediction).max()
                # pdb.set_trace()
                n_pesq, n_stoi, n_mse, n_sdi = Utils_C.cal_score(clean,noisy)
                s_pesq, s_stoi, s_mse, s_sdi = Utils_C.cal_score(clean,pred_clean_wav)
        
                wave_name = path.split('/')[-1].split('.')[0]
                with open(score_path, 'a') as f:
                    f.write(f'{wave_name},{n_pesq},{s_pesq},{n_stoi},{s_stoi},{n_mse},{s_mse},{n_sdi},{s_sdi}\n')
                    
            if FLAGS.SAVE_RESULTS=='True':
                predict_path = os.path.join(result_path, path.split('/')[-1])
                Utils_C.check_folder(predict_path)
                wavfile.write(predict_path, 16000, np.int16(prediction/2*2**15))
                              
        if FLAGS.Source:
            data = pd.read_csv(score_path)
            n_pesq_mean = data['Noisy_PESQ'].to_numpy().astype('float').mean()
            s_pesq_mean = data['Pred_PESQ'].to_numpy().astype('float').mean()
            n_stoi_mean = data['Noisy_STOI'].to_numpy().astype('float').mean()
            s_stoi_mean = data['Pred_STOI'].to_numpy().astype('float').mean()
            n_mse_mean = data['Noisy_MSE'].to_numpy().astype('float').mean()
            s_mse_mean = data['Pred_MSE'].to_numpy().astype('float').mean()
            n_sdi_mean = data['Noisy_SDI'].to_numpy().astype('float').mean()
            s_sdi_mean = data['Pred_SDI'].to_numpy().astype('float').mean()

            with open(score_path, 'a') as f:
                f.write(','.join(('Average',str(n_pesq_mean),str(s_pesq_mean),str(n_stoi_mean),str(s_stoi_mean),str(n_mse_mean),str(s_mse_mean),str(n_sdi_mean),str(s_sdi_mean)))+'\n')                    
        print('Test set: MODEL',MODEL_NAME,'epoch=',epoch,'FN=',filter_channel,'KS=',kernal_sizes,'  is done.')
                                                     

if __name__ == '__main__':
    main()


