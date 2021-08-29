import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

import deeprad_keras_tools

from PIL import Image
from time import time

from matplotlib import pyplot as plt

import tensorflow
from tensorflow.keras.callbacks import History, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import numpy
import Unet

tensorflow.keras.backend.set_image_data_format('channels_last')

def smooth_L1_loss(y_true, y_pred):
    return losses.huber(y_true, y_pred)

def execute():
    data_in_chan = 1
    data_out_chan = 1
    data_x = 512
    data_y = 512*data_in_chan    
    model_x = 512
    model_y = 512
    batch_size = 10

    model_name = 'nac2ct'

    X_folder = "./data_train/X/"
    Y_folder = "./data_train/Y/"

    X_progress_file = "./data_train/X/val/X_127_050.tiff"
    Y_progress_file = "./data_train/Y/val/Y_127_050.tiff"

    print('creating model')
    model = Unet.UNetContinuous([model_x,model_y,data_in_chan],out_ch=data_out_chan,start_ch=16,depth=4,inc_rate=2.,activation='relu',dropout=0.5,batchnorm=True,maxpool=True,upconv=True,residual=False)
    model = deeprad_keras_tools.wrap_model( model, (data_x,data_y,1), (data_x,data_y,1), (model_x,model_y,1), (model_x,model_y,1) )    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=smooth_L1_loss, metrics=[smooth_L1_loss,losses.mean_squared_error,losses.mean_absolute_error])
    model.summary()

    print('creating data generators')
    test_gen = deeprad_keras_tools.get_keras_tiff_generator( os.path.join(X_folder,'test'), os.path.join(Y_folder,'test'), batch_size )
    
    print('load model weights')
    model.load_weights("./" + model_name + "_model.h5")

    print('model eval')
    results = model.evaluate(test_gen, batch_size=batch_size, verbose=1, return_dict=True)
    numpy.save("results.npy", results)

    print('model predict')
    y_hat = model.evaluate(test_gen, batch_size=batch_size, verbose=1, return_dict=True, use_multiprocessing=False)
    numpy.save("yhat.npy", yhat)
    # model.save(model_name + '_model.h5')

if __name__ == "__main__": 
    execute()



