import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

import deeprad_keras_tools

from PIL import Image
from time import time

from matplotlib import pyplot as plt
from train_im2im_keras_nac2ct import mu_loss, smooth_L1_loss

import tensorflow
from tensorflow.keras.callbacks import History, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow_addons.image import gaussian_filter2d
import numpy as np
import nibabel as nib
import Unet
import glob
import os

tensorflow.keras.backend.set_image_data_format('channels_last')

def execute():
    data_in_chan = 5
    data_out_chan = 1
    data_x = 512
    data_y = 512   
    model_x = 512
    model_y = 512
    batch_size = 1
    loss_group = [mu_loss, smooth_L1_loss,
                  losses.mean_squared_error, losses.mean_absolute_error]

    model_name = 'nac2ct'

    X_folder = "./data_train/X/"
    Y_folder = "./data_train/Y/"

    print('creating model')
    # model = Unet.UNetContinuous([model_x,model_y,data_in_chan],out_ch=data_out_chan,start_ch=64,depth=4,inc_rate=2.,activation='relu',dropout=0.5,batchnorm=True,maxpool=True,upconv=True,residual=False)
    model = Unet.UNetContinuous([model_x,model_y,data_in_chan],
                                out_ch=data_out_chan,
                                start_ch=128, depth=4, inc_rate=2,
                                activation='relu', dropout=0.5,
                                normtype="none", maxpool=True, # turn off batchnorm
                                upconv=True, residual=False)

    # model = deeprad_keras_tools.wrap_model( model, (data_x,data_y,1), (data_x,data_y,1), (model_x,model_y,1), (model_x,model_y,1) )    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=mu_loss, metrics=loss_group)
    model.summary()

    print('creating data generators')
    test_gen = deeprad_keras_tools.get_keras_npy_generator( os.path.join(X_folder,'test'),
                                                            os.path.join(Y_folder,'test'),
                                                            batch_size )
    
    print('load model weights')
    model.load_weights("./" + model_name + "_weights.h5")

    print('model eval')
    predLoss = model.evaluate(test_gen, batch_size=batch_size, verbose=1, return_dict=True)
    np.save("predLoss.npy", predLoss)

    print('model predict')
    y_hat = model.predict(test_gen, batch_size=batch_size, verbose=1, use_multiprocessing=True)
    print(y_hat.shape)
    np.save("y_hat.npy", y_hat)

if __name__ == "__main__": 
    execute()



