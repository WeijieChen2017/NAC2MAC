import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

import deeprad_keras_tools

from PIL import Image
from time import time

from matplotlib import pyplot as plt

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

def smooth_L1_loss(y_true, y_pred):
    return losses.huber(y_true, y_pred)

def canny_loss(y_true, y_pred):
    edge_true = tensorflow.image.sobel_edges(y_true)
    edge_pred = tensorflow.image.sobel_edges(y_pred)
    return losses.MeanSquaredError(edge_true, edge_pred)

def mu_loss(y_true, y_pred, clip_delta=1.0):
    mu_huberL1 = 0.8
    mu_canny = 1-mu_mse

    edge_true = tensorflow.image.sobel_edges(y_true)
    edge_pred = tensorflow.image.sobel_edges(y_pred)
    edge_true_blur = gaussian_filter2d(edge_true)
    edge_pred_blur = gaussian_filter2d(edge_pred)
    canny = K.mean(K.square(edge_true_blur-edge_pred_blur))

    THRESHOLD = K.variable(1.0)
    mae = K.abs(y_true-y_pred)
    flag = K.greater(mae, THRESHOLD)
    huberL1 = K.mean(K.switch(flag, (mae - 0.5), K.pow(mae, 2)), axis=-1)
    
    return mu_huberL1 * huberL1 + mu_canny * canny

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
                                start_ch=64, depth=4, inc_rate=2,
                                activation='relu', dropout=0.5,
                                normtype="instance_norm", maxpool=True, # turn off batchnorm
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



