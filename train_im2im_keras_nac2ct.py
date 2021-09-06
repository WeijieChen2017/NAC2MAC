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
from skimage import feature
import numpy
import Unet

tensorflow.keras.backend.set_image_data_format('channels_last')

def smooth_L1_loss(y_true, y_pred):
    return losses.huber(y_true, y_pred)

def canny_loss(y_true, y_pred):
    edge_true = feature.canny(tensorflow.make_ndarray(y_true), sigma=1)
    edge_pred = feature.canny(tensorflow.make_ndarray(y_pred), sigma=1)
    return losses.MeanSquaredError(edge_true, edge_pred)

def mu_loss(y_true, y_pred):
    mu_mse = 0.7
    mu_canny = 1-mu_mse
    edge_true = tensorflow.image.sobel_edges(y_true)
    edge_pred = tensorflow.image.sobel_edges(y_pred)
    mse = K.mean(K.square(y_true-y_pred))
    canny = K.mean(K.square(edge_true-edge_pred))
    loss = mu_mse * mse + mu_canny * canny
    return loss

def execute():

    model_name = 'nac2ct'
    modelTag = "nac2ct_4-64_5-1_xBN_mu7"
    continue_train = False
    initial_epoch = 0 # 0-9 at first, start from 10
    loss_group = [mu_loss, smooth_L1_loss,
                  losses.mean_squared_error, losses.mean_absolute_error]

    data_in_chan = 5
    data_out_chan = 1
    data_x = 512
    data_y = 512   
    model_x = 512
    model_y = 512
    batch_size = 8
    num_epochs = 20 + initial_epoch

    X_folder = "./data_train/X/"
    Y_folder = "./data_train/Y/"

    # X_progress_file = "./data_train/X/val/X_127_050.tiff"
    # Y_progress_file = "./data_train/Y/val/Y_127_050.tiff"

    print('creating model')
    model = Unet.UNetContinuous([model_x,model_y,data_in_chan],
                                out_ch=data_out_chan,
                                start_ch=64, depth=4, inc_rate=2,
                                activation='relu', dropout=0.5,
                                batchnorm=False, maxpool=True, # turn off batchnorm
                                upconv=True, residual=False)
    # model = deeprad_keras_tools.wrap_model( model, (data_x,data_y,1), (data_x,data_y,1), (model_x,model_y,1), (model_x,model_y,1) )    
    # data_input_shape: The shape of the input data (from DeepRad). This is always a length two tuple [e.g., (M,N*C)]
    # data_output_shape: The shape of the ground truth and output data (from DeepRad). This is always a length two tuple [e.g., (M,N*C)]
    # model_input_shape: The shape of the Keras model input data. This is always a length three tuple [e.g., (M,N,C)]
    # model_output_shape: The shape of the Keras model ground truth and output data. This is always a length three tuple [e.g., (M,N,C)]
    # model = deeprad_keras_tools.wrap_model( model, 
    #                                       (512,512*5),
    #                                       (512,512),
    #                                       (512,512,5),
    #                                       (512,512,1))    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=mu_loss, metrics=loss_group)
    model.summary()

    if continue_train:
        initial_epoch_fit = initial_epoch
        print('load model weights')
        model.load_weights("./" + model_name + "_model.h5")
    else:
        initial_epoch_fit = 0

    print('creating data generators')
    train_gen = deeprad_keras_tools.get_keras_npy_generator( os.path.join(X_folder,'train'), os.path.join(Y_folder,'train'),
                                                             batch_size, shuffle=True )
    val_gen = deeprad_keras_tools.get_keras_npy_generator( os.path.join(X_folder,'val'), os.path.join(Y_folder,'val'),
                                                           batch_size, shuffle=True )

    print('creating callbacks')
    history = History()
    modelCheckpoint = ModelCheckpoint(model_name + '_weights.h5', monitor='loss', save_best_only=True)
    tblogdir = os.path.join('tblogs','{}'.format(time()), '{}'.format(modelTag))
    tensorboard = TensorBoard(log_dir=tblogdir)
    # X_progress = deeprad_keras_tools.read_images( [X_progress_file] )
    # Y_progress = deeprad_keras_tools.read_images( [Y_progress_file] )
    # tensorboardimage = deeprad_keras_tools.TensorBoardIm2ImCallback(log_dir=tblogdir,X=X_progress,Y=Y_progress)

    print('fitting model')
    model.fit(  train_gen,
                validation_data=val_gen,
                epochs=num_epochs,
                use_multiprocessing=True,
                max_queue_size=20,
                initial_epoch=initial_epoch_fit,
                workers=4,
                callbacks=[history, modelCheckpoint] )#, tensorboardimage, tensorboard

    model.save(model_name + '_model.h5')

if __name__ == "__main__": 
    execute()



