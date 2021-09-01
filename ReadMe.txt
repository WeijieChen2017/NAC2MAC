1: save the training dataset in "./data_train/", with the corresponding number
    example:

    ./NPR_SRC/NPR_098.nii.gz (NPR is NAC PET resampled with template of CT)
    ./CT_SRC/CT_098.nii.gz

2: run data_preprocessing.py, for save each slices of nifty files into separate files
    Save format can be tiff or npy
    Npy files can save and load faster and save 5 channels, so npy is selected here
    The test dataset order is saved.

3: Run train_im2im_keras_nac2ct.py
    model_name is the name of saved models
    modelTag is the name seen in tensorboard
    get_keras_npy_generator() is the new function supporting npy files and shuffle input
    tensorboardimage() is expired and not used for now

4: After training, run eval_im2im_keras_nac2ct.py
    It will predict all test npy files without shuffling. 
    The prediction is one by one, so take them as the order of test dataset in step 2

5: Restore predictions into nifty files
    for example:
    fileList = ['./data_train/NPR_SRC/NPR_011.nii.gz',
                './data_train/NPR_SRC/NPR_063.nii.gz',
                './data_train/NPR_SRC/NPR_143.nii.gz']
    So first 335 slices are for NPR_011.nii.gz
    the next 299 slices are for NPR_063.nii.gz
    the last 299 slices are for NPR_143.nii.gz
    The prections in nifty will be saved in ./data_pred