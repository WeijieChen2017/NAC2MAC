from PIL import Image
import nibabel as nib
import numpy as np
import glob
import os

def normX(data):
    data[data<0] = 0
    data[data>6000] = 6000
    data = data / 6000
    return data

def normY(data):
    data[data<-1000] = -1000
    data[data>3000] = 3000
    data = (data + 1000) / 4000
    return data

def denormY(data):
    data = data * 4000 - 1000
    return data

# load predictions and input NAC PET
testFolderX = "./data_train/X/test/"
testFolderY = "./data_train/Y/test/"
predFolderX = "./data_pred/X/"
predFolderY = "./data_pred/Y/"
predFolderY_ = "./data_pred/Y_/"
predDataFile = "./y_hat.npy"
predLossFile = "./predLoss.npy"

for folderName in [predFolderX, predFolderY, predFolderY_]:
    if not os.path.exists(folderName):
        os.makedirs(folderName)

fileList = ['./data_train/NPR_SRC/NPR_011.nii.gz',
            './data_train/NPR_SRC/NPR_063.nii.gz',
            './data_train/NPR_SRC/NPR_143.nii.gz']

predData = np.squeeze(np.load(predDataFile))
predNorm = denormY(predData)
predLoss = np.load(predLossFile, allow_pickle=True)
print("Pred data shape: ", predData.shape)

# restore pred data into nifty files
# test data generator is not shuffled, so restore them one by one

print("-"*50)
print("Restore nifty files.")
for fileXPath in fileList:
    print("^"*30)
    print(fileXPath)

    fileYPath = fileXPath.replace("NPR", "CT")
    os.system("cp " + fileXPath + " " + predFolderX)
    os.system("cp " + fileYPath + " " + predFolderY)

    niftyX = nib.load(fileXPath)
    dataX = niftyX.get_fdata()
    print("Data shape: ", dataX.shape)
    dataY_ = np.zeros(dataX.shape)
    for idx in range(dataX.shape[2]):
        dataY_[:, :, idx] = predNorm[idx, :, :]
    print("Pred shape: ", dataY_.shape)
    predNorm = predNorm[dataX.shape[2]:, :, :]
    print("Left pred shape: ", predNorm.shape)
    niftyY_ = nib.Nifti1Image(dataY_, niftyX.affine, niftyX.header)
    savenameY_ = predFolderY_ + "Y_" +os.path.basename(fileXPath).replace("NPR", "pCT")
    nib.save(niftyY_, savenameY_)
print("-"*50)
print(predLoss)
print("Pred finished.")