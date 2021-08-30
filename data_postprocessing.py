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
predFolderY_ = "./data_pred/Y_"
predDataFile = "./y_hat.npy"
predLossFile = "./results.npy"

for folderName in [predFolderX, predFolderY, predFolderY_]:
    if not os.path.exists(folderName):
        os.makedirs(folderName)

cmdCopyX = "cp " + testFolderX + "* " + predFolderX
cmdCopyY = "cp " + testFolderY + "* " + predFolderY
predData = np.load(predDataFile)
predNorm = denormY(predData)
predLoss = np.load(predLossFile, allow_pickle=True)
print("Pred data shape: ", predData.shape)
print("Copy test X command: ", cmdCopyX)
print("Copy test Y command: ", cmdCopyY)
os.system(cmdCopyX)
os.system(cmdCopyY)

# restore pred data into nifty files
# test data generator is not shuffled, so restore them one by one

print("-"*50)
print("Restore nifty files.")
fileList = glob.glob(testFolderX+"/*.nii") + glob.glob(testFolderX+"/*.nii.gz")
fileList.sort()
for filePath in fileList:
    print("^"*30)
    print(filePath)
    niftyX = nib.load(filePath)
    dataX = niftyX.get_fdata()
    print("Data shape: ", dataX.shape)
    dataY_ = predNorm[:dataX.shape[2], :, :]
    print("Pred shape: ", dataY_,shape)
    predNorm = predNorm[dataX.shape[2], :, :]
    print("Left pred shape: ", predNorm.shape)
    niftyY_ = nib.Nifti1Image(dataY_, niftyX.affine, niftyX.header)
    savenameY_ = predFolderY_ + os.path.basename(filePath)
    nib.save(niftyY_, savenameY_)
print("-"*50)
print(predLoss)
print("Pred finished.")