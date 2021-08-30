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

for folderName in [predFolderX, predFolderY, predFolderY_]:
    if not os.path.exists(folderName):
        os.makedirs(folderName)

cmdCopyX = "cp " + testFolderX + "* " + predFolderX
cmdCopyY = "cp " + testFolderY + "* " + predFolderY
predData = np.load(predDataFile)
print("Pred data shape: ", predData)
print("Copy test X command: ", cmdCopyX)
print("Copy test Y command: ", cmdCopyY)

fileList = glob.glob(testFolderX+"/*.nii") + glob.glob(testFolderX+"/*.nii.gz")
fileList.sort()
for filePath in fileList:
    print(filePath)

# shuffle and create train/val/test file list
# np.random.seed(813)
# fileList = np.asarray(fileList)
# np.random.shuffle(fileList)
# fileList = list(fileList)

# valList = fileList[:int(len(fileList)*valRatio)]
# valList.sort()
# testList = fileList[-int(len(fileList)*testRatio):]
# testList.sort()
# trainList = list(set(fileList) - set(valList) - set(testList))
# trainList.sort()

# print('-'*50)
# print("Training list: ", trainList)
# print('-'*50)
# print("Validation list: ", valList)
# print('-'*50)
# print("Testing list: ", testList)
# print('-'*50)

# # save each raw file as tiff image
# startZ = 0.45
# endZ = 0.75

# packageVal = [valList, valFolderX, valFolderY, "Validation"]
# packageTest = [testList, testFolderX, testFolderY, "Test"]
# packageTrain = [trainList, trainFolderX, trainFolderY, "Train"]

# for package in [packageTest, packageVal, packageTrain]:
#     fileList = package[0]
#     folderX = package[1]
#     folderY = package[2]
#     print("-"*25, package[3], "-"*25)

#     for pathX in fileList:
#         print(pathX)
#         pathY = pathX.replace("NPR", "CT")
#         filenameX = os.path.basename(pathX)[4:7]
#         filenameY = os.path.basename(pathY)[3:6]
#         dataX = nib.load(pathX).get_fdata()
#         dataY = nib.load(pathY).get_fdata()
#         lenZ = dataX.shape[2]
#         dataNormX = normX(dataX[:, :, int(lenZ*startZ):int(lenZ*endZ)])
#         dataNormY = normY(dataY[:, :, int(lenZ*startZ):int(lenZ*endZ)])
#         lenNormZ = dataNormX.shape[2]
#         for idx in range(lenNormZ):
#             sliceX = dataNormX[:, :, idx]
#             sliceY = dataNormY[:, :, idx]
#             savenameX = folderX + "X_" + filenameX + "_{0:03d}".format(idx) + ".tiff"
#             savenameY = folderY + "Y_" + filenameY + "_{0:03d}".format(idx) + ".tiff"
#             tiffX = Image.fromarray(sliceX)
#             tiffY = Image.fromarray(sliceY)
#             tiffX.save(savenameX)
#             tiffY.save(savenameY)
