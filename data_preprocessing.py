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

folderX = "./data_train/NPR_SRC/"
folderY = "./data_train/CT_SRC/"
valRatio = 0.2
testRatio = 0.1

# create directory and search nifty files
trainFolderX = "./data_train/X/train/"
trainFolderY = "./data_train/Y/train/"
testFolderX = "./data_train/X/test/"
testFolderY = "./data_train/Y/test/"
valFolderX = "./data_train/X/val/"
valFolderY = "./data_train/Y/val/"

for folderName in [trainFolderX, testFolderX, folderX,
                   trainFolderY, testFolderY, folderY]:
    if not os.path.exists(folderName):
        os.makedirs(folderName)

fileList = glob.glob(folderX+"/*.nii") + glob.glob(folderX+"/*.nii.gz")
fileList.sort()
for filePath in fileList:
    print(filePath)

# shuffle and create train/val/test file list
np.random.seed(813)
fileList = np.asarray(fileList)
np.random.shuffle(fileList)
fileList = list(fileList)

valList = fileList[:int(len(fileList)*valRatio)]
valList.sort()
testList = fileList[-int(len(fileList)*testRatio):]
testList.sort()
trainList = list(set(fileList) - set(valList) - set(testList))
trainList.sort()

print('-'*50)
print("Training list: ", trainList)
print('-'*50)
print("Validation list: ", valList)
print('-'*50)
print("Testing list: ", testList)
print('-'*50)

# save each raw file as tiff image
startZ = 0.45
endZ = 0.75

packageVal = [valList, folderX, folderY, "Validation"]
packageTest = [testList, testFolderX, testFolderY, "Test"]
packageTrain = [trainList, trainFolderX, trainFolderY, "Train"]

for package in [packageTest, packageVal, packageTrain]:
    fileList = package[0]
    folderX = package[1]
    folderY = package[2]
    print("-"*25, package[3], "-"*25)

    for pathX in valList:
        print(pathX)
        pathY = pathX.replace("NPR", "CT")
        filenameX = os.path.basename(pathX)[4:7]
        filenameY = os.path.basename(pathY)[3:6]
        dataX = nib.load(pathX).get_fdata()
        dataY = nib.load(pathY).get_fdata()
        lenZ = dataX.shape[2]
        dataNormX = normX(dataX[int(lenZ*startZ):int(lenZ*endZ)])
        dataNormY = normY(dataY[int(lenZ*startZ):int(lenZ*endZ)])
        lenNormZ = dataNormX.shape[2]
        for idx in range(lenNormZ):
            sliceX = dataNormX[:, :, idx]
            sliceY = dataNormY[:, :, idx]
            savenameX = folderX + "X_" + filenameX + "_{0:03d}".format(idx) + ".tiff"
            savenameY = folderY + "Y_" + filenameY + "_{0:03d}".format(idx) + ".tiff"
            tiffX = Image.fromarray(sliceX)
            tiffY = Image.fromarray(sliceY)
            tiffX.save(savenameX)
            tiffY.save(savenameY)
