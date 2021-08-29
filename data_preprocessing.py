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

for folderName in [trainFolderX, testFolderX, valFolderX,
                   trainFolderY, testFolderY, valFolderY]:
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
testList = fileList[:-int(len(fileList)*testRatio)]
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

print("-"*25, "Validation", "-"*25)
for valPathX in valList:
    print(valPathX)
    valPathY = valPathX.replace("NPR", "CT")
    filenameX = os.path.basename(valPathX)[4:7]
    filenameY = os.path.basename(valPathY)[3:6]
    dataX = nib.load(valPathX).get_fdata()
    dataY = nib.load(valPathY).get_fdata()
    lenZ = dataX.shape[2]
    dataNormX = normX(dataX[int(lenZ*startZ):int(lenZ*endZ)])
    dataNormY = normY(dataY[int(lenZ*startZ):int(lenZ*endZ)])
    lenNormZ = dataNormX.shape[2]
    for idx in range(lenNormZ):
        sliceX = dataNormX[:, :, idx]
        sliceY = dataNormY[:, :, idx]
        savenameX = valFolderX + "X_" + filenameX + "_{0:03d}".format(idx) + ".tiff"
        savenameY = valFolderY + "Y_" + filenameY + "_{0:03d}".format(idx) + ".tiff"
        tiffX = Image.fromarray(sliceX)
        tiffY = Image.fromarray(sliceY)
        tiffX.save(savenameX)
        tiffY.save(savenameY)
        # print(savenameX)
        # print(savenameY)


# for train_name in train_list:
#     train_nameX = folderX+"/"+train_name
#     train_nameY = folderY+"/"+train_name.replace("NPR", "CT")
#     cmdX = "mv "+train_nameX+" "+train_folderX
#     cmdY = "mv "+train_nameY+" "+train_folderY
#     print(cmdX)
#     print(cmdY)
#     os.system(cmdX)
#     os.system(cmdY)
