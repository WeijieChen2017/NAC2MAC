from PIL import Image

import numpy as np
import glob
import os

folderX = "./data_train/NPR_SRC/"
folderY = "./data_train/CT_SRC/"
valRatio = 0.2
testRatio = 0.1

# create directory and search nifty files
trainFolderX = "./data_train/" + folderX + "/train/"
trainFolderY = "./data_train/" + folderY + "/train/"
testFolderX = "./data_train/" + folderX + "/test/"
testFolderY = "./data_train/" + folderY + "/test/"
valFolderX = "./data_train/" + folderX + "/val/"
valFolderY = "./data_train/" + folderY + "/val/"

for folderName in [trainFolderX, testFolderX, valFolderX,
                   trainFolderY, testFolderY, valFolderY]:
    if not os.path.exists(folderName):
        os.makedirs(folderName)

fileList = glob.glob(folderX+"/*.nii") + glob.glob(folderX+"/*.nii.gz")
fileList.sort()
for filePath in fileList:
    print(filePath)

# shuffle and create train/val/test file list
np.random.seed("813")
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

# for valid_name in valid_list:
#     valid_nameX = folderX+"/"+valid_name
#     valid_nameY = folderY+"/"+valid_name.replace("NPR", "CT")
#     cmdX = "mv "+valid_nameX+" "+valid_folderX
#     cmdY = "mv "+valid_nameY+" "+valid_folderY
#     print(cmdX)
#     print(cmdY)
#     os.system(cmdX)
#     os.system(cmdY)

# for train_name in train_list:
#     train_nameX = folderX+"/"+train_name
#     train_nameY = folderY+"/"+train_name.replace("NPR", "CT")
#     cmdX = "mv "+train_nameX+" "+train_folderX
#     cmdY = "mv "+train_nameY+" "+train_folderY
#     print(cmdX)
#     print(cmdY)
#     os.system(cmdX)
#     os.system(cmdY)

# return [train_folderX, train_folderY, valid_folderX, valid_folderY]
