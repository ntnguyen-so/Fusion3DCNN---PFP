import numpy as np
import os, fnmatch
import matplotlib.pyplot as plt
from keras import *
import sys
from utils.metrics import *
from utils.data_config import *

#######################
## Configure dataset ##
#######################
dataset_path = '../dataset/'
WD = {
    'input': {
        'test' : {
          'factors'    : dataset_path + '/input/',
          'predicted'  : dataset_path + '/output/'
        },
    'model_weights' : './pretrained_weights.h5'
    }
}

print('Loading testing data...')
testDataFiles = fnmatch.filter(os.listdir(WD['input']['test']['factors']), '201507*.npz')
testDataFiles.sort()
numSamples = len(testDataFiles)
print('Nunber of testing data = {0}'.format(numSamples))

###########################################
## Load data for training and evaluating ##
###########################################
def loadDataFile(path, areaId, mode):
    try:
        data = np.load(path)
        data = data['arr_0']
    except Exception:
        data = None
    if mode == 'X'  :
        mask = np.zeros(GLOBAL_SIZE_X)
    else:
        mask = np.zeros(GLOBAL_SIZE_Y)
    data = data[:, BOUNDARY_AREA[areaId][0]:BOUNDARY_AREA[areaId][1], BOUNDARY_AREA[areaId][2]:BOUNDARY_AREA[areaId][3], :]
    mask[:, PADDING[areaId][0]:PADDING[areaId][1], PADDING[areaId][2]:PADDING[areaId][3], :] = data

    return mask

def appendFactorData(factorName, factorData, X):
    # Load data
    data = factorData[:, :, :, FACTOR[factorName]]
    data = np.expand_dims(data, axis=3)
    data = np.expand_dims(data, axis=0)
    
    if factorName == 'Input_accident' or factorName == 'Input_sns':
        data[data > 0] = 1
    
    # Standardize data
    data = data.astype(float)
    data /= MAX_FACTOR[factorName]

    if X[factorName] is None:
        X[factorName] = data
    else:
        X[factorName] = np.vstack((X[factorName], data))

    return X

def loadTestData(dataFiles, fileId, areaId):
    # Initalize data
    X = {}
    for key in FACTOR.keys():
        X[key] = None
    
    y = {}
    y['default'] = None    

    seqName = dataFiles[fileId]
    
    factorData = loadDataFile(WD['input']['test']['factors'] + seqName, areaId, 'X')
    predictedData = loadDataFile(WD['input']['test']['predicted'] + seqName, areaId, 'Y')    

    # Load factors and predicted data
    for key in FACTOR.keys():
        X = appendFactorData(key, factorData, X)
    
    y = appendFactorData('default', predictedData, y)

    del X['default']
    return X, y

def logging(mode, contentLine):
    f = open(WD['loss'], mode)
    f.write(contentLine)
    f.close()

##########################
## Build learning model ##
##########################
def buildCNN(cnnInputs, imgShape, filters, kernelSize, factorName, isFusion=False, cnnOutputs=None):
    if isFusion == True:
        cnnInput = layers.add(cnnOutputs, name='Fusion_{0}'.format(factorName))
    else:
        cnnInput = layers.Input(shape=imgShape, name='Input_{0}'.format(factorName))

    for i in range(len(filters)):
        counter = i+1
        if i == 0:
            cnnOutput = cnnInput

        cnnOutput = layers.Conv3D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='tanh',
                                  name='Conv3D_{0}{1}'.format(factorName, counter))(cnnOutput)
        cnnOutput = layers.BatchNormalization(name='BN_{0}{1}'.format(factorName, counter))(cnnOutput)
    
    if cnnInputs is not None:
        cnnModel = Model(inputs=cnnInputs, outputs=cnnOutput)
    else:
        cnnModel = Model(inputs=cnnInput, outputs=cnnOutput)
    return cnnModel
    
def buildPrediction(orgInputs, filters, kernelSize, lastOutputs=None):
    predictionOutput = None
    for i in range(len(filters)):
        counter = i + 1
        if i == 0:
            if lastOutputs is not None:
                predictionOutput = lastOutputs
            else:
                predictionOutput = orgInputs
                    
        predictionOutput = layers.Conv3D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='sigmoid', 
                                         name='Conv3D_prediction{0}1'.format(counter))(predictionOutput)        
        predictionOutput = layers.Conv3D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='relu', 
                                         name='Conv3D_prediction{0}2'.format(counter))(predictionOutput)
        
    predictionOutput = Model(inputs=orgInputs, outputs=predictionOutput)
    return predictionOutput

def buildCompleteModel(imgShape, filtersDict, kernelSizeDict):
    # Define architecture for learning individual factors
    filters = filtersDict['factors']
    kernelSize= kernelSizeDict['factors']

    congestionCNNModel   = buildCNN(cnnInputs=None, imgShape=imgShape, filters=filters, kernelSize=kernelSize, factorName='congestion')
    rainfallCNNModel     = buildCNN(cnnInputs=None, imgShape=imgShape, filters=filters, kernelSize=kernelSize, factorName='rainfall')
    accidentCNNModel     = buildCNN(cnnInputs=None, imgShape=imgShape, filters=filters, kernelSize=kernelSize, factorName='accident')

    # Define architecture for fused layers
    filters = filtersDict['factors_fusion']
    kernelSize= kernelSizeDict['factors_fusion']

    fusedCNNModel       = buildCNN(cnnInputs=[congestionCNNModel.input, rainfallCNNModel.input, accidentCNNModel.input],
                                   cnnOutputs=[congestionCNNModel.output, rainfallCNNModel.output, accidentCNNModel.output],
                                   imgShape=imgShape,
                                   filters=filters, kernelSize=kernelSize,
                                   factorName='factors', isFusion=True
                                  )

    # Define architecture for prediction layer
    filters = filtersDict['prediction']
    kernelSize= kernelSizeDict['prediction']
    predictionModel     = buildPrediction(orgInputs=[congestionCNNModel.input, rainfallCNNModel.input, accidentCNNModel.input],
                                          filters=filters,
                                          kernelSize=kernelSize,
                                          lastOutputs=fusedCNNModel.output
                                         )        

    return predictionModel

###############################
## Define model architecture ##
###############################
imgShape = (6,60,80,1)
filtersDict = {}; filtersDict['factors'] = [128, 128, 256]; filtersDict['factors_fusion'] = [256, 256, 256, 128]; filtersDict['prediction'] = [64, 1]
kernelSizeDict = {}; kernelSizeDict['factors'] = (3,3,3); kernelSizeDict['factors_fusion'] = (3,3,3); kernelSizeDict['prediction'] = (3,3,3)

predictionModel = buildCompleteModel(imgShape, filtersDict, kernelSizeDict)
predictionModel.summary()
utils.plot_model(predictionModel,to_file='architecture.png',show_shapes=True)

#################################
## Load weights for prediction ##
#################################
predictionModel = buildCompleteModel(imgShape, filtersDict, kernelSizeDict)
predictionModel.load_weights(WD['input']['model_weights'])

################
## Evaluation ##
################
fResults = open('predict_export.csv', 'w')

start = 0 
numSamples = numSamples
for fileId in range(start, numSamples):
    print(fileId+1, numSamples)
    for areaId in range(len(BOUNDARY_AREA)):
        datetime = testDataFiles[fileId].split('.')[0]
        
        Xtest, ytest = loadTestData(testDataFiles, fileId, areaId)
        ypredicted = predictionModel.predict(Xtest)
        
        for step in range(GLOBAL_SIZE_Y[0]):        
          dataPredicted = ypredicted[0, step, :, :, 0]
          congestedLocations = np.argwhere(dataPredicted > 0)
          
          # calculate coordinates of congested locations
          for congestedLocation in congestedLocations:
              if areaId == 0:
                  relativeCongestedLocation = [congestedLocation[0] + 20, congestedLocation[1] + 50]
              elif areaId == 1:
                  relativeCongestedLocation = [congestedLocation[0] + 40, congestedLocation[1] + 100]
              else:
                  relativeCongestedLocation = [congestedLocation[0] + 20, congestedLocation[1] + 180]

              xy = '{:03d}{:03d}'.format(relativeCongestedLocation[0],relativeCongestedLocation[1])
              line = '{0},{1},{2},{3},{4},{5}\n'.format(\
                                            datetime,step,relativeCongestedLocation[0],relativeCongestedLocation[1],xy,\
                                            dataPredicted[congestedLocation[0], congestedLocation[1]]*MAX_FACTOR['Input_congestion'])
              fResults.write(line)
            
fResults.close()