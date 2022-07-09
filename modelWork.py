import json
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

def saveModel(model, name):
    convertedUF = []
    for x in list(model.user_factors):
        tmp = []
        for y in list(x):
            tmp.append(float(y))
        convertedUF.append(tmp)
    convertedIF = []
    for x in list(model.item_factors):
        tmp = []
        for y in list(x):
            tmp.append(float(y))
        convertedIF.append(tmp)
    mapOfModel = {'user_factors':convertedUF,'item_factors':convertedIF,'factors':model.factors,
    'regularization':model.regularization,'iterations':model.iterations,'num_threads':model.num_threads,
    'shapeU0':model.user_factors.shape[0],'shapeU1':model.user_factors.shape[1],
    'shapeI0':model.item_factors.shape[0],'shapeI1':model.item_factors.shape[1],}
    fp = open(name + '.txt', 'w')
    fp.write(json.dumps(mapOfModel))
    fp.close()

def loadModel(name):
    fp = open(name + '.txt', 'r')
    mapOfModel = json.load(fp)
    model = AlternatingLeastSquares(factors = mapOfModel['factors'], regularization = mapOfModel['regularization'], iterations = mapOfModel['iterations'], num_threads = mapOfModel['num_threads'])
    
    tmp = np.ndarray(shape = (mapOfModel['shapeU0'],mapOfModel['shapeU1']), dtype = np.float32)
    for x in range(len(mapOfModel['user_factors'])):
        tmp1 = np.ndarray( shape = (mapOfModel['shapeU1'],), dtype = np.float32 )
        for y in range(len(mapOfModel['user_factors'][x])):
            tmp1[y] = (np.float32(mapOfModel['user_factors'][x][y]))
        tmp[x] = tmp1
    model.user_factors = tmp

    tmp = np.ndarray(shape = (mapOfModel['shapeI0'],mapOfModel['shapeI1']), dtype = np.float32)
    for x in range(len(mapOfModel['item_factors'])):
        tmp1 = np.ndarray( shape = (mapOfModel['shapeI1'],), dtype = np.float32 )
        for y in range(len(mapOfModel['item_factors'][x])):
            tmp1[y] = (np.float32(mapOfModel['item_factors'][x][y]))
        tmp[x] = tmp1
    model.item_factors = tmp

    fp.close()
    return model