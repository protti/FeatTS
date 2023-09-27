import csv
import pickle

import numpy as np
from scipy.spatial.distance import euclidean



def getListOfValue(path):
    listComplete = []
    with open(path, 'r') as csvFile:
        dictTS = list()
        reader = csv.reader(csvFile)
        id = 0
        for row in reader:
            listApp = []
            splitted = row[0].split('\t')
            for i in range(1,len(splitted)):
                listApp.append(float(splitted[i]))

            dictSingl = {"class": splitted[0], "list": listApp,"id":id}
            id+=1

            # print(dictSingl)
            listComplete.append(dictSingl)
    return listComplete




def saveObject(filename,object):
    outfile = open(filename, 'wb')
    pickle.dump(object, outfile)
    outfile.close()


def getObject(filename):
    infile = open(filename, 'rb')
    dictOfDist = pickle.load(infile)
    return dictOfDist


def fillDict(dictOfValue):
    listaId = list(dictOfValue.keys())
    for x in range(0, len(listaId)):
        dictApp = dictOfValue[listaId[x]]
        for idDistance in dictApp:
            dictSingl = {"id": listaId[x], "distance": idDistance['distance']}
            if not dictSingl in dictOfValue[idDistance['id']]:
                dictOfValue[idDistance['id']].append(dictSingl)
    return dictOfValue



def getDistOt(dictTotal,idS):
    for value in dictTotal:
        if value['id'] == idS:
            return value['distance']



def getDistU(dictTotal,idS):
    for value in dictTotal:
        if value['id'] == idS:
            return value['u']

def max_num_in_list( list ):
    max = list[0]
    index = 0
    for i in range(1, len(list)):
        if list[i] > max:
            max = list[i]
            index = i
    return max, index

def min_num_in_list( list ):
    min = list[ 0 ]
    index = 0
    for i in range(1,len(list)):
        if list[i] < min:
            min = list[i]
            index = i
    return min,index


def getListOfIndex(listOfRep):
    listOfEmptySpace = []
    listOfRepCop = list()

    for value in listOfRep:
        listOfRepCop.append(value)

    for index in range(0,len(listOfRepCop)):
        if listOfRepCop.count(listOfRepCop[index]) > 1:
            listOfEmptySpace.append(index)
            listOfRepCop[index] =  -1
    return listOfEmptySpace



def getUForId(idChoose,dictTotal,m):
    dictOfU = dict()
    for idX in idChoose:
        listOfDist = []
        for idDist in dictTotal[idX]:
            # print(idDist)
            if not idDist['id'] in idChoose:
                if idDist['distance'] != 0:
                    num = pow((1 / idDist['distance']), (1 / (m - 1)))
                    den = 0.0
                    for idY in idChoose:
                        if getDistOt(dictTotal[idY], idDist['id']) != 0:
                            den += pow(((1 / getDistOt(dictTotal[idY], idDist['id']))), (1 / (m - 1)))

                    if den != 0:
                        distance = num / den
                    else:
                        distance = 0

                    dictSingl = {"id": idDist['id'], "u": distance}
                else:
                    dictSingl = {"id": idDist['id'], "u": 1}
                listOfDist.append(dictSingl)
        dictOfU[idX] = listOfDist
    return dictOfU



def getInitialIndex(dictTotal,cluster):
    listOfMinDist = list()
    listOfMinId = list()
    dictSumDistance = dict()
    for idX in dictTotal.keys():
        somma = 0
        for singleDict in dictTotal[idX]:
            somma += singleDict["distance"]
        dictSumDistance[idX]=somma
        if len(listOfMinDist) < cluster:
            listOfMinDist.append(dictSumDistance[idX])
            listOfMinId.append(idX)
        else:
            if max_num_in_list(listOfMinDist)[0] > somma:
                listOfMinDist[max_num_in_list(listOfMinDist)[1]] = dictSumDistance[idX]
                listOfMinId[listOfMinDist.index(dictSumDistance[idX])] = idX

    return listOfMinId

def getMinAccept(dictU,minVal):
    listOfMemb = []
    for value in dictU:
        listOfMemb.append(value['u'])


    listOfMemb.sort()
    indexCount = int(len(listOfMemb)*minVal)
    return listOfMemb[indexCount]