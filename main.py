from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint

from scipy.fftpack import fft
from scipy.io import wavfile  # get the api

import os
import io


import keras
import numpy


pathToFolder = './data_v_7_stc/'
pathToDataAudio = 'audio/'
pathToMeta = 'meta/meta.txt'
pathToTestData = "test/"
pathToResult = 'result.txt'
pathToBest = pathToFolder +'bestModel'

classes = ['background', 'bags', 'door', 'keyboard', 'knocking_door', 'ring', 'speech', 'tool']


noSubclasses = 80
fragmentDuration = 0.25  # in secs
maxDuration = 10.0
fragmentsCountInSet = int(maxDuration / fragmentDuration)

UsefulHZ = 11025  # As most (~8000) files has Sample Rate 22050

datasets = []
classifications = []


def getClassification(fileName):
    ret = numpy.zeros(len(classes))
    with io.open(pathToFolder + pathToMeta, encoding='utf-8') as file:
        for line in file:
            if fileName in line:
                lst = line.split()
                ind = classes.index(lst[4])
                ret[ind] = 1.0
    return ret


def procOneFragment(signalSamples, startSample, endSample, fs):
    retAmps = []
    signalChunk = signalSamples[startSample: endSample]

    nonZero = numpy.nonzero(signalChunk)[0]
    if len(nonZero) == 0:
        return retAmps

    signalChunk = signalChunk[nonZero[0]:]
    spectrs = fft(signalChunk)

    upl = len(spectrs) * UsefulHZ / fs  # upl -- useful part length
    uplRound = int(upl)

    amps = abs(spectrs[1: uplRound])  # rid of unnecessary part and got the magnitude
    lAmp = len(amps)
    if lAmp < noSubclasses:
        return retAmps

    maxAmp = numpy.max(amps)
    if maxAmp == 0:
        return retAmps

    amps = amps / maxAmp  # normalise
    delta = int(lAmp / noSubclasses)
    for i in range(0, noSubclasses):
        start = delta * i
        end = (delta + 1) * i
        if end < lAmp and i == noSubclasses:
            end = lAmp
        tempAmps = amps[start: end]
        mean = float(sum(tempAmps)) / max(len(tempAmps), 1)
        retAmps.append(mean)
    return retAmps

def createDerivativePair(retAmpsNumpy):
    step = UsefulHZ / noSubclasses
    deriv = []
    npair = numpy.array([retAmpsNumpy[0], 0])
    deriv.append(npair)
    for i in range(1, len(retAmpsNumpy) - 1):
        orig = retAmpsNumpy[i]
        der = (retAmpsNumpy[i+1] - retAmpsNumpy[i-1]) / step
        npair = numpy.array([orig, der])
        deriv.append(npair)
    npair = numpy.array([retAmpsNumpy[len(retAmpsNumpy) - 1], 0])
    deriv.append(npair)
    return numpy.array(deriv)



def prepareData(folder, fileName):
    dset = []
    fullPath = folder + fileName
    fs, data = wavfile.read(fullPath)  # Has to be 22050HZ
    signalSamples = data.T

    fragmentNo = 0
    okFrament = 0
    samplesLength = len(signalSamples)
    while fragmentNo * fragmentDuration * fs < samplesLength:
        startSec = fragmentNo * fragmentDuration
        endSec = (fragmentNo + 1) * fragmentDuration
        startSample = int(fs * startSec)
        endSample = int(fs * endSec)

        if endSample > samplesLength:
            endSample = samplesLength

        retAmps = procOneFragment(signalSamples, startSample, endSample, fs)
        if len(retAmps) != noSubclasses:
            fragmentNo += 1
            continue

        retAmpsNumpy = numpy.array(retAmps)
        retAmpsDerPair = createDerivativePair(retAmpsNumpy)
        dset.append(retAmpsDerPair)
        fragmentNo += 1
        okFrament += 1


    if okFrament % fragmentsCountInSet == 0:
        return dset

    okFrament %= fragmentsCountInSet
    for i in range(okFrament, fragmentsCountInSet, 1):
        retAmpsNumpy = numpy.zeros(noSubclasses)
        retAmpsDerPair = createDerivativePair(retAmpsNumpy)
        dset.append(retAmpsDerPair)

    return dset


def classifyFile(fileName):
    classif = getClassification(fileName)
    dset = prepareData(pathToFolder + pathToDataAudio, fileName)
    dsetLen = len(dset)
    for i in range(0, dsetLen, fragmentsCountInSet):
        ndata = dset[i: i + fragmentsCountInSet]
        nset = numpy.array(ndata)
        datasets.append(nset)
        classifications.append(classif)


def buildNetwork():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), input_shape=(fragmentsCountInSet, noSubclasses, 2)))
    network.add(BatchNormalization())
    network.add(Activation('relu'))
    network.add(Conv2D(32, (3, 3)))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    network.add(Activation('relu'))
    network.add(Conv2D(64, (3, 3)))
    network.add(BatchNormalization())
    network.add(Activation('relu'))
    network.add(Conv2D(64, (3, 3)))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    network.add(Activation('relu'))
    network.add(Conv2D(128, (3, 3)))
    network.add(BatchNormalization())
    network.add(Activation('relu'))
    network.add(Conv2D(128, (3, 3)))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    network.add(Activation('relu'))
    network.add(Flatten())
    network.add(Dense(768))
    network.add(Dropout(0.5))
    network.add(Activation('relu'))
    network.add(Dense(768))
    network.add(Dropout(0.5))
    network.add(Activation('relu'))
    network.add(Dense(len(classes), activation='softmax'))
    return network


def learn():
    global datasets
    global classifications

    files = os.listdir(pathToFolder + pathToDataAudio)
    soundFiles = filter(lambda x: x.endswith('.wav'), files)
    for file in soundFiles:
        classifyFile(file)

    datasets = numpy.array(datasets)
    classifications = numpy.array(classifications)

    # break-up datasets
    datasetsSize = len(datasets)
    indices = numpy.arange(datasetsSize)
    numpy.random.seed(2)
    numpy.random.shuffle(indices)
    trainSize = int(datasetsSize * 0.85)
    trainIndices = indices[0: trainSize]
    verifyIndices = indices[trainSize:]
    trainingDatasets = datasets[trainIndices]
    verifyingDatasets = datasets[verifyIndices]
    trainingClassifications = classifications[trainIndices]
    verifyingClassifications = classifications[verifyIndices]

    model = buildNetwork()
    opt = optimizers.adam()
    mdp = ModelCheckpoint(pathToBest, monitor='val_acc', save_best_only=True)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    model.fit(trainingDatasets, trainingClassifications, batch_size=32, epochs=50,
              validation_data=(verifyingDatasets, verifyingClassifications),
              callbacks=[mdp])
    scores = model.evaluate(verifyingDatasets, verifyingClassifications)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def predict(model):
    resultFile = io.open(pathToFolder + pathToResult, "w")

    testFiles = os.listdir(pathToFolder + pathToTestData)
    soundFiles = filter(lambda x: x.endswith('.wav'), testFiles)
    for file in soundFiles:
        testSet = prepareData(pathToFolder + pathToTestData, file)
        dsetLen = len(testSet)
        datasets = []
        for i in range(0, dsetLen, fragmentsCountInSet):
            nset = numpy.array(testSet[i: i + fragmentsCountInSet])
            datasets.append(nset)

        datasets = numpy.array(datasets)
        classif = model.predict(datasets)

        classif = sum(classif) / max(len(classif), 1)
        m = max(classif)

        ind = 0
        for i in range(0, 8):
            if m == classif[i]:
                ind = i
                break

        className = classes[ind]
        resultFile.write("%s    %.2f%%  %s\n" % (file, classif[ind] * 100, className))
    resultFile.close()

def main():
    learn()
    model = keras.models.load_model(pathToBest)
    predict(model)

main()