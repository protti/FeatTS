import time
import numpy as np
from src import FeatTS

if __name__ == '__main__':
    with open('result_FeatTS.csv','a') as f:
            f.write("Dataset,AMI,NMI,RI,ARI,Time\n")

    listDataset = ['Coffee', 'Haptics', 'SyntheticControl', 'Worms', 'Computers', 'HouseTwenty', 'GestureMidAirD3', 'Chinatown',
                   'UWaveGestureLibraryAll', 'Strawberry', 'Car', 'GunPointAgeSpan', 'GestureMidAirD2', 'BeetleFly',
                   'Wafer', 'CBF', 'Adiac', 'ItalyPowerDemand', 'Yoga', 'AllGestureWiimoteY', 'Trace',
                   'PigAirwayPressure',
                   'ShapesAll', 'Beef', 'GesturePebbleZ2', 'Mallat', 'GunPointOldVersusYoung', 'MiddlePhalanxTW',
                   'AllGestureWiimoteX', 'Meat', 'Herring', 'MiddlePhalanxOutlineCorrect', 'InsectEPGRegularTrain',
                   'FordA', 'SwedishLeaf', 'InlineSkate', 'DodgerLoopDay', 'UMD', 'CricketY', 'WormsTwoClass',
                   'SmoothSubspace', 'OSULeaf', 'Ham', 'CricketX', 'SonyAIBORobotSurface1', 'ToeSegmentation1',
                   'ScreenType', 'PigArtPressure', 'SmallKitchenAppliances', 'Crop', 'MoteStrain',
                   'MelbournePedestrian',
                   'ECGFiveDays', 'Wine', 'SemgHandMovementCh2', 'FreezerSmallTrain', 'UWaveGestureLibraryZ',
                   'NonInvasiveFetalECGThorax1',
                   'TwoLeadECG', 'Lightning7', 'Phoneme', 'SemgHandSubjectCh2', 'DodgerLoopWeekend',
                   'MiddlePhalanxOutlineAgeGroup',
                   'GestureMidAirD1', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'FacesUCR', 'ECG5000',
                   'ShakeGestureWiimoteZ',
                   'GesturePebbleZ1', 'HandOutlines', 'GunPointMaleVersusFemale', 'Rock',
                   'MixedShapesSmallTrain',
                   'AllGestureWiimoteZ', 'FordB', 'FiftyWords', 'InsectWingbeatSound', 'MedicalImages', 'Symbols',
                   'ArrowHead',
                   'ProximalPhalanxOutlineAgeGroup', 'EOGHorizontalSignal', 'TwoPatterns', 'ChlorineConcentration',
                   'Plane', 'ACSF1',
                   'PhalangesOutlinesCorrect', 'ShapeletSim', 'DistalPhalanxOutlineAgeGroup', 'InsectEPGSmallTrain',
                   'PickupGestureWiimoteZ',
                   'EOGVerticalSignal', 'CricketZ', 'FaceFour', 'RefrigerationDevices', 'PLAID',
                   'MixedShapesRegularTrain', 'GunPoint',
                   'DodgerLoopGame', 'ECG200', 'ToeSegmentation2', 'WordSynonyms', 'Fungi', 'BirdChicken',
                   'SemgHandGenderCh2',
                   'OliveOil', 'BME', 'LargeKitchenAppliances', 'SonyAIBORobotSurface2', 'Lightning2', 'EthanolLevel',
                   'UWaveGestureLibraryX', 'FreezerRegularTrain', 'Fish', 'ProximalPhalanxOutlineCorrect',
                   'NonInvasiveFetalECGThorax2',
                   'UWaveGestureLibraryY', 'FaceAll', 'StarLightCurves', 'ElectricDevices', 'Earthquakes', 'PowerCons',
                   'DiatomSizeReduction', 'CinCECGTorso', 'PigCVP', 'ProximalPhalanxTW']

    for nameDataset in listDataset:
        timeStart = time.time()
        amiValue, normMutualInfo, randIndex, adjRandInd = FeatTS.FeatTS(nameDataset,True)
        finalTime = time.time() - timeStart
        with open('result_FeatTS.csv','a') as f:
            f.write("{},{},{},{},{},{}\n".format(
                nameDataset,
                amiValue,
                normMutualInfo,
                randIndex,
                adjRandInd,
                finalTime))