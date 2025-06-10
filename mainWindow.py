import logging
import os
import re

import cv2
import imageio
import copy
import pickle
import shutil

import keras.regularizers
import matplotlib.patches
import numpy as np
import pandas as pd
from keras import models
from scipy import spatial

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from openpyxl import load_workbook

import core.CNNClassifier
import gui.mriWindow

from generated.mriWindow_ui import Ui_MRIWindow
from generated import mainWindow_ui
from util import constants, util
from util.constants import config
from util.fileDialog import FileDialog
from core import zonare
from core import iqBModeConverter
from core import segmentation
from core import IQ2RFconverter
from core import fatMapping
from core import classifier

from core.loadImageData import convertImageFromArray, changeToBinaryMask, floodFillContour
from core.HyperspectralImagesGeneration import *
from core.CNNClassifier import *

import gui.sliceWidget

# Packages necessary for new scanner data
from core import rdataread as rd
from scipy.signal import hilbert
from os import walk, path

logger = logging.getLogger(__name__)

# class for main window of UI
class MainWindow(QMainWindow, mainWindow_ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.parent = parent
        self.mriWindow = gui.mriWindow.MRIWindow(parent=self)

        # uncomment if you want the MRI window to open on starting the app
        # self.mriWindow.show()

        MainWindow.setFixedSize(self, 1227, 789)

        self.sliceWidgetBMode.mpl_connect('motion_notify_event', self.on_sliceWidgetBMode_mouseMoved)
        self.sliceWidgetBMode.mpl_connect('key_press_event', self.on_sliceWidgetBMode_keyPressed)
        self.sliceWidgetBMode.mpl_connect('button_press_event', self.on_sliceWidgetBMode_clicked)

        # Setting widget flag as 0 for IQ slice widget and 1 for IQ slice widget, 0 and 1 are arbitrary numbers
        self.sliceWidgetIQ.widgetFlag = 0
        self.sliceWidgetBMode.widgetFlag = 1

        self.sliceWidgetIQ.mpl_connect('motion_notify_event', self.on_sliceWidgetIQ_mouseMoved)
        # self.sliceWidgetIQ.mpl_connect('key_press_event', self.on_sliceWidgetIQ_keyPressed)
        self.sliceWidgetIQ.mpl_connect('button_press_event', self.on_sliceWidgetIQ_clicked)

        self.setupDefaults()
        self.loadSettings()

        self.roiDataList = None
        self.landMarkViewNumber = 0

        self.manualBModeROIs = []
        self.manualIQROIs = []
        self.tissueTypeList = []
        self.tissueType = 0

        self.bModeImages = []
        self.bModeImagesTemp = []

        self.sliceWidgetBMode.isLAS = True
        self.ptSettings = None
        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        self.progressBarImageRecon.setMaximum(100)
        self.progressBarImageRecon.setValue(0)

        self._iqFileName = ''
        self._iqFileNamePath = ''

        self.contourPointFilePath = ''

        self.bModeMorphPoints = []

        self.iqImageHeight = config["IMAGE PROPERTIES"]["IQ IMAGE DIMENSIONS"][0]
        self.iqImageWidth = config["IMAGE PROPERTIES"]["IQ IMAGE DIMENSIONS"][1]

        self.bmodeImageHeight = config["IMAGE PROPERTIES"]["BMODE IMAGE DIMENSIONS"][0]
        self.bmodeImageWidth = config["IMAGE PROPERTIES"]["BMODE IMAGE DIMENSIONS"][1]
        self.bmodeInitialRadius = config["IMAGE PROPERTIES"]["INITIAL RADIUS"]
        self.bmodeFinalRadius = config["IMAGE PROPERTIES"]["FINAL RADIUS"]
        self.bmodeImageDepthMM = config["IMAGE PROPERTIES"]["BMODE IMAGE DEPTH MM"]

        self.dataTopLevelPath = config["ULTRASOUND CLASSIFICATION"]["DATA TOP LEVEL PATH"]
        self.iqChannelROIsPath = config["ULTRASOUND CLASSIFICATION"]["US CHANNEL AND ROIS PATH"]
        self.channelSpectraPath = config["ULTRASOUND CLASSIFICATION"]["CHANNEL SPECTRA PATH"]
        self.classifierModelsPath = config["ULTRASOUND CLASSIFICATION"]["CLASSIFIER MODELS PATH"]

        self.usDataPath = config["ULTRASOUND CLASSIFICATION"]["ULTRASOUND DATA PATH"]
        self.mriDataPath = config["ULTRASOUND CLASSIFICATION"]["MRI DATA PATH"]

        self.observerPriority = config["OBSERVERS"]["OBSERVER PRIORITY"]

        if not os.path.exists(self.dataTopLevelPath):
            os.makedirs(self.dataTopLevelPath)

        self.fatPoints = []
        self.shiftNum = 0
        self.binaryFat = 0
        self.fatMask = np.zeros((self.iqImageHeight, self.iqImageWidth))
        self.nonFatMask = np.zeros((self.iqImageHeight, self.iqImageWidth))
        self.channel = np.zeros((self.iqImageHeight, self.iqImageWidth))
        self.fatMaskBMode = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))
        self.nonFatMaskBMode = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))
        self.channelBMode = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))
        self.channelOutsidePoints = None
        self.channelParam = np.zeros((1, 1000))

        self.ROIThreshPercent = 1
        self.ROIFatFlag = None
        self.contourSlicesIQ = []
        self.contourSlicesBMode = []
        self.countourSliceFrameNums = []

        self.showSpectraFlag = False
        self.paramList = []
        self.paramArray = None
        self.normParamArray = None

        self.EDFrame = 0
        self.ESFrame = 0

        self.classifierAmpThresh = 30
        self.roiAvgThresh = 10
        self.ant_inf_size = 50

        self.classifierAcc = 0
        self.classifierSens = 0
        self.classifierSpec = 0
        self.classifierPrec = 0
        self.classifierRec = 0
        self.classifierFscore = 0
        self.classifierShuffleFlag = True

        self.CNNClassifierAcc = 0
        self.CNNClassifierSens = 0
        self.CNNClassifierSpec = 0
        self.CNNClassifierPrec = 0
        self.CNNClassifierRec = 0
        self.CNNClassifierFscore = 0

        self.accAvg = 0
        self.sensAvg = 0
        self.specAvg = 0
        self.precAvg = 0
        self.recAvg = 0
        self.accSpread = 0
        self.sensSpread = 0
        self.specSpread = 0
        self.precSpread = 0
        self.recSpread = 0
        self.accStd = 0
        self.sensStd = 0
        self.specStd = 0
        self.precStd = 0
        self.recStd = 0

        self.modelSaveFlag = 1
        self.modelSaveFolder = ''
        self.saveCNNFlag = 0

        self.binaryFatThresh = self.mriWindow.get_binary_fat_thresh_mm()

        fat1 = str(self.binaryFatThresh).split('.')[0]
        fat2 = str(self.binaryFatThresh).split('.')[1]
        # self.SAVEPATH = fr'D:\school\BIRL\us_fat_thickness_data_rois\fat_{fat1}_{fat2}\thickness_{self.sliceWidgetBMode.ContourThickness}'  # path to save contours/data/ROIs
        self.SAVEPATH = fr'{self.iqChannelROIsPath}\fat_{fat1}_{fat2}\thickness_{self.sliceWidgetBMode.ContourThickness}'  # path to save contours/data/ROIs

        self.mriPath = None

        self.autoLoadFlag = 0
        self.iqPath = None
        self.contourPath = None
        self.thicknessList = [12]
        self.fatThreshList = [3.5]
        self.roiNumList = [30]
        self.roiAvgThreshList = [0]

        self.thicknessListRF = [10]
        self.fatThreshListRF = [5.5]
        self.roiNumListRF = [60]
        self.roiAvgThreshListRF = [0]
        self.probThreshList = [0]
        self.learningRateList = [0.000025]
        self.learningRatePatienceList = [10]
        self.learningRateDropList = [0.2]
        self.batchSizeList = [4]
        # self.classifiersList = [
        #     'RF',
        #     'QDA',
        #     'SVM',
        #     'AdaBoost',
        #     'KNN',
        #     'MLP',
        #     'XGBoost',
        #     'LGBM'
        # ]
        self.classifiersList = ["RF"]

        # self.DLclassifiersList = {
        #     "VuNet1D": uNetModel1D(num_filters=32, input_size=(64, 13), binary=True, num_classes=1,
        #                            dropout_rate=0.4, dilation_rate=4),
        #     "SimpleUNet1D": simpleModel1D(num_filters=16, input_size=(64, 13), binary=True, num_classes=1,
        #                                   dropout_rate=0.4, dilation_rate=4),
        #     "DmitrievaUNet1D": uNet1D_Dmitrieva(num_filters=16, kernel=9, input_size=(64, 13), num_classes=1,
        #                                         dropout_rate=0.4),
        #     "SimpleRNN": simpleRNN(input_size=(64, 13))
        # }
        self.DLLearningRate = 0.00300500155092942
        self.DLLearningRatePatience = 14
        self.DLLearningRateDrop = 0.3
        self.DLBatchSize = 2
        self.DLVerbose = 2

        self.prob_thresh = 0

        self.predictedLabels = []
        self.predictedThickness = []

        self.modelPath = None
        self.fatDSCList = []
        self.fatMSEList = []

        self.avgFatThickness = 0

        self.machineLearningListWidget.setCurrentRow(0)
        self.deepLearningListWidget.setCurrentRow(0)
        self.classifierTypeML = 'RF'    # machine learning classifier type to be used in "createClassifier"
        self.classifierTypeDL = "VuNet1D"   # deep learning classifier type to be used in "UnetClassifier"

        ######### Hide buttons from GUI that aren't currently being used ############
        self.pushButtonMergeMRI.setVisible(False)
        self.pushButtonSaveFiducial.setVisible(False)
        self.pushButtonLoadFiducial.setVisible(False)
        self.pushButtonRemoveMorphPoint.setVisible(False)
        self.pushButtonClearMorphPoints.setVisible(False)
        self.pushButtonExportFatNonFatData.setVisible(False)
        self.pushButtonDrawFatROIs.setVisible(False)
        self.pushButtonDrawNonFatROIs.setVisible(False)
        self.pushButtonFatMapEvalAll.setVisible(False)
        self.pushButtonDrawChannelROIs_2.setVisible(False)

        self.pushButtonDrawNonFatROIs.setEnabled(False)
        self.ROIsizehoriz.setEnabled(False)
        self.ROIsizevert.setEnabled(False)

    def loadSettings(self):
        settings = QSettings(constants.applicationName, constants.organizationName)
        settings.beginGroup('EchoFatSegmentationGUI')

        geometry = settings.value('geometry', QByteArray(), type=QByteArray)
        if not geometry.isEmpty():
            self.restoreGeometry(geometry)

            # Fixes QTBUG-46620 issue
            if settings.value('maximized', False, type=bool):
                self.showMaximized()
                self.setGeometry(QApplication.desktop().availableGeometry(self))

        # default path
        self.defaultOpenPath = settings.value('defaultOpenPath', QDir.homePath(), type=str)

        settings.endGroup()

    def saveSettings(self):
        settings = QSettings(constants.applicationName, constants.organizationName)
        settings.beginGroup('EchoFatSegmentationGUI')

        settings.setValue('geometry', self.saveGeometry())
        settings.setValue('maximized', self.isMaximized())
        settings.setValue('defaultOpenPath', self.defaultOpenPath)

        settings.endGroup()

    def setupDefaults(self):
        self.sliceSlider.setValue(self.sliceWidgetBMode.sliceNumber)
        self.sliceSlider.setMinimum(0)
        self.sliceSlider.setMaximum(100)  # TODO set to max value of image loop based on number of images
        self.labelSlice.setText('Slice: 0')
        self.roiModeRadioButton.setChecked(True)

    @pyqtSlot(bool)
    def on_roiModeRadioButton_toggled(self, checked):
        if not checked:
            return
        print('roi mode radio button checked')

        # sets tab to ROI to match the mode
        self.tabWidget.setCurrentIndex(0)

        # TODO figure out why this won't run if uncommented
        # self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings)
        # self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings)

    @pyqtSlot(bool)
    def on_tissueMappingModeRadioButton_toggled(self, checked):
        if not checked:
            return
        print('tissue mapping mode radio button checked')

        # sets tab to Tissue Mapping to match the mode
        self.tabWidget.setCurrentIndex(1)

        # TODO figure out why this won't run if uncommented
        # self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings)
        # self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings)

    @pyqtSlot(bool)
    def on_segmentationModeRadioButton_toggled(self, checked):
        if not checked:
            return
        print('segmentation mode radio button checked')

        # sets tab to Segmentation to match the mode
        self.tabWidget.setCurrentIndex(2)

        # TODO figure out why this won't run if uncommented
        # self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings)

    @pyqtSlot(bool)
    def on_mergeMRIModeRadioButton_toggled(self, checked):
        if not checked:
            return
        print('merge MRI radio button checked')

        self.tabWidget.setCurrentIndex(3)

    @pyqtSlot(bool)
    def on_classifyModeRadioButton_toggled(self, checked):
        if not checked:
            return
        print('classify radio button checked')
        self.tabWidget.setCurrentIndex(4)

    @pyqtSlot(bool)
    def on_fatMapModeRadioButton_toggled(self, checked):
        if not checked:
            return
        print('fat map radio button checked')
        self.tabWidget.setCurrentIndex(5)

    @pyqtSlot(bool)
    def on_fatRadioButton_toggled(self, checked):
        if not checked:
            return

        self.tissueType = 1
        print(self.tissueType)

    @pyqtSlot(bool)
    def on_myocardiumRadioButton_toggled(self, checked):
        if not checked:
            return

        self.tissueType = 2
        print(self.tissueType)

    @pyqtSlot(bool)
    def on_bloodRadioButton_toggled(self, checked):
        if not checked:
            return

        self.tissueType = 3
        print(self.tissueType)

    @pyqtSlot(bool)
    def on_nonFatRadioButton_toggled(self, checked):
        if not checked:
            return

        self.tissueType = 4
        print(self.tissueType)

    @pyqtSlot(int)
    def on_sliceSlider_valueChanged(self, value):
        print('slice number: %d' % value)

        self.bModeImages.clear()
        self.bModeImages.append(self.bModeImagesTemp[int(value)])

        if value == int(self.EDFrame):
            self.EDRadioButton.setChecked(True)
            self.ESRadioButton.setChecked(False)
            MainWindow.on_EDRadioButton_clicked(self)
        elif value == int(self.ESFrame):
            self.ESRadioButton.setChecked(True)
            self.EDRadioButton.setChecked(False)
            MainWindow.on_ESRadioButton_clicked(self)
        else:
            self.EDRadioButton.setChecked(False)
            self.ESRadioButton.setChecked(False)

        # Clears drawn ROI lists when the slice changes
        self.manualBModeROIs.clear()
        self.manualIQROIs.clear()

        self.sliceWidgetBMode.sliceNumber = value
        self.sliceWidgetIQ.sliceNumber = value
        self.labelSlice.setText('Slice: %d' % value)
        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonGoToED_clicked(self):
        # jump to end diastole frame obtained from name of expert tracing file
        MainWindow.on_sliceSlider_valueChanged(self, int(self.EDFrame))

    @pyqtSlot()
    def on_pushButtonGoToES_clicked(self):
        # jump to end systole frame obtained from name of expert tracing file
        MainWindow.on_sliceSlider_valueChanged(self, int(self.ESFrame))

    @pyqtSlot()
    def on_sliceWidgetBMode_mouseMoved(self, event):
        pass
        # print('on_sliceWidgetBMode_mouseMoved')

    @pyqtSlot()
    def on_sliceWidgetIQ_mouseMoved(self, event):
        pass
        # print('on_sliceWidgetIQ_moved: x: %d, y: %d and x data: %d, y data: %d' % (event.x, event.y, event.xdata, event.ydata))

    @pyqtSlot()
    def on_sliceWidgetBMode_keyPressed(self, event):
        if event.key == 'left':
            print('on_sliceWidgetBMode_keyPressed: left')
        elif event.key == 'right':
            print('on_sliceWidgetBMode_keyPressed: right')
        if event.key == 'delete':
            print('Last ROI deleted')

    @pyqtSlot()
    def on_sliceWidgetBMode_clicked(self, event):

        # check to see which radio button is active indicating which "mode" is active
        if self.roiModeRadioButton.isChecked():  # roi mode, so enter roi
            self.x = round(event.xdata, 3)
            self.y = round(event.ydata, 3)
            print('on_sliceWidgetBMode_clicked: %d, %d' % (self.x, self.y))

            self.sliceWidgetIQ.secondPointIQ = None
            self.sliceWidgetIQ.firstPointIQ = None

            # After one ROI is selected checks for next point and sets it to first point and sets second point to None
            if self.sliceWidgetBMode.secondPointBMode is not None:
                self.sliceWidgetBMode.firstPointBMode = [event.xdata, event.ydata]
                self.sliceWidgetBMode.secondPointBMode = None
                return

            # Sets second point
            elif ((self.sliceWidgetBMode.secondPointBMode is None) and (
                    self.sliceWidgetBMode.firstPointBMode is not None)):
                self.sliceWidgetBMode.secondPointBMode = [event.xdata, event.ydata]
                clickedROI = [self.sliceWidgetBMode.firstPointBMode, self.sliceWidgetBMode.secondPointBMode]
                self.manualBModeROIs.append(clickedROI)
                self.tissueTypeList.append(self.tissueType)

                self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                   self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
                self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
                return

            # For the first ROI, checks for zero points selected and sets the intial first point
            elif ((self.sliceWidgetBMode.secondPointBMode is None) and (self.sliceWidgetBMode.firstPointBMode is None)):
                self.sliceWidgetBMode.firstPointBMode = [event.xdata, event.ydata]


        elif self.segmentationModeRadioButton.isChecked():  # segmentation mode
            print('on_sliceWidgetBMode_clicked: segmentation mode')
            if self.sliceWidgetBMode.imageBMode is None:
                print('No image, pixel coords (%d, %d), data coords (%d, %d).' %
                      (event.x, event.y, event.xdata, event.ydata))
            else:
                print('Image loaded, pixel coords (%d, %d), data coords (%d, %d).' %
                      (event.x, event.y, event.xdata, event.ydata))

                # check to see if it is first press, second press, or beyond
                if self.sliceWidgetBMode.segSecondButtonPress is not None:
                    print('Already have both points.')
                    return  # do nothing if already have both locations

                elif ((self.sliceWidgetBMode.segSecondButtonPress is None) and (
                        self.sliceWidgetBMode.segFirstButtonPress is not None)):
                    print('Have first point, entering second.')
                    self.sliceWidgetBMode.segSecondButtonPress = [event.xdata, event.ydata]
                    self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                       self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

                elif ((self.sliceWidgetBMode.segSecondButtonPress is None) and (
                        self.sliceWidgetBMode.segFirstButtonPress is None)):
                    print('Do not have either point, entering first.')
                    self.sliceWidgetBMode.segFirstButtonPress = [event.xdata, event.ydata]
                    self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                       self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

                else:
                    print('Do not know how many points have been entered.')

        elif self.tissueMappingModeRadioButton.isChecked():  # tissue mapping mode
            print('on_sliceWidget_clicked: tissue mapping mode')

        ## old mode for merging mri and us images - not used
        # elif self.mergeMRIModeRadioButton.isChecked():
        #     self.x = round(event.xdata, 0)
        #     self.y = round(event.ydata, 0)
        #
        #     self.sliceWidgetBMode.morphBModePoints.append((int(self.x), int(self.y)))
        #     self.bModeMorphPoints.append((int(self.x), int(self.y)))
        #
        #     print('on_sliceWidget_clicked: merge MRI mode\nNumber of BMode points: %i' % (
        #         len(self.sliceWidgetBMode.morphBModePoints)))
        #
        #     self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
        #                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_sliceWidgetIQ_clicked(self, event):
        self.x = round(event.xdata, 3)
        self.y = round(event.ydata, 3)
        print('on_sliceWidgetIQ_clicked: %d, %d' % (self.x, self.y))

        self.sliceWidgetBMode.firstPointBMode = None
        self.sliceWidgetBMode.secondPointBMode = None

        # After one ROI is selected checks for next point and sets it to first point and sets second point to None
        if self.sliceWidgetIQ.secondPointIQ is not None:
            self.sliceWidgetIQ.firstPointIQ = [event.xdata, event.ydata]
            self.sliceWidgetIQ.secondPointIQ = None
            return

        # Sets second point
        elif ((self.sliceWidgetIQ.secondPointIQ is None) and (self.sliceWidgetIQ.firstPointIQ is not None)):

            self.sliceWidgetIQ.secondPointIQ = [event.xdata, event.ydata]
            clickedROI = [self.sliceWidgetIQ.firstPointIQ, self.sliceWidgetIQ.secondPointIQ]
            self.manualIQROIs.append(clickedROI)
            self.tissueTypeList.append(self.tissueType)

            self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                               self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
            self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                            self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        # For the first ROI, checks for zero points selected and sets the initial first point
        elif ((self.sliceWidgetIQ.secondPointIQ is None) and (self.sliceWidgetIQ.firstPointIQ is None)):

            self.sliceWidgetIQ.firstPointIQ = [event.xdata, event.ydata]

    @pyqtSlot()
    def on_pushButtonEditConfig_clicked(self):
        try:
            os.startfile('config.json')

        except FileNotFoundError:
            QMessageBox.warning(self, 'Unable to open config file',
                                'config.json does not exist',
                                QMessageBox.Ok)

    @pyqtSlot()
    def on_pushButtonLoadROI_clicked(self):
        """load and draw ROIs from .txt file"""
        fileNameFullPath, filterReturn = FileDialog.getOpenFileName(self, 'Select ROI Data text file',
                                                                    self.defaultOpenPath, '*.txt')
        if not fileNameFullPath:
            return

        # cleaning up file name
        self._roiFileNamePath = util.cleanPath(fileNameFullPath)
        self._roiFileName = util.splitext(os.path.basename(self._roiFileNamePath))[0]

        # extract the text data from the ROI file
        roiData = open(self._roiFileNamePath, "r")
        self.roiDataList = roiData.read().splitlines()

        self.tissueTypeList.clear()
        self.manualBModeROIs.clear()
        self.manualIQROIs.clear()

        # First 2 lines are Case and labels
        # Need tissue type(3), aline start(4), aline stop(5), sample start(6), sample stop(7)
        # loop thru rows and grab needed data - discard first 2 rows
        for i in range(2, len(self.roiDataList)):
            row = self.roiDataList[i].split(", ")

            self.tissueType = int(row[3])
            alinestart = int(row[4])
            alinestop = int(row[5])
            samplestart = int(row[6])
            samplestop = int(row[7])

            self.sliceWidgetIQ.firstPointIQ = [alinestart, samplestart]
            self.sliceWidgetIQ.secondPointIQ = [alinestop, samplestop]

            ROI = [self.sliceWidgetIQ.firstPointIQ, self.sliceWidgetIQ.secondPointIQ]
            self.manualIQROIs.append(ROI)
            self.tissueTypeList.append(self.tissueType)

        print(self._roiFileName)
        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonLoadIQ_clicked(self):
        print('on_pushButtonLoadIQ_clicked')

        # changed to zero for our volunteer iq data to work
        newdata = 0
        if newdata:
            # Lena code for new scanner
            newdata = 0
            path, _ = FileDialog.getOpenFileName(self, 'Select file',
                                                                        self.defaultOpenPath)

            #if not fileNameFullPath:
            if not path:
                return

            # Read in tar files from new ultrasound
            ext = path.split('.')[-1]
            if ext == 'tar':
                name1 = path.split('/')[-1]
                path1 = path + ' -aoa -o' + path[:-len(name1)]
                os.system('7z.exe x ' + path1)
                #path = path[:-4]

            #files = os.listdir(path[:-len(name1)])
            #ignored = {'.tar'}
            #files = [x for x in os.listdir(path[:-len(name1)]) if x not in ignored]

            ext = path.split('.')[-1]
            if ext == 'lzo':
                os.system('lzop.exe -d ' + path)
                path = path[:-4]

            if path[-3:] == 'raw':        # Clarius raw data
                if path[-7:-4] == 'env':    # Extract Envelope data
                    hdr, timestamps, data = rd.read_env(path)
                    data = np.transpose(data)

                    # display b data
                    numframes = 1  # hdr['frames']
                    for frame in range(numframes):
                        plt.figure(figsize=(5, 5))
                        plt.imshow(np.transpose(data[:, :, frame]), cmap=plt.cm.gray, aspect='auto', vmin=0, vmax=255)
                        plt.title('envelope frame ' + str(frame))
                        plt.show()

                elif path[-6:-4] == 'iq':   # Extract IQ data
                    hdr, timestamps, data = rd.read_iq(path)
                    # separating i and q data
                    idata = data[:, 0::2, :]
                    qdata = data[:, 1::2, :]

                    # covnert IQ to B
                    numframes = 1  # hdr['frames']
                    bdata = np.zeros((hdr['lines'], hdr['samples'], hdr['frames']), dtype='float')
                    for frame in range(numframes):
                        bdata[:, :, frame] = 10 * np.log10(
                            1 + np.power(idata[:, :, frame], 2) + np.power(qdata[:, :, frame], 2))

                    # display images
                    for frame in range(numframes):
                        plt.figure(figsize=(15, 5))
                        plt.subplot(1, 3, 1)
                        plt.imshow(np.transpose(idata[:, :, frame]), cmap=plt.cm.viridis, aspect='auto', vmin=-100,
                                   vmax=100)
                        plt.title('I frame ' + str(frame))
                        plt.subplot(1, 3, 2)
                        plt.imshow(np.transpose(qdata[:, :, frame]), cmap=plt.cm.viridis, aspect='auto', vmin=-100,
                                   vmax=100)
                        plt.title('Q frame ' + str(frame))
                        plt.subplot(1, 3, 3)
                        plt.imshow(np.transpose(bdata[:, :, frame]), cmap=plt.cm.gray, aspect='auto', vmin=15, vmax=70)
                        plt.title('sample B frame ' + str(frame))
                        plt.show()

                elif path[-6:-4] == 'rf':   # Extract RF sector data
                    # Read RF data
                    hdr, timestamps, data = rd.read_rf(path)
                    data = np.transpose(data)
                    numframes = 1  # hdr['frames']
                    # convert to B
                    bdata = np.zeros((hdr['lines'], hdr['samples'], hdr['frames']), dtype='float')
                    for frame in range(numframes):
                        bdata[:, :, frame] = 20 * np.log10(np.abs(1 + hilbert(data[:, :, frame])))

                    # display images
                    for frame in range(numframes):
                        plt.figure(figsize=(10, 5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(np.transpose(data[:, :, frame]), cmap=plt.cm.plasma, aspect='auto', vmin=-1000,
                                   vmax=1000)
                        plt.title('RF frame ' + str(frame))
                        plt.subplot(1, 2, 2)
                        plt.imshow(np.transpose(bdata[:, :, frame]), cmap=plt.cm.gray, aspect='auto', vmin=15, vmax=70)
                        plt.title('sample B frame ' + str(frame))
                        plt.show()
                plt.pause(5)
            #return  path = path + '.iq'
            bModeImage3D = data
            iqImage3D = data

            [numSlices, numALines, numSamples] = data.shape
            iqImageSize = (numSamples, numALines)

            # Now that we know the size of the data, we need to tell the slice widget what size to set
            self.sliceWidgetBMode.setIQSize(numALines, numSamples)
            self.sliceWidgetIQ.setIQSize(numALines, numSamples)


        else:
            # Clears drawn ROI lists when a new IQ file is loaded
            self.manualBModeROIs.clear()
            self.manualIQROIs.clear()

            if self.autoLoadFlag:
                fileNameFullPath = self.iqPath
            else:
                fileNameFullPath, filterReturn = FileDialog.getOpenFileName(self, 'Select IQ file',
                                                                            self.defaultOpenPath, '*.iq')

            if not fileNameFullPath:
                return

            # change cursor here
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Path is cleaned to look "pretty", i.e. switch all backward slashes (\\) to forward slashes (/) for the
            # user to see a nice path
            self._iqFileNamePath = util.cleanPath(fileNameFullPath)

            # grab just the name to use for loading
            self._iqFileName = util.splitext(os.path.basename(self._iqFileNamePath))[0]

            # display file name in GUI
            self.labelCase.setText(f"Case: {self._iqFileName}")

            # extract the 'landmark' number (aka 'view') from the filename and store for future use
            # order reversed for files using _ in the name, example: '230131_PIG_HEART01_06'
            for i in range(len(self._iqFileName)-1,0,-1):
                if self._iqFileName[i] == '_':
                    self.landMarkViewNumber = int(self._iqFileName[(i+1):])
                    break

            # use file names in 'Traced Contour' path to find ED and ES frames for case
            dir = self._iqFileNamePath.split('/IQ')[0] + r'/Traced Contour'
            dirED = dir + r'/EndDiastole'
            dirES = dir + r'/EndSystole'

            if os.path.exists(dirED):
                # Find numbers in traced contour file - ED/ES frame is the last number
                highest_obs = 99
                for file in os.listdir(dirED):
                    if file.lower().find(self._iqFileName.lower()) != -1:
                        s = file.split('.')[0]
                        observer = s.split('_')[-1]
                        try:
                            obs_priority = self.observerPriority.index(observer)
                        except ValueError:
                            continue
                        if obs_priority < highest_obs:
                            highest_obs = obs_priority
                            self.EDFrame = s.split('_')[3]     #re.findall('[0-9]+', s)[2]
                highest_obs = 99
                for file in os.listdir(dirES):
                    if file.lower().find(self._iqFileName.lower()) != -1:
                        s = file.split('.')[0]
                        observer = s.split('_')[-1]
                        try:
                            obs_priority = self.observerPriority.index(observer)
                        except ValueError:
                            continue
                        if obs_priority < highest_obs:
                            highest_obs = obs_priority
                            self.ESFrame = s.split('_')[3]     #re.findall('[0-9]+', s)[2]

                # show frame numbers in GUI
                self.labelEDFrame.setText(f"ED Frame: {self.EDFrame}")
                self.labelESFrame.setText(f"ES Frame: {self.ESFrame}")

            # load the iq data associated with the selected filename
            data = zonare.loadIQ(self._iqFileNamePath)
            cineIQData = data['CineIQ']['B']['data']

            # get number of slices and size of IQ images
            bModeImageSize = (self.bmodeImageHeight,
                              self.bmodeImageWidth)  # TODO handle this better in cooperation with iqBModeConverter file
            [numSlices, numALines, numSamples] = cineIQData.shape
            iqImageSize = (numSamples, numALines)

            # Now that we know the size of the data, we need to tell the slice widget what size to set
            self.sliceWidgetBMode.setIQSize(numALines, numSamples)
            self.sliceWidgetIQ.setIQSize(numALines, numSamples)

            # set progress bar maximum value and reset to zero if necessary
            self.progressBarImageRecon.setMaximum(numSlices)
            self.progressBarImageRecon.setValue(0)

            # initialize data for storing both Bmode and IQ image data
            bModeImage3D = np.zeros((numSlices, bModeImageSize[0], bModeImageSize[1]), dtype=np.uint8)
            iqImage3D = np.zeros((numSlices, iqImageSize[0], iqImageSize[1]), dtype=np.uint8)

            self.bModeImagesTemp.clear()

            # loop through the number of slices and reconstruct one at a time so
            # progress bar can be updated
            for currentSlice in range(0, numSlices):
                # set status message for this image
                self.labelStatus.setText('Reconstructing image: %d' % currentSlice)

                # grab current slice - do separately for now to make easier to debug
                thisSliceData = cineIQData[currentSlice, :, :]

                # get the rectangular IQ images and also make  make the B-mode images
                # TODO allow changing parameters for image recon?
                bModeImage, resolution, self.ptSettings = iqBModeConverter.convertBMode(cineIQData=thisSliceData,
                                                                                        bModeMasterGain=0, tgcParameters=(
                    0, 0, 0, 0, 0, 0, 0, 0),
                                                                                        dynamicRange=(30, 95),
                                                                                        mode='wavelet')

                bModePolarImage = iqBModeConverter.convertBMode(cineIQData=thisSliceData, bModeMasterGain=0,
                                                                tgcParameters=(0, 0, 0, 0, 0, 0, 0, 0),
                                                                dynamicRange=(30, 95), mode='wavelet',
                                                                scanConvert=False)

                # rotate iq image so is vertical, i.e. number of alines is horizontal and number samples vertical
                bModePolarImage = np.rot90(bModePolarImage)

                # flip bmode image vertically and save to be exported for merging
                bModeIm = cv2.flip(bModeImage, 0)
                bModeIm = Image.fromarray(bModeIm)
                self.bModeImagesTemp.append(bModeIm)

                # For IQ image, normalize and convert to uint8 (0 -> 255 range), this would've already happened
                # in the scan conversion for the bmode image
                dynamicRange = (30, 95)
                bModePolarImage = np.clip(bModePolarImage, dynamicRange[0], dynamicRange[1])
                bModePolarImage = \
                    ((bModePolarImage - dynamicRange[0]) / (dynamicRange[1] - dynamicRange[0]) * 255.0).astype(np.uint8)

                # flip bMode image left-right so it's correct orientation
                bModeImage = np.fliplr(bModeImage)

                # copy resulting data into 3D volume objects
                bModeImage3D[currentSlice, :, :] = bModeImage[:, :]
                iqImage3D[currentSlice, :, :] = bModePolarImage[:, :]

                # set progress bar value
                self.progressBarImageRecon.setValue(currentSlice + 1)

        # Since new filename was selected, save the last path used
        self.defaultOpenPath = os.path.dirname(self._iqFileNamePath)

        # Save the settings after getting new path
        self.saveSettings()

        # change cursor here
        QApplication.restoreOverrideCursor()

        # set the image data for the slice widget and update it, also set max of slider based on number of slices
        self.sliceWidgetBMode.imageBMode = bModeImage3D
        self.sliceWidgetIQ.imageIQ = iqImage3D
        self.labelStatus.setText('Image reconstruction complete.')
        self.sliceSlider.setMaximum(numSlices - 1)  # 0-based for slice number slider
        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        # log message for which file
        logger.debug('Chose %s.' % self._iqFileNamePath)

        # temporarily set frame number to 50 for purposes of debugging ROI overlay for
        # iq file 'MF0304POST_3.iq'
        self.sliceSlider.setValue(50)
        self.sliceWidgetBMode.sliceNumber = 50
        self.bModeImages.clear()
        self.bModeImages.append(self.bModeImagesTemp[self.sliceWidgetBMode.sliceNumber])

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        # enable relevant push buttons
        self.pushButtonLoadPoints.setEnabled(True)
        self.pushButtonGetOutsidePoints.setEnabled(True)

    @pyqtSlot()
    def on_pushButtonExportImages_clicked(self):
        print('on_pushButtonExportImages_clicked')

        # grab the image data, both b-mode and iq, overlay any ROIs and export to image files
        if self.sliceWidgetBMode.imageBMode is not None:
            self.sliceWidgetBMode.exportROIImage(self.roiDataList, self.landMarkViewNumber, self.ptSettings)
        if self.sliceWidgetIQ.imageIQ is not None:
            self.sliceWidgetIQ.exportROIImage(self.roiDataList, self.landMarkViewNumber, self.ptSettings)

    @pyqtSlot()
    def on_pushButtonDijkstraTest_clicked(self):

        # Read image and convert to black and white if it is RGB
        debugDir = 'C:\\Users\\16186\\Downloads\\IQ_Files'
        # costImagePath = os.path.join(debugDir, 'costTest.png')
        # start = (0, 0)
        # stop = (0, 255)
        costImagePath = os.path.join(debugDir, 'costTest2.png')
        start = (202, 193)
        stop = (192, 110)
        # costImagePath = os.path.join(debugDir, 'costTest3.png')
        # start = (89, 18)
        # stop = (start[0]+1, start[1])
        # costImagePath = os.path.join(debugDir, 'costTest4.png')
        # start = (10, 7)
        # stop = (start[0]+1, start[1])

        costImage = imageio.imread(costImagePath)
        if costImage.ndim == 3:
            costImage = costImage[:, :, 0]

        print('\n=== Dijkstra Image ===')
        costs, path1 = segmentation.dijkstraImage(costImage, start, stop, neighborhood='east', loop='horizontal')
        # costs, path1 = segmentation.dijkstraImage(costImage, start, stop, neighborhood='8conn')
        # path2 = segmentation.getPathFromTree(parents, (188, 50))  # alternative stop point

        # costImage2 = np.array([[7, 2, 1, 3, 5, 6],
        #                        [7, 2, 6, 7, 7, 7],
        #                        [2, 1, 0, 2, 4, 3],
        #                        [4, 0, 2, 3, 5, 7],
        #                        [4, 0, 5, 7, 8, 9],
        #                        [2, 1, 0, 2, 4, 3],
        #                        [4, 0, 2, 3, 5, 7],
        #                        [4, 0, 5, 7, 8, 9]]).T
        #
        # # costs2, path2 = segmentation.dijkstraImage(costImage2, (3, 1), (4, 1), neighborhood='south', loop='vertical')
        # costs2, path2 = segmentation.dijkstraImage(costImage2, (1, 4), (1, 3), neighborhood='east', loop='horizontal')
        # print(path2)

        # plt.subplot(211)
        plt.imshow(costImage, cmap='gray')
        plt.plot(path1[:, 1], path1[:, 0], '--y', lw=2)
        plt.scatter(start[1], start[0], c='g')
        plt.scatter(stop[1], stop[0], c='r')

        # plt.subplot(212)
        # plt.imshow(costImage, cmap='gray')
        # plt.plot(path2[:, 1], path2[:, 0], '--r', lw=2)
        # plt.scatter(start[1], start[0], c='g')
        # plt.scatter(50, 188, c='r')

        plt.savefig(os.path.join(debugDir, 'costTestResult.png'))
        plt.close()

    @pyqtSlot()
    def on_pushButtonTestSetup_clicked(self):

        # set it to segmentation mode
        self.segmentationModeRadioButton.setChecked(True)
        self.on_segmentationModeRadioButton_toggled(True)

        # assume already loaded the right IQ file (MF03POST_3.iq), but move it to
        # frame 56 as in test frame
        self.sliceSlider.setValue(56)

        # now setup the center and start points
        self.sliceWidgetBMode.segFirstButtonPress = [438, 312]  # for MF03POST_3 center and start
        self.sliceWidgetBMode.segSecondButtonPress = [402, 323]

        # update the figure so it displays these points as the center and start
        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonRunSegmentation_clicked(self):

        # only do it if segmentation center and start have been chosen
        if self.sliceWidgetBMode.segSecondButtonPress is not None:

            # change cursor here
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # get the current slice to pass into the segmentation
            currImage = self.sliceWidgetBMode.imageBMode[self.sliceWidgetBMode.sliceNumber, :, :]
            currImageHeight, currImageWidth = currImage.shape

            # adjust center and start so they will match the image data as stored and passed into the
            # segmentation algorithm - the image data is flipped in both LR and UD directions
            center = self.sliceWidgetBMode.segFirstButtonPress.copy()
            start = self.sliceWidgetBMode.segSecondButtonPress.copy()
            center[1] = currImageHeight - center[1]
            start[1] = currImageHeight - start[1]
            center[0] = currImageWidth - center[0]
            start[0] = currImageWidth - start[0]

            # run the segmentation algorithm here after entering the center point and
            # first boundary point
            segImage, pathCartesian = segmentation.segmentEndocardium(currImage, center, start,
                                                                      self.progressBarImageRecon, self.labelStatus,
                                                                      constants.debugImages, constants.debugDir)

            # display the contour on the image, but undo the conversion process above first
            numPts = pathCartesian.shape[0]
            newPathCartesian = np.zeros(pathCartesian.shape)
            for ii in range(0, numPts):
                newPathCartesian[ii, 0] = currImageWidth - pathCartesian[ii, 0]
                newPathCartesian[ii, 1] = currImageHeight - pathCartesian[ii, 1]
            self.sliceWidgetBMode.endoCardialContour = newPathCartesian
            self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                               self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

            # change cursor back here
            QApplication.restoreOverrideCursor()

        else:
            QMessageBox.warning(self, 'Unable to run segmentation.',
                                'You must first choose the center and start points before running segmentation!',
                                QMessageBox.Ok)

    @pyqtSlot()
    def on_pushButtonNewContour_clicked(self):
        """draw new channel"""

        # check for bmode outside points bspline
        # if none exists, draw expert traced contour
        if len(self.sliceWidgetBMode.bModeOutsidePoints) == 0:
            fileNameFullPathList, filterReturn = FileDialog.getOpenFileNames(self,
                                                                             'Select Intersection Contour text file',
                                                                             self.defaultOpenPath, '*.nrrd')
            if not fileNameFullPathList:
                return

            # change cursor here
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # clear all contours
            self.sliceWidgetIQ.allLoadedContours.clear()
            self.sliceWidgetBMode.allLoadedContours.clear()

            self.sliceWidgetIQ.channelContours.clear()
            self.sliceWidgetBMode.channelContours.clear()

            # get dimensions of current IQ image
            currImageIQ = self.sliceWidgetIQ.imageIQ[self.sliceWidgetIQ.sliceNumber, :, :]
            currImageHeightIQ, currImageWidthIQ = currImageIQ.shape

            # get number of filenames read in
            for currFileName in fileNameFullPathList:

                nrrdTraced = currFileName
                data, _ = nrrd.read(nrrdTraced)

                # Skip background
                layers = []
                for layer_index in range(1, data.shape[0]):
                    layer = data[layer_index, :, :, :]

                    # Read layer and convert to binary
                    convertedLayer = convertImageFromArray(layer)
                    binaryLayer = changeToBinaryMask(convertedLayer)
                    layers.append(np.array(binaryLayer))

                leftEndo, leftEpi, right = tuple(layers[:3])

                # Make left myocardium mask = leftEndo (flooded area)  - leftEpi (flooded area)
                # Left ventricle is filled left Epi
                filled_leftEndo = floodFillContour(leftEndo)
                filled_leftEpi = floodFillContour(leftEpi)

                leftVentricle = copy.deepcopy(filled_leftEndo)
                leftVentricle[np.where((leftVentricle == [0, 255, 255]).all(axis=2))] = [0, 0, 0]
                _, leftVentricle = cv2.threshold(leftVentricle, 127, 255, cv2.THRESH_BINARY)

                leftMyocardium = cv2.bitwise_or(filled_leftEndo, filled_leftEpi)
                leftMyocardium[np.where((leftMyocardium == [255, 0, 0]).all(axis=2))] = [255, 255, 255]
                _, leftMyocardium = cv2.threshold(leftMyocardium, 127, 255, cv2.THRESH_BINARY)

                # right ventricle contour does not always close completely, so slightly dilate
                kernel = np.ones((7, 7), 'uint8')
                right = cv2.dilate(right, kernel, iterations=1)

                # Make a mask for right ventricle by filling in contour of both right side and left endo,
                # then subtracting it from left endo mask
                combined_contour = cv2.bitwise_or(leftEpi, right)
                filled_combined_contour = floodFillContour(combined_contour)
                rightVentricle = cv2.bitwise_xor(filled_combined_contour, filled_leftEpi)

                rightVentricle[np.where((rightVentricle == [255, 0, 0]).all(axis=2))] = [255, 255, 255]
                _, rightVentricle = cv2.threshold(rightVentricle, 127, 255, cv2.THRESH_BINARY)

                # Flip them before transform them to polar domain
                leftMyocardium = np.flipud(leftMyocardium)
                rightVentricle = np.flipud(rightVentricle)

                # create dilated filled contour
                kernel = np.ones((30, 30), 'uint8')
                rightVentricleDilate = cv2.dilate(rightVentricle, kernel, iterations=1)
                leftMyocardiumDilate = cv2.dilate(leftMyocardium, kernel, iterations=1)

                # combine left myo and right vent into one contour for dilated and original
                originalarray = np.array(leftMyocardium + rightVentricle)
                originalarray = originalarray[:, :, 0]
                dilatedarray = np.array(leftMyocardiumDilate + rightVentricleDilate)
                dilatedarray = dilatedarray[:, :, 0]

                originalarray_channel = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))
                dilatedarray_channel = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))

        # if bspline is loaded, make channel based on fat thickness from MRI
        else:
            # change cursor here
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # get dimensions of current IQ image
            currImageIQ = self.sliceWidgetIQ.imageIQ[self.sliceWidgetIQ.sliceNumber, :, :]
            currImageHeightIQ, currImageWidthIQ = currImageIQ.shape

            # get outside points spline and make this the inside of the contour
            points = np.array(self.sliceWidgetBMode.bModeOutsidePoints)
            points = np.around(points, 0)
            points = points.astype(int)
            xPts = points[:, 1]
            yPts = points[:, 0]

            # fill mask with outside points
            originalarray = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))
            originalarray[xPts, yPts] = 255
            fillcontour = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth, 3))
            fillcontour[:, :, 1] = originalarray[:, :]
            fillcontour = fillcontour.astype('uint8')

            # slightly dilate contour to make sure it's closed
            kernel = np.ones((2, 2))
            fillcontour = cv2.dilate(fillcontour, kernel)

            # fill contour
            originalarray = floodFillContour(fillcontour)[:, :, 0]

            # get fat thickness points and make this the outside of the contour
            dilatedarray = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))
            dilatedarray_channel = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))

            # convert fat thickness from mm to pixels
            conversion = self.bmodeImageDepthMM / (self.bmodeFinalRadius - self.bmodeInitialRadius)
            fatThickness = np.array(self.fatPoints)
            fatThickness /= conversion
            fatThickness = np.around(fatThickness, 0)
            fatThickness = fatThickness.astype(int)

            # convert channel thickness from mm to pixels
            contourThickness = np.around(self.sliceWidgetBMode.ContourThickness / conversion, 0).astype(int)

            xFat = []
            yFat = []

            xChannel = []
            yChannel = []

            # generate normal vector for each outside point, place a point at each location along this vector up until
            # fat thickness
            for i in range(len(fatThickness)):
                for j in range(contourThickness):
                    if j in range(fatThickness[i]):
                        crossX, crossY = fatMapping.getNormalVectors(self.sliceWidgetBMode.bModeOutsidePoints,
                                                                     length=j)
                        xFat.append(self.sliceWidgetBMode.bModeOutsidePoints[i, 1] + crossX[0, i])
                        yFat.append(self.sliceWidgetBMode.bModeOutsidePoints[i, 0] - crossY[0, i])

                    else:
                        crossX, crossY = fatMapping.getNormalVectors(self.sliceWidgetBMode.bModeOutsidePoints,
                                                                     length=j)
                        xChannel.append(self.sliceWidgetBMode.bModeOutsidePoints[i, 1] + crossX[0, i])
                        yChannel.append(self.sliceWidgetBMode.bModeOutsidePoints[i, 0] - crossY[0, i])

            xFat = np.around(np.array(xFat), 0)
            xFat = xFat.astype(int)
            yFat = np.around(np.array(yFat), 0)
            yFat = yFat.astype(int)

            xChannel = np.around(np.array(xChannel), 0)
            xChannel = xChannel.astype(int)
            yChannel = np.around(np.array(yChannel), 0)
            yChannel = yChannel.astype(int)

            # fill mask with fat thickness points
            dilatedarray[xFat, yFat] = 255
            dilatedarray = dilatedarray.astype('uint8')
            originalarray = originalarray.astype('uint8')
            dilatedarray = cv2.bitwise_or(dilatedarray, originalarray)

            filldilated = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth, 3))
            filldilated[:, :, 1] = dilatedarray[:, :]
            filldilated = filldilated.astype('uint8')
            dilatedarray = floodFillContour(filldilated)[:, :, 0]

            dilatedarray_channel[xChannel, yChannel] = 255
            dilatedarray_channel = dilatedarray_channel.astype('uint8')
            originalarray = originalarray.astype('uint8')
            dilatedarray_channel = cv2.bitwise_or(dilatedarray_channel, originalarray)

            filldilated_channel = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth, 3))
            filldilated_channel[:, :, 1] = dilatedarray_channel[:, :]
            filldilated_channel = filldilated_channel.astype('uint8')
            dilatedarray_channel = floodFillContour(filldilated_channel)[:, :, 0]

            # # get original and dilated array for non-fat points
            # # original array will be the same, outside points of bspline
            # originalarray_channel = np.zeros((600, 800))
            # originalarray_channel[:, :] = originalarray[:, :]
            #
            # kernel = np.ones((self.sliceWidgetBMode.ContourThickness, self.sliceWidgetBMode.ContourThickness), 'uint8')
            # dilatedarray_channel = cv2.dilate(originalarray_channel, kernel, iterations=1)

        dilatedarray_channel_orig = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))
        # remove overlap between contours
        dilatedarray[:, :] = dilatedarray[:, :] - originalarray[:, :]
        dilatedarray_channel[:, :] = dilatedarray_channel[:, :] - originalarray[:, :]

        # get rid of negative values from subtraction
        dilatedarray[dilatedarray < 0] = 0
        dilatedarray_channel[dilatedarray_channel < 0] = 0

        dilatedarray = dilatedarray.astype('uint8')
        dilatedarray_channel = dilatedarray_channel.astype('uint8')

        # convert to polar and orientate correctly for iq image
        dilatedarray_polar = convert_mask_from_cartesian_to_polar(dilatedarray)
        dilatedarray_polar = np.flipud(dilatedarray_polar)

        dilatedarray_polar_channel = convert_mask_from_cartesian_to_polar(dilatedarray_channel)
        dilatedarray_polar_channel = np.flipud(dilatedarray_polar_channel)

        dilatedarray_polar_channel_orig = convert_mask_from_cartesian_to_polar(dilatedarray_channel_orig)
        self.channel[:, :] = dilatedarray_polar_channel_orig[:, :]

        # temp variable of nonfat channel
        dilatedarray_polar_channel1 = np.zeros((self.iqImageHeight, self.iqImageWidth))
        dilatedarray_polar_channel1[:, :] = dilatedarray_polar_channel

        # remove any overlapping points after conversion
        dilatedarray_polar_channel = dilatedarray_polar_channel.astype('uint8')
        dilatedarray_polar_channel1 = dilatedarray_polar_channel1.astype('uint8')
        dilatedarray_polar = dilatedarray_polar.astype('uint8')
        dilatedarray_polar_channel = cv2.bitwise_xor(dilatedarray_polar_channel, dilatedarray_polar)
        dilatedarray_polar_channel = cv2.bitwise_and(dilatedarray_polar_channel, dilatedarray_polar_channel1)

        alineStart = []
        alineStop = []
        sampleStart = []
        sampleStop = []

        alineStart_channel = []
        alineStop_channel = []
        sampleStart_channel = []
        sampleStop_channel = []

        # save mask of fat and nonfat contours for use in drawing ROIs
        self.fatMask[:, :] = dilatedarray_polar[:, :]
        self.nonFatMask[:, :] = dilatedarray_polar_channel[:, :]

        dilatedarray_channel = dilatedarray_channel - dilatedarray
        dilatedarray_channel[dilatedarray_channel < 0] = 0
        dilatedarray[dilatedarray==1] = 0
        dilatedarray_channel[dilatedarray_channel==1] = 0

        dilatedarray_polar = np.ma.masked_where(dilatedarray_polar == 0, dilatedarray_polar)
        dilatedarray_polar_channel = np.ma.masked_where(dilatedarray_polar_channel == 0, dilatedarray_polar_channel)
        dilatedarray = np.ma.masked_where(dilatedarray == 0, dilatedarray)
        dilatedarray_channel = np.ma.masked_where(dilatedarray_channel == 0, dilatedarray_channel)

        self.sliceWidgetIQ.fatMaskIQ = dilatedarray_polar
        self.sliceWidgetIQ.nonFatMaskIQ = dilatedarray_polar_channel
        self.sliceWidgetBMode.fatMaskBMode = dilatedarray
        self.sliceWidgetBMode.nonFatMaskBMode = dilatedarray_channel

        # # create arrays of stop and start points for intersecting alines
        # for i in range(0, currImageWidthIQ, 1):
        #     j = 0
        #     while j < currImageHeightIQ:
        #         if dilatedarray_polar[j][i] != 0:
        #             alineStart.append(i)
        #             alineStop.append(i)
        #             sampleStart.append(j)
        #             for x in range(j + 1, currImageHeightIQ):
        #                 if dilatedarray_polar[x][i] == 0:
        #                     sampleStop.append(x)
        #                     j = x + 1
        #                     break
        #                 elif x == currImageHeightIQ:
        #                     sampleStop.append(currImageHeightIQ)
        #                     j = x + 1
        #         else:
        #             j += 1
        #
        # if len(self.sliceWidgetBMode.bModeFatPoints) != 0:
        #     # create arrays of stop and start points for channel
        #     for i in range(0, currImageWidthIQ, 1):
        #         j = 0
        #         while j < currImageHeightIQ:
        #             if dilatedarray_polar_channel[j][i] != 0:
        #                 alineStart_channel.append(i)
        #                 alineStop_channel.append(i)
        #                 sampleStart_channel.append(j)
        #                 for x in range(j + 1, currImageHeightIQ):
        #                     if dilatedarray_polar_channel[x][i] == 0:
        #                         sampleStop_channel.append(x)
        #                         j = x + 1
        #
        #                         if np.any(dilatedarray_polar[j:x + 1, i]):
        #                             alineStart_channel.pop()
        #                             alineStop_channel.pop()
        #                             sampleStart_channel.pop()
        #                             sampleStop_channel.pop()
        #                         break
        #                     elif x == currImageHeightIQ:
        #                         sampleStop_channel.append(currImageHeightIQ)
        #                         j = x + 1
        #             else:
        #                 j += 1
        #
        # # convert IQ points to BMode and append both to respective contour
        # for i in range(len(alineStart)):
        #     alines = [(alineStart[i], sampleStart[i]), (alineStop[i], sampleStop[i])]
        #
        #     bModeAlineStart = alineStart[i]
        #     bModeAlineStop = alineStop[i]
        #     bModeSampleStart = currImageHeightIQ - sampleStart[i]
        #     bModeSampleStop = currImageHeightIQ - sampleStop[i]
        #
        #     poly = [[bModeAlineStart, bModeSampleStart], [bModeAlineStop, bModeSampleStop]]
        #
        #     # creates storage lists for converted points
        #     bModePoints = []
        #     bModeContourPoints = []
        #
        #     # Iterates through the list of points and puts them in
        #     for j in range(len(poly)):
        #         x1 = poly[j][1]
        #         y1 = poly[j][0]
        #         bModePoints.append([x1, y1])
        #
        #         # getting BMode points from IQ points and adding to a list for each contour
        #         (x, y) = polarTransform.getCartesianPointsImage(bModePoints[j], self.ptSettings)
        #         bModeContourPoints.append([x, y])
        #
        #     self.sliceWidgetIQ.allLoadedContours.append(alines)
        #     self.sliceWidgetBMode.allLoadedContours.append(bModeContourPoints)
        #
        # if len(self.sliceWidgetBMode.bModeFatPoints) != 0:
        #     for i in range(len(alineStart_channel)):
        #         channel = [(alineStart_channel[i], sampleStart_channel[i]),
        #                    (alineStop_channel[i], sampleStop_channel[i])]
        #
        #         bModeAlineStart_channel = alineStart_channel[i]
        #         bModeAlineStop_channel = alineStop_channel[i]
        #         bModeSampleStart_channel = currImageHeightIQ - sampleStart_channel[i]
        #         bModeSampleStop_channel = currImageHeightIQ - sampleStop_channel[i]
        #
        #         poly_channel = [[bModeAlineStart_channel, bModeSampleStart_channel],
        #                         [bModeAlineStop_channel, bModeSampleStop_channel]]
        #
        #         bModeChannelPoints = []
        #         bModeContourChannelPoints = []
        #         for j in range(len(poly_channel)):
        #             x1 = poly_channel[j][1]
        #             y1 = poly_channel[j][0]
        #             bModeChannelPoints.append([x1, y1])
        #
        #             # getting BMode points from IQ points and adding to a list for each contour
        #             (x, y) = polarTransform.getCartesianPointsImage(bModeChannelPoints[j], self.ptSettings)
        #             bModeContourChannelPoints.append([x, y])
        #
        #         self.sliceWidgetIQ.channelContours.append(channel)
        #         self.sliceWidgetBMode.channelContours.append(bModeContourChannelPoints)

        # if 1 to save contour to specified location
        if 1:
            path = fr'{self.SAVEPATH}\contours\thickness_{self.sliceWidgetBMode.ContourThickness}'

            if not os.path.exists(path):
                os.makedirs(path)

            frameFlag = self.mriWindow.get_frame_flag()
            path = fr'{self.SAVEPATH}\contours\thickness_{self.sliceWidgetBMode.ContourThickness}\{self._iqFileName}_E{frameFlag}'

            if not os.path.exists(path):
                os.makedirs(path)

            np.save(path + r'\bmodeFat.npy', np.array(self.sliceWidgetBMode.fatMaskBMode))
            np.save(path + r'\iqFat.npy', np.array(self.sliceWidgetIQ.fatMaskIQ))
            np.save(path + r'\bmodeChannel.npy', np.array(self.sliceWidgetBMode.nonFatMaskBMode))
            np.save(path + r'\iqChannel.npy', np.array(self.sliceWidgetIQ.nonFatMaskIQ))
            np.save(path + r'\fatMask.npy', self.fatMask)
            np.save(path + r'\nonFatMask.npy', self.nonFatMask)

        self.channelBMode = cv2.bitwise_or(self.sliceWidgetBMode.fatMaskBMode,
                                           self.sliceWidgetBMode.nonFatMaskBMode)

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        # enable draw ROIs button
        self.pushButtonDrawChannelROIs.setEnabled(True)

    # change cursor here
    QApplication.restoreOverrideCursor()

    @pyqtSlot()
    def on_pushButtonLoadContour_clicked(self):
        """Load channel if one exists for currently loaded case"""
        frameFlag = self.mriWindow.get_frame_flag()
        if frameFlag:
            path = fr'{self.SAVEPATH}\contours\thickness_{self.sliceWidgetBMode.ContourThickness}\{self._iqFileName}_E{frameFlag}'

        else:
            return

        if os.path.exists(path):
            self.sliceWidgetBMode.fatMaskBMode = np.load(path + r'\bmodeFat.npy')
            self.sliceWidgetIQ.fatMaskIQ = np.load(path + r'\iqFat.npy')
            self.sliceWidgetBMode.nonFatMaskBMode = np.load(path + r'\bmodeChannel.npy')
            self.sliceWidgetIQ.nonFatMaskIQ = np.load(path + r'\iqChannel.npy')

            self.sliceWidgetBMode.fatMaskBMode = np.ma.masked_where(self.sliceWidgetBMode.fatMaskBMode == 0, self.sliceWidgetBMode.fatMaskBMode)
            self.sliceWidgetIQ.fatMaskIQ = np.ma.masked_where(self.sliceWidgetIQ.fatMaskIQ == 0, self.sliceWidgetIQ.fatMaskIQ)
            self.sliceWidgetBMode.nonFatMaskBMode = np.ma.masked_where(self.sliceWidgetBMode.nonFatMaskBMode == 0, self.sliceWidgetBMode.nonFatMaskBMode)
            self.sliceWidgetIQ.nonFatMaskIQ = np.ma.masked_where(self.sliceWidgetIQ.nonFatMaskIQ == 0, self.sliceWidgetIQ.nonFatMaskIQ)

            self.fatMask = np.load(path + r'\fatMask.npy')
            self.nonFatMask = np.load(path + r'\nonFatMask.npy')
            self.channel = cv2.bitwise_or(self.fatMask, self.nonFatMask)
            self.channelBMode = cv2.bitwise_or(self.sliceWidgetBMode.fatMaskBMode,
                                                self.sliceWidgetBMode.nonFatMaskBMode)

            self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                               self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
            self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                            self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

            # enable draw roi button
            self.pushButtonDrawChannelROIs.setEnabled(True)

    @pyqtSlot()
    def on_pushButtonExportROI_clicked(self):
        """Save ROI coordinated and tissue type to .txt file"""
        frameFlag = self.mriWindow.get_frame_flag()

        if len(self.manualIQROIs) == 0 and len(self.contourSlicesIQ) == 0:
            QMessageBox.warning(self, "Can't Save ROIs", "No ROIs loaded")
            return

        # if self.mriWindow.get_frame_flag():
        # if 0:
        #     fileNameFullPathList = FileDialog.getSaveFileName(self, 'Select location to store ROI',
        #                                                       self.defaultOpenPath[:-2] + self._iqFileName + '_' +
        #                                                       self.ROIFatFlag + '_E' + self.mriWindow.get_frame_flag(), '*.txt')
        # else:
        #     fileNameFullPathList = FileDialog.getSaveFileName(self, 'Select location to store ROI',
        #                                                       self.defaultOpenPath[:-2] + self._iqFileName, '*.txt')
        # data = util.cleanPath(fileNameFullPathList[0])

        # hard code location to save data
        # make sure path exists, create if not

        if self.ROIFatFlag == "Channel":
            path = fr'{self.SAVEPATH}\ROIs\{self.ROIFatFlag.lower()}\{self.sliceWidgetBMode.ROIChannelNum}\{self._iqFileName}_{self.ROIFatFlag}_E{frameFlag}'
            maskPath = fr'{self.SAVEPATH}\ROI_masks\{self.ROIFatFlag.lower()}\{self.sliceWidgetBMode.ROIChannelNum}\{self._iqFileName}_{self.ROIFatFlag}_E{frameFlag}'
            maskPathBMode = fr'{self.SAVEPATH}\ROI_masks_bmode\{self.ROIFatFlag.lower()}\{self.sliceWidgetBMode.ROIChannelNum}\{self._iqFileName}_{self.ROIFatFlag}_E{frameFlag}'
        else:
            path = fr'{self.SAVEPATH}\ROIs\{self.ROIFatFlag.lower()}\{self.sliceWidgetBMode.ROIsizevert}x{self.sliceWidgetBMode.ROIsizehoriz}'

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(maskPath):
            os.makedirs(maskPath)
        if not os.path.exists(maskPathBMode):
            os.makedirs(maskPathBMode)

        if self.ROIFatFlag != "Channel":
            data = fr'{self.SAVEPATH}\ROIs\{self.ROIFatFlag.lower()}\{self.sliceWidgetBMode.ROIsizevert}x{self.sliceWidgetBMode.ROIsizehoriz}\{self._iqFileName}_{self.ROIFatFlag}_E{frameFlag}.txt'

        # set file name according to IQ file loaded into GUI
        fileName = self._iqFileName
        # data = os.path.join(data, fileName + '.txt')
        # if not data:
        #     return
        #
        # filename = data

        if self.ROIFatFlag != 'Channel':
            f = open(data, "w")
            f.write(str(self._iqFileName) + '\n')

            # set up labels for each data value that will be written out
            f.write('Landmark,' + ' ' + 'Frame#,' + ' ' + 'ROI#,' + ' ' + 'TissueType,' + ' ' + 'AlineStart,' + ' '
                    + 'AlineStop,' + ' ' + 'SampleStart,' + ' ' + 'SampleStop' + '\n')

        if len(self.sliceWidgetIQ.iqROIPointsList) != 0:  # if there are ROIs in the IQ image
            for i in range(len(self.sliceWidgetIQ.iqROIPointsList)):  # for each ROI displayed
                if self.ROIFatFlag == 'Channel':

                    data = fr'{path}\{fileName}_{self.ROIFatFlag}_E{frameFlag}_{i}.txt'

                    if not data:
                        return

                    f = open(data, "w")
                    f.write(str(self._iqFileName) + '\n')

                    # set up labels for each data value that will be written out
                    f.write(
                        'Landmark,' + ' ' + 'Frame#,' + ' ' + 'ROI#,' + ' ' + 'TissueType,' + ' ' + 'AlineStart,' + ' '
                        + 'AlineStop,' + ' ' + 'SampleStart,' + ' ' + 'SampleStop' + '\n')

                # gets first and second clicked point for ROIs in IQ image
                (alineStop, sampleStart) = (
                self.sliceWidgetIQ.iqROIPointsList[i][0][0], self.sliceWidgetIQ.iqROIPointsList[i][2][1])
                (alineStart, sampleStop) = (
                self.sliceWidgetIQ.iqROIPointsList[i][2][0], self.sliceWidgetIQ.iqROIPointsList[i][0][1])

                # actual data needs to be adjusted according to the number of Alines in the BMode image and
                # number of samples in the IQ image
                newAlineStart = alineStart
                newAlineStop = alineStop
                newSampleStart = sampleStart
                newSampleStop = sampleStop

                # # set tissue type to write based on which tissue type is selected
                # if self.fatRadioButton.isChecked():
                #     self.tissueType = 1
                # elif self.myocardiumRadioButton.isChecked():
                #     self.tissueType = 2
                # elif self.bloodRadioButton.isChecked():
                #     self.tissueType = 3
                # elif self.nonFatRadioButton.isChecked():
                #     self.tissueType = 4
                self.tissueType = self.tissueTypeList[i]

                # write everything out to a text file with some features for 'beautification'
                f.write(str(self.landMarkViewNumber) + ', ' + str(self.sliceWidgetBMode.sliceNumber) + ', ' +
                        str(i + 1) + ', ' + str(int(self.tissueType)) + ', ' + str(int(newAlineStart)) + ', ' +
                        str(int(newAlineStop)) + ', ' + str(int(newSampleStart)) + ', ' + str(
                    int(newSampleStop)) + '\n')

            print(f'Saved to {data}\n')
            self.savedConfirmation.setText(f'ROIs saved to {data}')
            f.close()

        # if there are mask ROIs
        if len(self.contourSlicesIQ) != 0:
            for i in range(len(self.contourSlicesIQ)):
                tissueType = self.tissueTypeList[i]
                sliceNum = self.countourSliceFrameNums[i]
                mask = fr'{maskPath}\{fileName}_{self.ROIFatFlag}_E{frameFlag}_{sliceNum}_{tissueType}_{i}.npy'
                maskBMode = fr'{maskPathBMode}\{fileName}_{self.ROIFatFlag}_E{frameFlag}_{sliceNum}_{tissueType}_{i}.npy'

                np.save(mask, self.contourSlicesIQ[i])
                np.save(maskBMode, self.contourSlicesBMode[i])

            print(f'Saved to {mask}\n')

    @pyqtSlot()
    def on_pushButtonExportFatNonFatData_clicked(self):
        # if self.RFRadioButton.isChecked():
        #     fileNameFullPathList = FileDialog.getSaveFileName(self, 'Select location to store fat Alines',
        #                                                       self.defaultOpenPath[:-2] + self._iqFileName + '_Fat_E'
        #                                                        + self.mriWindow.get_frame_flag(), '*.npy')
        # elif self.IQRadioButton.isChecked():
        #     fileNameFullPathList = FileDialog.getSaveFileName(self, 'Select location to store fat Alines',
        #                                                       self.defaultOpenPath[:-2] + self._iqFileName + '_Fat_E'
        #                                                        + self.mriWindow.get_frame_flag(), '*.npy')
        # else:
        #     return
        frameFlag = self.mriWindow.frameFlag

        fileNameFullPathList = fr'{self.SAVEPATH}\data\IQ\fat\{self._iqFileName}_Fat_E{frameFlag}.npy'

        # check that IQ data is loaded
        if self.sliceWidgetBMode.imageBMode is not None:

            # load current iq data
            iqdata = zonare.loadIQ(self._iqFileNamePath)
            cineIQData = iqdata['CineIQ']['B']['data']

            iqmatrix = np.zeros((80, self.iqImageWidth, self.iqImageHeight), dtype=complex)

            # loop thru contour
            for i in range(len(self.sliceWidgetIQ.allLoadedContours)):
                # get start and stop points for alines
                alinestart = self.sliceWidgetIQ.allLoadedContours[i][0][0]

                samplestart = self.sliceWidgetIQ.allLoadedContours[i][0][1]
                samplestop = self.sliceWidgetIQ.allLoadedContours[i][1][1]

                # get iq data for the current aline snippet
                line = cineIQData[self.sliceWidgetBMode.sliceNumber][alinestart][samplestart:samplestop]

                # add line of iq data to empty matrix
                iqmatrix[self.sliceWidgetBMode.sliceNumber][alinestart][samplestart:samplestop] = line

            if self.RFRadioButton.isChecked():
                # save data to chosen path
                rfmatrix = IQ2RFconverter.IQ2RF(iqmatrix)
                np.save(fileNameFullPathList, rfmatrix)
                print(f'Saved to {fileNameFullPathList}')

            elif self.IQRadioButton.isChecked():
                # save iq data as an array to chosen path
                np.save(fileNameFullPathList, iqmatrix)
                print(f'Saved to {fileNameFullPathList}')

            # save non-fat contour as well
            if len(self.sliceWidgetBMode.channelContours) != 0:
                # if self.RFRadioButton.isChecked():
                #     fileNameFullPathList = FileDialog.getSaveFileName(self, 'Select location to store nonfat Alines',
                #                                                       self.defaultOpenPath[
                #                                                       :-2] + self._iqFileName + '_Nonfat_E'
                #                                                       + self.mriWindow.get_frame_flag(), '*.npy')
                # elif self.IQRadioButton.isChecked():
                #     fileNameFullPathList = FileDialog.getSaveFileName(self, 'Select location to store nonfat Alines',
                #                                                       self.defaultOpenPath[
                #                                                       :-2] + self._iqFileName + '_Nonfat_E'
                #                                                       + self.mriWindow.get_frame_flag(), '*.npy')
                # else:
                #     return

                fileNameFullPathList = fr'{self.SAVEPATH}\data\IQ\nonfat\{self._iqFileName}_Nonfat_E{frameFlag}.npy'

                iqmatrix_channel = np.zeros((80, self.iqImageWidth, self.iqImageHeight), dtype=complex)

                # loop thru contour
                for i in range(len(self.sliceWidgetIQ.channelContours)):
                    # get start and stop points for alines
                    alinestart_channel = self.sliceWidgetIQ.channelContours[i][0][0]

                    samplestart_channel = self.sliceWidgetIQ.channelContours[i][0][1]
                    samplestop_channel = self.sliceWidgetIQ.channelContours[i][1][1]

                    # get iq data for the current aline snippet
                    line = cineIQData[self.sliceWidgetBMode.sliceNumber][alinestart_channel][
                           samplestart_channel:samplestop_channel]

                    # add line of iq data to empty matrix
                    iqmatrix_channel[self.sliceWidgetIQ.sliceNumber][alinestart_channel][
                    samplestart_channel:samplestop_channel] = line

                if self.RFRadioButton.isChecked():
                    # save data to chosen path
                    rfmatrix_channel = IQ2RFconverter.IQ2RF(iqmatrix_channel)
                    np.save(fileNameFullPathList, rfmatrix_channel)
                    print(f'Saved to {fileNameFullPathList}')

                elif self.IQRadioButton.isChecked():
                    # save iq data as an array to chosen path
                    np.save(fileNameFullPathList, iqmatrix_channel)
                    print(f'Saved to {fileNameFullPathList}')

    @pyqtSlot()
    def on_pushButtonExportChannel_clicked(self):
        """save IQ data within drawn channel as .npy file"""
        frameFlag = self.mriWindow.get_frame_flag()

        # save to this path
        path = fr'{self.SAVEPATH}\data\IQ\channel\channelContours'

        if not os.path.exists(path):
            os.makedirs(path)

        # check that IQ data is loaded
        if self.sliceWidgetBMode.imageBMode is not None:

            # load current iq data
            iqdata = zonare.loadIQ(self._iqFileNamePath)
            cineIQData = iqdata['CineIQ']['B']['data']

            fileNameFullPathList = fr'{path}\{self._iqFileName}_Channel_E{frameFlag}.npy'

            channel = self.channel
            channel[channel == 255] = 1

            iqmatrix = cineIQData[self.sliceWidgetBMode.sliceNumber] * channel.T

            if self.RFRadioButton.isChecked():
                # save data to chosen path
                rfmatrix = IQ2RFconverter.IQ2RF(iqmatrix)
                np.save(fileNameFullPathList, rfmatrix)
                print(f'Saved to {fileNameFullPathList}')

            elif self.IQRadioButton.isChecked():
                # save iq data as an array to chosen path
                np.save(fileNameFullPathList, iqmatrix)
                print(f'Saved to {fileNameFullPathList}')
                self.savedConfirmation.setText(f'Channel data saved to {fileNameFullPathList}')

    @pyqtSlot()
    def on_pushButtonClearContours_clicked(self):
        # clear all drawn contours

        # don't load contour if images aren't there
        if self.sliceWidgetBMode.imageBMode is not None:

            # don't load contour if contour/s aren't there
            if self.sliceWidgetBMode.fatMaskBMode is not None:
                # clear lists that contain contours in IQ and BMode images
                self.sliceWidgetIQ.allLoadedContours.clear()
                self.sliceWidgetBMode.allLoadedContours.clear()

                self.sliceWidgetBMode.fatMaskBMode = None
                self.sliceWidgetBMode.nonFatMaskBMode = None
                self.sliceWidgetIQ.fatMaskIQ = None
                self.sliceWidgetIQ.nonFatMaskIQ = None

                self.sliceWidgetIQ.channelContours.clear()
                self.sliceWidgetBMode.channelContours.clear()

                self.sliceWidgetBMode.channelSegments.clear()
                self.sliceWidgetIQ.channelSegments.clear()

                # update both IQ and BMode images with cleared lists to "clear" contours from the images
                self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                   self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
                self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
                print('Cleared')

        #     else:
        #         QMessageBox.warning(self, 'Unable to clear contours.', 'Contour/s must be loaded first',
        #                                                                 QMessageBox.Ok)
        # else:
        #     QMessageBox.warning(self, 'Cannot clear contours now', 'Must load IQ file first',
        #                                                             QMessageBox.Ok)

    @pyqtSlot()
    def on_pushButtonClearROIs_clicked(self):
        # clear all ROIs draw on image
        # clear lists that contain contours in IQ and BMode images
        self.manualBModeROIs.clear()
        self.manualIQROIs.clear()

        self.contourSlicesIQ.clear()
        self.sliceWidgetIQ.IQROIMasks.clear()
        self.sliceWidgetBMode.BModeROIMasks.clear()

        self.tissueTypeList.clear()
        self.sliceWidgetBMode.tissueTypeList.clear()
        self.sliceWidgetIQ.tissueTypeList.clear()

        # update both IQ and BMode images with cleared lists to "clear" ROIs from the images
        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        print('Cleared')

        #     else:
        #         QMessageBox.warning(self, 'Unable to clear ROIs.', 'ROIs must be placed first',
        #                                                                 QMessageBox.Ok)
        # else:
        #     QMessageBox.warning(self, 'Cannot clear ROIs now', 'Must load IQ file first',
        #                                                             QMessageBox.Ok)

    @pyqtSlot()
    def on_pushButtonClearAll_clicked(self):
        # clear anything that may be drawn on bmode or iq image
        self.sliceWidgetBMode.firstPointBModeSpline.clear()
        self.sliceWidgetBMode.bModeFatMap = list(self.sliceWidgetBMode.bModeFatMap)
        self.sliceWidgetBMode.bModeFatMap.clear()
        self.predictedLabels = list(self.predictedLabels)
        self.predictedLabels.clear()
        MainWindow.on_pushButtonClearContours_clicked(self)
        MainWindow.on_pushButtonClearROIs_clicked(self)
        MainWindow.on_pushButtonClearContourMorph_clicked(self)
        MainWindow.on_pushButtonClearMorphPoints_clicked(self)
        MainWindow.on_pushButtonClearFatPoints_clicked(self)
        MainWindow.on_pushButtonClearOutsidePoints_clicked(self)

        self.sliceWidgetBMode.removeOverlayImage()

    @pyqtSlot(int)
    def on_ROIsizevert_valueChanged(self, value):
        # change height of fat only/nonfat only ROIs
        print('ROI size vertical: %d' % value)

        self.sliceWidgetIQ.ROIsizevert = value
        self.sliceWidgetBMode.ROIsizevert = value

    @pyqtSlot(int)
    def on_ROIsizehoriz_valueChanged(self, value):
        # change width of fat only/nonfat only ROIs
        print('ROI size horizontal: %d' % value)

        self.sliceWidgetIQ.ROIsizehoriz = value
        self.sliceWidgetBMode.ROIsizehoriz = value

    @pyqtSlot(int)
    def on_ROIThresh_valueChanged(self, value):
        # changes % of fat an ROI must contain to be labeled a "fat" ROI
        # not currently used
        print('ROI Threshold: %d' % value)

        self.ROIThreshPercent = value

    @pyqtSlot(int)
    def on_ROIChannelNum_valueChanged(self, value):
        # change number of ~even sized ROIs to be drawn in channel
        print('Number of Channel ROIs: %d' % value)

        self.sliceWidgetIQ.ROIChannelNum = value
        self.sliceWidgetBMode.ROIChannelNum = value

    @pyqtSlot()
    def on_pushButtonDrawNonFatROIs_clicked(self):
        """Draw ROIs only outside fat map obtained from MRI"""

        nonfat = np.zeros((self.iqImageHeight, self.iqImageWidth))
        nonfat[:, :] = self.nonFatMask[:, :]
        nonfat = nonfat.astype('uint8')

        fat = np.zeros((self.iqImageHeight, self.iqImageWidth))
        fat[:, :] = self.fatMask[:, :]
        fat = fat.astype('uint8')

        channel = fat + nonfat

        # set flag for naming the ROI file
        self.ROIFatFlag = 'Nonfat'

        if self.sliceWidgetBMode.imageBMode is not None:

            # Clears drawn ROI lists when the ROI parameters change
            self.manualBModeROIs.clear()
            self.manualIQROIs.clear()

            # check that contour is loaded
            if self.sliceWidgetIQ.channelContours is not None:
                # nonfat tissue type
                self.tissueType = 4

                ret, im_th = cv2.threshold(channel, 90, 255, cv2.THRESH_BINARY)
                ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                width = self.sliceWidgetIQ.ROIsizehoriz
                height = self.sliceWidgetIQ.ROIsizevert
                thresh_perc = self.ROIThreshPercent

                threshold = (width * height * thresh_perc) / 100

                # plt.figure()
                # plt.imshow(im_th)

                for contour in ctrs:
                    x, y, w, h = cv2.boundingRect(contour)

                    j = x
                    channel_width = 0
                    while j + channel_width < x + w:

                        k = y
                        while k < y + h:

                            count = np.count_nonzero(channel[k:k + height, j:j + width])

                            if count > threshold:
                                # plt.scatter(j, k)
                                # plt.scatter(j+width, k+height)
                                #
                                # channel_width = width
                                # if im_th[k+height, j+width] == 255:
                                #     while 1:
                                #         if j+channel_width > 199 or im_th[k+height, j+channel_width] == 0:
                                #             break
                                #         else:
                                #             channel_width += 1
                                self.sliceWidgetIQ.firstPointIQ = [j, k]
                                self.sliceWidgetIQ.secondPointIQ = [j + width, k + height]

                                ROI = [self.sliceWidgetIQ.firstPointIQ, self.sliceWidgetIQ.secondPointIQ]
                                self.manualIQROIs.append(ROI)
                                self.tissueTypeList.append(self.tissueType)

                            k += height

                        j += int(width/2)

                self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                   self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
                self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonDrawFatROIs_clicked(self):
        """Draw ROIs only in areas inside fat map obtained from MRI"""

        fat = np.zeros((self.iqImageHeight, self.iqImageWidth))
        fat[:, :] = self.fatMask[:, :]
        fat = fat.astype('uint8')

        # set flag for naming the ROI file
        self.ROIFatFlag = 'Fat'

        if self.sliceWidgetBMode.imageBMode is not None:

            # Clears drawn ROI lists when the ROI parameters change
            self.manualBModeROIs.clear()
            self.manualIQROIs.clear()

            # check that contour is loaded
            if self.sliceWidgetIQ.allLoadedContours is not None:
                # fat tissue type
                self.tissueType = 1

                ret, im_th = cv2.threshold(fat, 90, 255, cv2.THRESH_BINARY)
                ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                width = self.sliceWidgetIQ.ROIsizehoriz
                height = self.sliceWidgetIQ.ROIsizevert
                thresh_perc = self.ROIThreshPercent

                threshold = (width * height * thresh_perc) / 100

                for contour in ctrs:
                    x, y, w, h = cv2.boundingRect(contour)

                    j = x
                    while j < x + w:

                        k = y
                        while k < y + h:

                            count = np.count_nonzero(fat[k:k + height, j:j + width])

                            if count > threshold:
                                self.sliceWidgetIQ.firstPointIQ = [j, k]
                                self.sliceWidgetIQ.secondPointIQ = [j + width, k + height]

                                ROI = [self.sliceWidgetIQ.firstPointIQ, self.sliceWidgetIQ.secondPointIQ]
                                self.manualIQROIs.append(ROI)
                                self.tissueTypeList.append(self.tissueType)

                            k += height

                        j += width

                self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                   self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
                self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonDrawChannelROIs_clicked(self):
        """Draw ROIs around channel surrounding heart"""

        # Get channel (fat & nonfat combined)
        channel = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))
        channelTemp = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))
        channel[:, :] = self.channelBMode[:, :]
        channel = channel.astype('uint8')
        channelTemp[:, :] = channel[:, :]

        channel = np.flipud(channel)

        self.ROIFatFlag = "Channel"

        # clear all ROIs and previous segments
        self.tissueTypeList.clear()
        self.sliceWidgetBMode.tissueTypeList.clear()
        self.sliceWidgetBMode.tissueTypeList.clear()
        self.manualBModeROIs.clear()
        self.manualIQROIs.clear()
        self.sliceWidgetIQ.channelSegments.clear()
        self.sliceWidgetBMode.channelSegments.clear()
        self.sliceWidgetIQ.iqROIPointsList.clear()
        self.sliceWidgetBMode.bModeROIPointsList.clear()
        self.contourSlicesIQ.clear()
        self.contourSlicesBMode.clear()
        self.countourSliceFrameNums.clear()
        self.predictedLabels.clear()

        self.contourSlicesIQ.clear()
        self.sliceWidgetIQ.IQROIMasks.clear()
        self.sliceWidgetBMode.BModeROIMasks.clear()

        # self.on_pushButtonClearContours_clicked()

        # find centroid of bmode channel mask
        count = (channel == 255).sum()
        center = tuple(np.argwhere(channel == 255).sum(0) / count)
        center = tuple(int(l) for l in center)
        center = tuple(reversed(center))

        # number of contour slices and length of slice (should go out of the image)
        divisions = self.sliceWidgetIQ.ROIChannelNum
        distance = 700

        testIm = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth, 3), np.uint8)
        testIms = []
        testROIs = []

        # create masks of sector from centroid to edge of image - AND with original channel to get section of channel
        for i in range(divisions):
            # Get start and end points that slice the image into pieces
            x = math.sin(2 * i * math.pi / divisions) * distance + center[0]
            y = math.cos(2 * i * math.pi / divisions) * distance + center[1]
            x2 = math.sin(2 * (i + 1) * math.pi / divisions) * distance + center[0]
            y2 = math.cos(2 * (i + 1) * math.pi / divisions) * distance + center[1]
            xMid = math.sin(2 * (i + .5) * math.pi / divisions) * 50 + center[0]
            yMid = math.cos(2 * (i + .5) * math.pi / divisions) * 50 + center[1]

            # get top line, bottom line, and middle line of sector mask
            top = tuple(np.array((x, y), int))
            bottom = tuple(np.array((x2, y2), int))
            midpoint = tuple(np.array((xMid, yMid), int))

            # Draw filled in mask for the specific section of contour
            mask = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth), np.uint8)
            cv2.line(mask, center, top, 255, 1)
            cv2.line(channelTemp, center, top, 255, 3)
            cv2.line(mask, center, bottom, 255, 1)
            cv2.floodFill(mask, None, midpoint, 255)

            # AND mask with contour to get slice
            contourSlice = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth), np.uint8)
            contourSlice[:, :] = channel[:, :]

            # kernel = np.ones((5, 5))
            # mask = cv2.erode(mask, kernel)

            contourSlice = cv2.bitwise_and(contourSlice, mask)

            # convert slice to IQ
            bModePts = np.asarray(get_list_of_roi(contourSlice))
            for ii in range(len(bModePts)):
                bModePts[ii, 1] = self.bmodeImageHeight - bModePts[ii, 1]

            polarPts = generate_polar_points(bModePts)
            contourSliceIQ = generate_mask_from_points(polarPts)
            contourSliceIQ = np.flipud(contourSliceIQ)

            self.contourSlicesIQ.append(contourSliceIQ)
            self.contourSlicesBMode.append(contourSlice)

            contourEdgeIQ = cv2.Canny(image=contourSliceIQ, threshold1=100, threshold2=200)
            kernel = np.ones((2, 2))
            contourEdgeIQ = cv2.dilate(contourEdgeIQ, kernel)
            contourEdgeIQ = np.ma.masked_where(contourEdgeIQ == 0, contourEdgeIQ)
            self.sliceWidgetIQ.IQROIMasks.append(contourEdgeIQ)

            contourEdgeBMode = cv2.Canny(image=contourSlice, threshold1=100, threshold2=200)
            kernel = np.ones((2, 2))
            contourEdgeBMode = cv2.dilate(contourEdgeBMode, kernel)
            contourEdgeBMode = np.ma.masked_where(contourEdgeBMode == 0, contourEdgeBMode)
            self.sliceWidgetBMode.BModeROIMasks.append(np.flipud(contourEdgeBMode))

            # determine if fat or not - if number of fat pixels / total pixels > thresh/100, then fat
            overlap = np.logical_and(contourSliceIQ, self.fatMask)

            overlap_sum = overlap.sum()
            contour_sum = contourSliceIQ.sum() / 255
            fatRatio = overlap_sum / contour_sum

            if fatRatio > self.ROIThreshPercent / 100:
                self.tissueType = 1
            else:
                self.tissueType = 4

            # Draw bounding rect around slice for ROI
            iqcnt = cv2.findContours(contourSliceIQ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # if no contour (channel goes out of image) then set ROI to all zeros
            if iqcnt[1] is None:
                x = 0
                y = 0
                w = 0
                h = 0
            else:
                # draw box around largest contour
                maxcnt = max(iqcnt[0][:], key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(maxcnt)

            # Add bounding rect as ROI
            self.sliceWidgetIQ.firstPointIQ = [x, y]
            self.sliceWidgetIQ.secondPointIQ = [x + w, y + h]

            # ROI = [self.sliceWidgetIQ.firstPointIQ, self.sliceWidgetIQ.secondPointIQ]
            # self.manualIQROIs.append(ROI)
            self.tissueTypeList.append(self.tissueType)
            self.countourSliceFrameNums.append(self.sliceWidgetBMode.sliceNumber)
            self.sliceWidgetIQ.tissueTypeList.append(self.tissueType)
            self.sliceWidgetBMode.tissueTypeList.append(self.tissueType)

            self.predictedLabels.append(self.tissueType)

            ### Draw image for testing
            cnts = cv2.findContours(contourSlice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            contourIm = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth, 3), np.uint8)
            cv2.drawContours(contourIm, cnts, -1, (255, 255, 255))
            testIm = cv2.bitwise_or(testIm, contourIm)

            testIQ = np.zeros((self.iqImageHeight, self.iqImageWidth, 3), np.uint8)
            contourPts = np.where(contourSliceIQ == 255)
            testIQ[contourPts[0][:], contourPts[1][:]] = 255
            cv2.rectangle(testIQ, (x, y), (x + w, y + h), (255, 0, 0), 1)

            testIms.append(contourPts)
            testROIs.append([x, y, w, h])

            if i > 0:
                im1 = testIms[i]
                im2 = testIms[i-1]
                ROI1 = testROIs[i]
                ROI2 = testROIs[i-1]

                test = np.zeros((self.iqImageHeight, self.iqImageWidth, 3), np.uint8)

                test[im1[0][:], im1[1][:]] = 255

                for it in range(len(im2[0])):
                    if (test[im2[0][it], im2[1][it]] == [255, 255, 255]).all():
                        test[im2[0][it], im2[1][it]] = [0, 255, 0]
                    else:
                        test[im2[0][it], im2[1][it]] = [0, 0, 255]

                cv2.rectangle(test, (ROI1[0], ROI1[1]), (ROI1[0] + ROI1[2], ROI1[1] + ROI1[3]), (255, 0, 0), 1)
                cv2.rectangle(test, (ROI2[0], ROI2[1]), (ROI2[0] + ROI2[2], ROI2[1] + ROI2[3]), (255, 0, 0), 1)


            # Loop thru image and find aline & sample start/stop points
            # currImageHeightIQ = self.iqImageHeight
            # currImageWidthIQ = self.iqImageWidth
            # alineStart = []
            # alineStop = []
            # sampleStart = []
            # sampleStop = []
            #
            # pointsIQ = []
            # pointsBMode = []
            #
            # for h in range(currImageWidthIQ):
            #     j = 0
            #     while j < currImageHeightIQ:
            #         if contourSliceIQ[j][h] != 0:
            #             alineStart.append(h)
            #             alineStop.append(h)
            #             sampleStart.append(j)
            #             for x in range(j + 1, currImageHeightIQ):
            #                 if contourSliceIQ[x][h] == 0:
            #                     sampleStop.append(x)
            #                     j = x + 1
            #                     break
            #                 elif x == currImageHeightIQ:
            #                     sampleStop.append(currImageHeightIQ)
            #                     j = x + 1
            #         else:
            #             j += 1
            #
            # for k in range(len(alineStart)):
            #     alines = [(alineStart[k], sampleStart[k]), (alineStop[k], sampleStop[k])]
            #
            #     bModeAlineStart = alineStart[k]
            #     bModeAlineStop = alineStop[k]
            #     bModeSampleStart = currImageHeightIQ - sampleStart[k]
            #     bModeSampleStop = currImageHeightIQ - sampleStop[k]
            #
            #     poly = [[bModeAlineStart, bModeSampleStart], [bModeAlineStop, bModeSampleStop]]
            #
            #     # creates storage lists for converted points
            #     bModePoints = []
            #     bModeContourPoints = []
            #
            #     # Iterates through the list of points and puts them in
            #     for j in range(len(poly)):
            #         x1 = poly[j][1]
            #         y1 = poly[j][0]
            #         bModePoints.append([x1, y1])
            #
            #         # getting BMode points from IQ points and adding to a list for each contour
            #         (x, y) = polarTransform.getCartesianPointsImage(bModePoints[j], self.ptSettings)
            #         bModeContourPoints.append([x, y])
            #
            #     pointsIQ.append(alines)
            #     pointsBMode.append(bModeContourPoints)
            #
            # self.sliceWidgetIQ.channelSegments.append(pointsIQ)
            # self.sliceWidgetBMode.channelSegments.append(pointsBMode)

            # Show slice with ROI
            # cv2.imshow('divisions', testIm)
            # cv2.waitKey()

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonDrawChannelROIs_2_clicked(self):
        """Draw ROIs around channel surrounding heart"""

        # Get channel (fat & nonfat combined)
        nonfat = np.zeros((self.iqImageHeight, self.iqImageWidth))
        nonfat[:, :] = self.nonFatMask[:, :]
        nonfat = nonfat.astype('uint8')

        fat = np.zeros((self.iqImageHeight, self.iqImageWidth))
        fat[:, :] = self.fatMask[:, :]
        fat = fat.astype('uint8')

        channel = fat + nonfat

        self.ROIFatFlag = "Channel"

        # clear all ROIs and previous segments
        self.tissueTypeList.clear()
        self.manualBModeROIs.clear()
        self.manualIQROIs.clear()
        self.sliceWidgetIQ.channelSegments.clear()
        self.sliceWidgetBMode.channelSegments.clear()
        self.sliceWidgetIQ.iqROIPointsList.clear()
        self.sliceWidgetBMode.bModeROIPointsList.clear()
        self.contourSlicesIQ.clear()

        # find centroid of bmode channel mask
        count = (channel == 255).sum()
        center = tuple(np.argwhere(channel == 255).sum(0) / count)
        center = tuple(int(l) for l in center)
        center = tuple(reversed(center))

        ret, im_th = cv2.threshold(channel, 90, 255, cv2.THRESH_BINARY)

        im_th[:, 200] = 0

        thinned = cv2.ximgproc.thinning(im_th, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        testIm = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth, 3), np.uint8)

        points = fatMapping.getRightVentriclePoints(thinned, thinned, type='channel')

        spread = 25

        # create masks of sector from centroid to edge of image - AND with original channel to get section of channel
        for i in range(0, len(points), spread):
            mask = np.zeros((self.iqImageHeight, self.iqImageWidth))

            # get end points of sector
            pt1 = points[i]

            if i + spread >= len(points):
                pt2 = points[i+spread]
            else:
                break

            # scale points to be outside image
            pt1[0] = center[0] + (pt1[0] - center[0]) * 5
            pt1[1] = center[1] + (pt1[1] - center[1]) * 5

            pt2[0] = center[0] + (pt2[0] - center[0]) * 5
            pt2[1] = center[1] + (pt2[1] - center[1]) * 5

            # draw on mask
            cv2.line(mask, center, pt1, 255, 1)
            cv2.line(mask, center, pt2, 255, 1)
            mask = floodFillContour(mask.astype('uint8'))

            contourSliceIQ = cv2.bitwise_and(mask.astype('uint8'), im_th.astype('uint8'))

            # determine if fat or not - if number of fat pixels / total pixels > thresh/100, then fat
            overlap = np.logical_and(contourSliceIQ, self.fatMask)

            overlap_sum = overlap.sum()
            contour_sum = contourSliceIQ.sum() / 255
            fatRatio = overlap_sum / contour_sum

            if fatRatio > self.ROIThreshPercent / 100:
                self.tissueType = 1
            else:
                self.tissueType = 4

            # Draw bounding rect around slice for ROI
            iqcnt = cv2.findContours(contourSliceIQ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # if no contour (channel goes out of image) then set ROI to all zeros
            if iqcnt[1] is None:
                x = 0
                y = 0
                w = 0
                h = 0
            else:
                # draw box around largest contour
                maxcnt = max(iqcnt[0][:], key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(maxcnt)

            # Add bounding rect as ROI
            self.sliceWidgetIQ.firstPointIQ = [x, y]
            self.sliceWidgetIQ.secondPointIQ = [x + w, y + h]

            ROI = [self.sliceWidgetIQ.firstPointIQ, self.sliceWidgetIQ.secondPointIQ]
            self.manualIQROIs.append(ROI)
            self.tissueTypeList.append(self.tissueType)

            # Loop thru image and find aline & sample start/stop points
            currImageHeightIQ = self.iqImageHeight
            currImageWidthIQ = self.iqImageWidth
            alineStart = []
            alineStop = []
            sampleStart = []
            sampleStop = []

            pointsIQ = []
            pointsBMode = []

            for h in range(currImageWidthIQ):
                j = 0
                while j < currImageHeightIQ:
                    if contourSliceIQ[j][h] != 0:
                        alineStart.append(h)
                        alineStop.append(h)
                        sampleStart.append(j)
                        for x in range(j + 1, currImageHeightIQ):
                            if contourSliceIQ[x][h] == 0:
                                sampleStop.append(x)
                                j = x + 1
                                break
                            elif x == currImageHeightIQ:
                                sampleStop.append(currImageHeightIQ)
                                j = x + 1
                    else:
                        j += 1

            for k in range(len(alineStart)):
                alines = [(alineStart[k], sampleStart[k]), (alineStop[k], sampleStop[k])]

                bModeAlineStart = alineStart[k]
                bModeAlineStop = alineStop[k]
                bModeSampleStart = currImageHeightIQ - sampleStart[k]
                bModeSampleStop = currImageHeightIQ - sampleStop[k]

                poly = [[bModeAlineStart, bModeSampleStart], [bModeAlineStop, bModeSampleStop]]

                # creates storage lists for converted points
                bModePoints = []
                bModeContourPoints = []

                # Iterates through the list of points and puts them in
                for j in range(len(poly)):
                    x1 = poly[j][1]
                    y1 = poly[j][0]
                    bModePoints.append([x1, y1])

                    # getting BMode points from IQ points and adding to a list for each contour
                    (x, y) = polarTransform.getCartesianPointsImage(bModePoints[j], self.ptSettings)
                    bModeContourPoints.append([x, y])

                pointsIQ.append(alines)
                pointsBMode.append(bModeContourPoints)

            self.sliceWidgetIQ.channelSegments.append(pointsIQ)
            self.sliceWidgetBMode.channelSegments.append(pointsBMode)

            # Show slice with ROI
            # cv2.imshow('divisions', testIm)
            # cv2.waitKey()

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    # @pyqtSlot()
    # def on_pushButtonDrawChannelROIs_2_clicked(self):
    #     nonfat = np.zeros((self.iqImageHeight, self.iqImageWidth))
    #     nonfat[:, :] = self.nonFatMask[:, :]
    #     nonfat = nonfat.astype('uint8')
    #
    #     fat = np.zeros((self.iqImageHeight, self.iqImageWidth))
    #     fat[:, :] = self.fatMask[:, :]
    #     fat = fat.astype('uint8')
    #
    #     channel = fat + nonfat
    #
    #     # set flag for naming the ROI file
    #     self.ROIFatFlag = 'channel_2'
    #
    #     if self.sliceWidgetBMode.imageBMode is not None:
    #
    #         # Clears drawn ROI lists when the ROI parameters change
    #         self.manualBModeROIs.clear()
    #         self.manualIQROIs.clear()
    #
    #         # check that contour is loaded
    #         if self.sliceWidgetIQ.channelContours is not None:
    #             # nonfat tissue type
    #             self.tissueType = 4
    #
    #             ret, im_th = cv2.threshold(channel, 90, 255, cv2.THRESH_BINARY)
    #             ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    #             thinned = cv2.ximgproc.thinning(im_th, thinningType = cv2.ximgproc.THINNING_ZHANGSUEN)
    #
    #             points = np.where(thinned == 255)
    #             points = np.array(points)
    #
    #             width = self.sliceWidgetIQ.ROIsizehoriz
    #             height = self.sliceWidgetIQ.ROIsizevert
    #             thresh_perc = self.ROIThreshPercent
    #
    #             threshold = (width * height * thresh_perc) / 100
    #
    #             # plt.figure()
    #             # plt.imshow(im_th)
    #
    #             im_h = self.iqImageHeight
    #             im_w = self.iqImageWidth
    #
    #             plt.figure()
    #             plt.imshow(im_th)
    #
    #             # for i in range(0, points.shape[1], 25):
    #             #     # check that current point is not within previous roi
    #             #     if i > 0:
    #             #         if points[0, i] + height > roi_left[0] and points[0, i] + height < roi_right[0]:
    #             #             continue
    #             #     # need to move left and right corners of ROI until they hit the edges of the channel (second point in array)
    #             #     roi_left = [points[0, i], points[1, i]]
    #             #
    #             #     roi_right = [points[0, i] + height, points[1, i]]
    #             #
    #             #     # move left point as far left as possible
    #             #     while im_th[roi_left[0], roi_left[1]] != 0 and roi_left[1] > 0:
    #             #         roi_left[1] -= 1
    #             #     # move right point
    #             #     while im_th[roi_right[0], roi_right[1]] != 0 and roi_right[1] < im_w - 2:
    #             #         roi_right[1] += 1
    #             #
    #             #     plt.scatter(roi_left[1], roi_left[0])
    #             #     plt.scatter(roi_right[1], roi_right[0])
    #             #
    #             #     self.sliceWidgetIQ.firstPointIQ = [roi_left[1], roi_left[0]]
    #             #     self.sliceWidgetIQ.secondPointIQ = [roi_right[1], roi_right[0]]
    #             #
    #             #     ROI = [self.sliceWidgetIQ.firstPointIQ, self.sliceWidgetIQ.secondPointIQ]
    #             #     self.manualIQROIs.append(ROI)
    #             #     self.tissueTypeList.append(self.tissueType)
    #
    #             first_point = [0, 0]
    #             second_point = [0, 0]
    #             for contour in ctrs:
    #                 x, y, w, h = cv2.boundingRect(contour)
    #
    #                 j = x
    #                 channel_width_right = 0
    #                 channel_width_left = 0
    #                 while j + channel_width_right < im_w and j - channel_width_left >= 0:
    #
    #                     k = y
    #                     while k < y + h:
    #
    #                         if im_th[k, j] == 255:
    #                             # plt.scatter(j, k)
    #                             # plt.scatter(j+width, k+height)
    #                             #
    #                             first_point = [j, k]
    #                             second_point = [j+1, k+height]
    #
    #                             while im_th[first_point[1], first_point[0]] != 0 and first_point[0] > 0:
    #                                 first_point[0] -= 1
    #                             while im_th[second_point[1], second_point[0]] != 0 and second_point[0] < im_w-2:
    #                                 second_point[0] += 1
    #
    #                             self.sliceWidgetIQ.firstPointIQ = first_point
    #                             self.sliceWidgetIQ.secondPointIQ = second_point
    #
    #                             ROI = [self.sliceWidgetIQ.firstPointIQ, self.sliceWidgetIQ.secondPointIQ]
    #                             self.manualIQROIs.append(ROI)
    #                             self.tissueTypeList.append(self.tissueType)
    #
    #                         k += height
    #
    #                     if second_point[0] != 0:
    #                         j = second_point[0]
    #                     else:
    #                         j += 1
    #
    #             self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
    #                                                self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
    #             self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
    #                                             self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonMergeMRI_clicked(self):
        self.window = QMainWindow()
        self.ui = Ui_MRIWindow()
        self.ui.setupUi(self.window)
        self.window.show()

    @pyqtSlot()
    def on_pushButtonRemoveMorphPoint_clicked(self):
        if len(self.sliceWidgetBMode.morphBModePoints) != 0:
            point = self.sliceWidgetBMode.morphBModePoints.pop()
            self.bModeMorphPoints.pop()

            print('Removed ', point)
            print('Number of BMode points: ', len(self.sliceWidgetBMode.morphBModePoints))

            self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                               self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        else:
            QMessageBox.warning(self, 'Unable to remove point', 'No points selected',
                                QMessageBox.Ok)

    @pyqtSlot()
    def on_pushButtonLoadPoints_clicked(self):
        if self.autoLoadFlag:
            fileNameFullPathList = self.contourPath
            fileNameFullPathList = [fileNameFullPathList.replace(os.sep, '/')]

        else:
            fileNameFullPathList, filterReturn = FileDialog.getOpenFileNames(self, 'Select Traced Contour NRRD',
                                                                             self.defaultOpenPath, '*.nrrd')

        if not fileNameFullPathList:
            return

        self.contourPointFilePath = fileNameFullPathList

        QApplication.setOverrideCursor(Qt.WaitCursor)

        # get number of filenames read in
        for currFileName in fileNameFullPathList:

            nrrdTraced = currFileName
            data, _ = nrrd.read(nrrdTraced)

            # Skip background
            layers = []
            for layer_index in range(1, data.shape[0]):
                layer = data[layer_index, :, :, :]

                # Read layer and convert to binary
                convertedLayer = convertImageFromArray(layer)
                binaryLayer = changeToBinaryMask(convertedLayer)
                layers.append(np.array(binaryLayer))

            leftEndo, leftEpi, right = tuple(layers[:3])

            # Make a mask for right ventricle by filling in contour of both right side and left endo,
            # then subtracting it from left endo mask
            combined_contour = cv2.bitwise_or(leftEpi, right)
            combined_contour = cv2.bitwise_or(combined_contour, leftEndo)
            combined_contour = np.flipud(combined_contour)

            points = np.nonzero(combined_contour[:, :, 1])

            for i in range(len(points[0])):
                self.sliceWidgetBMode.contourPoints.append((points[0][i], points[1][i]))

            self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                               self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        # enable show/hide contour button
        self.pushButtonHideContour.setEnabled(True)

    @pyqtSlot()
    def on_pushButtonHideContour_clicked(self):
        if self.sliceWidgetBMode.contourPointsFlag:
            self.sliceWidgetBMode.contourPointsFlag = 0
        else:
            self.sliceWidgetBMode.contourPointsFlag = 1

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonSaveFiducial_clicked(self):
        fileNameFullPathList = FileDialog.getSaveFileName(self, 'Select location to store fiducial points',
                                                          self._iqFileName + '_FiducialPointsUS', '*.txt')

        if self.sliceWidgetBMode.morphBModePoints is not None:
            np.savetxt(fileNameFullPathList[0], self.sliceWidgetBMode.morphBModePoints)

        else:
            QMessageBox.warning(self, 'Unable to save fiducial points', 'No points selected',
                                QMessageBox.Ok)

    @pyqtSlot()
    def on_pushButtonLoadFiducial_clicked(self):
        fileNameFullPathList = FileDialog.getOpenFileName(self, 'Select Fiducial Point Text File', self.defaultOpenPath,
                                                          '*.txt')

        if self.sliceWidgetBMode.imageBMode is not None:
            temp = np.loadtxt(fileNameFullPathList[0])

            for i in range(len(temp)):
                self.sliceWidgetBMode.morphBModePoints.append((int(temp[i][0]), int(temp[i][1])))
                self.bModeMorphPoints.append((int(temp[i][0]), int(temp[i][1])))

            self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                               self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        else:
            QMessageBox.warning(self, 'Unable to load fiducial points', 'No image loaded',
                                QMessageBox.Ok)

    @pyqtSlot()
    def on_pushButtonClearMorphPoints_clicked(self):
        if self.sliceWidgetBMode.morphBModePoints is not None:
            self.sliceWidgetBMode.morphBModePoints.clear()
            self.bModeMorphPoints.clear()

            self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                               self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        else:
            QMessageBox.warning(self, 'Unable clear fiducial points', 'No points loaded',
                                QMessageBox.Ok)

    @pyqtSlot()
    def on_pushButtonClearContourMorph_clicked(self):
        if self.sliceWidgetBMode.contourPoints is not None:
            self.sliceWidgetBMode.contourPoints.clear()

            self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                               self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

            self.pushButtonHideContour.setEnabled(False)

        else:
            QMessageBox.warning(self, 'Unable clear contour points', 'No points loaded',
                                QMessageBox.Ok)

    @pyqtSlot()
    def on_pushButtonGetOutsidePoints_clicked(self):
        """Plot outside points bspline curve using traced contours"""

        # if no contour loaded, load one
        if len(self.sliceWidgetBMode.contourPoints) == 0:
            fileNameFullPathList, filterReturn = FileDialog.getOpenFileNames(self, 'Select Traced Contour NRRD',
                                                                             self.defaultOpenPath, '*.nrrd')

            if not fileNameFullPathList:
                return

            self.contourPointFilePath = fileNameFullPathList

        else:
            fileNameFullPathList = self.contourPointFilePath

            if not fileNameFullPathList:
                return

        # if self.mriWindow.get_frame_flag() == 'D':
        #     frame = "EndDiastole"
        # fileNameFullPathList = rf'C:\Users\lgillet\School\BIRL\Utrasound Data\Expert Tracing\{self._iqFileName[0:5]}\{self._iqFileName[6:9]}-Training\Traced Contour\'

        QApplication.setOverrideCursor(Qt.WaitCursor)

        # get number of filenames read in
        for currFileName in fileNameFullPathList:

            nrrdTraced = currFileName
            data, _ = nrrd.read(nrrdTraced)

            # Skip background
            layers = []
            for layer_index in range(1, data.shape[0]):
                layer = data[layer_index, :, :, :]

                # Read layer and convert to binary
                convertedLayer = convertImageFromArray(layer)
                binaryLayer = changeToBinaryMask(convertedLayer)
                layers.append(np.array(binaryLayer))

            leftEndo, leftEpi, right = tuple(layers[:3])

            leftEpi = cv2.rotate(leftEpi, cv2.ROTATE_90_CLOCKWISE)
            right = cv2.rotate(right, cv2.ROTATE_90_CLOCKWISE)
            outside = cv2.bitwise_xor(leftEpi, right)

            outside_points = fatMapping.getRightVentriclePoints(outside[:, :, 0], outside[:, :, 0], type='us_lr')
            outside_points = np.rot90(outside_points, k=3)

            newX, newY, _, _, self.channelParam = fatMapping.reSampleAndSmoothPoints(outside_points[0, :],
                                                                                     outside_points[1, :],
                                                                                     len(outside_points[0, :]), 1000,
                                                                                     15, 12, self.shiftNum)

            outside_points = np.array((newX[0, :], newY[0, :]))
            outside_points = np.transpose(outside_points)

            currImageHeightIQ, _ = self.sliceWidgetIQ.imageIQ[self.sliceWidgetIQ.sliceNumber, :, :].shape
            outside_points_iq = []

            for j in range(len(outside_points)):
                (x, y) = polarTransform.getPolarPointsImage(outside_points[j], self.ptSettings)
                outside_points_iq.append([y, currImageHeightIQ - x])

            # make sure outside points are lists then clear
            self.sliceWidgetBMode.bModeOutsidePoints = list(self.sliceWidgetBMode.bModeOutsidePoints)
            self.sliceWidgetBMode.bModeOutsidePoints.clear()
            self.sliceWidgetIQ.iqOutsidePoints = list(self.sliceWidgetIQ.iqOutsidePoints)
            self.sliceWidgetIQ.iqOutsidePoints.clear()

            # add outside points to slice widget list and end with first point to complete contour
            self.sliceWidgetBMode.bModeOutsidePoints.append(outside_points[:])
            self.sliceWidgetBMode.bModeOutsidePoints = self.sliceWidgetBMode.bModeOutsidePoints[0]

            self.sliceWidgetIQ.iqOutsidePoints.append(outside_points_iq)

            self.sliceWidgetBMode.firstPointBModeSpline.clear()

            # draw multiple points
            # for i in range(0, len(outside_points), 50):
            #     self.sliceWidgetBMode.firstPointBModeSpline.append((outside_points[i, 0], outside_points[i, 1]))

            self.sliceWidgetBMode.firstPointBModeSpline.append((outside_points[0, 0], outside_points[0, 1]))

            self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                               self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
            self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                            self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

            # enable relevant buttons
            self.pushButtonBinaryFat.setEnabled(True)
            self.pushButtonFatThickness.setEnabled(True)

    @pyqtSlot()
    def on_pushButtonBinaryFat_clicked(self):
        """Plot fat map around US bspline using thresholded average fat thickness from MRI
        i.e. fat only drawn if fat threshold is exceeded"""

        binaryFat = self.mriWindow.get_binary_fat()
        if binaryFat:
            conversion = self.bmodeImageDepthMM / (self.bmodeFinalRadius - self.bmodeInitialRadius)
            fatThresh = self.mriWindow.get_binary_fat_thresh_mm()
            self.fatPoints = np.zeros((len(binaryFat[0])))
            self.fatPoints[:] = binaryFat[0]
            self.fatPoints *= (fatThresh / conversion)

            fatBin = binaryFat[0]
            outside_points = np.array(self.sliceWidgetBMode.bModeOutsidePoints)

            if self.sliceWidgetBMode.bModeOutsidePoints is not None:

                crossX, crossY = fatMapping.getNormalVectors(outside_points, length=fatThresh / conversion)

                for i in range(len(fatBin)):
                    if fatBin[i] == 1:
                        self.sliceWidgetBMode.bModeFatPoints.append(
                            (outside_points[i, 0] - crossY[0, i], outside_points[i, 1] + crossX[0, i]))
                    else:
                        self.sliceWidgetBMode.bModeFatPoints.append(outside_points[i])

                if fatBin[0] == 1:
                    self.sliceWidgetBMode.bModeFatPoints.append(
                        (outside_points[0, 0] - crossY[0, 0], outside_points[0, 1] + crossX[0, 0]))
                else:
                    self.sliceWidgetBMode.bModeFatPoints.append(outside_points[0])

                currImageHeightIQ, _ = self.sliceWidgetIQ.imageIQ[self.sliceWidgetIQ.sliceNumber, :, :].shape
                iqFatPoints = []
                for j in range(len(self.sliceWidgetBMode.bModeFatPoints)):
                    (x, y) = polarTransform.getPolarPointsImage(self.sliceWidgetBMode.bModeFatPoints[j],
                                                                self.ptSettings)
                    iqFatPoints.append([y, currImageHeightIQ - x])

                self.sliceWidgetIQ.iqFatPoints.append(iqFatPoints)

                self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                   self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
                self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonFatThickness_clicked(self):
        """Plot fat map around US bspline using average fat thickness from MRI"""
        thickness_mm = self.mriWindow.get_thickness_by_point()
        if thickness_mm:
            conversion = self.bmodeImageDepthMM / (self.bmodeFinalRadius - self.bmodeInitialRadius)
            self.fatPoints = thickness_mm[0]
            self.binaryFat = 0

            self.sliceWidgetBMode.bModeFatPoints.clear()
            self.sliceWidgetIQ.iqFatPoints.clear()

            thicknessByPoint = np.array(self.fatPoints)
            thicknessByPoint /= conversion

            outside_points = np.array(self.sliceWidgetBMode.bModeOutsidePoints)

            if outside_points is not None:

                for i in range(len(thicknessByPoint)):
                    crossX, crossY = fatMapping.getNormalVectors(outside_points, length=thicknessByPoint[i])

                    self.sliceWidgetBMode.bModeFatPoints.append(
                        (outside_points[i, 0] - crossY[0, i], outside_points[i, 1] + crossX[0, i]))

                # append first point again to close shape
                self.sliceWidgetBMode.bModeFatPoints[-1] = (self.sliceWidgetBMode.bModeFatPoints[0][0], self.sliceWidgetBMode.bModeFatPoints[0][1])

                currImageHeightIQ, _ = self.sliceWidgetIQ.imageIQ[self.sliceWidgetIQ.sliceNumber, :, :].shape
                iqFatPoints = []
                for j in range(len(self.sliceWidgetBMode.bModeFatPoints)):
                    (x, y) = polarTransform.getPolarPointsImage(self.sliceWidgetBMode.bModeFatPoints[j],
                                                                self.ptSettings)
                    iqFatPoints.append([y, currImageHeightIQ - x])

                self.sliceWidgetIQ.iqFatPoints.append(iqFatPoints)

                self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                   self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
                self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                                self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonClearOutsidePoints_clicked(self):
        # clear outside point bspline from US

        self.sliceWidgetBMode.bModeOutsidePoints = list(self.sliceWidgetBMode.bModeOutsidePoints)
        self.sliceWidgetBMode.bModeOutsidePoints.clear()
        self.sliceWidgetIQ.iqOutsidePoints = list(self.sliceWidgetIQ.iqOutsidePoints)
        self.sliceWidgetIQ.iqOutsidePoints.clear()

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        # disable plot fat buttons
        self.pushButtonBinaryFat.setEnabled(False)
        self.pushButtonFatThickness.setEnabled(False)

    @pyqtSlot()
    def on_pushButtonClearFatPoints_clicked(self):
        # clear fat map from MRI

        self.sliceWidgetBMode.bModeFatPoints.clear()
        self.sliceWidgetIQ.iqFatPoints.clear()

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot(int)
    def on_spinBoxShift_valueChanged(self, value):
        # shift first Bspline point
        self.shiftNum = value

    @pyqtSlot(float)
    def on_doubleSpinBoxBinFatThresh_valueChanged(self, value):
        # fat threshold for labeling ROIs from MRI changed

        print('Binary Fat Threshold:', round(value, 2))

        self.mriWindow.binary_fat_thresh_changed(round(value, 2))
        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

        fat1 = str(binaryFatThreshmm).split('.')[0]
        fat2 = str(binaryFatThreshmm).split('.')[1]
        self.SAVEPATH = fr'{self.iqChannelROIsPath}\fat_{fat1}_{fat2}\thickness_{self.sliceWidgetBMode.ContourThickness}'

    @pyqtSlot(int)
    def on_ContourThickness_valueChanged(self, value):
        # thickness of channel changed

        print('Contour Thickness: %d' % value)

        self.sliceWidgetIQ.ContourThickness = value
        self.sliceWidgetBMode.ContourThickness = value

        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

        fat1 = str(binaryFatThreshmm).split('.')[0]
        fat2 = str(binaryFatThreshmm).split('.')[1]

        self.SAVEPATH = fr'{self.iqChannelROIsPath}\fat_{fat1}_{fat2}\thickness_{self.sliceWidgetBMode.ContourThickness}'

    @pyqtSlot()
    def on_EDRadioButton_clicked(self):
        # indicate currently loaded frame is end diastole
        self.mriWindow.frame_flag_changed('D')
        print(f'Frame: E{self.mriWindow.get_frame_flag()}')

    @pyqtSlot()
    def on_ESRadioButton_clicked(self):
        # indicate currently loaded frame is end systole
        self.mriWindow.frame_flag_changed('S')
        print(f'Frame: E{self.mriWindow.get_frame_flag()}')

    # @pyqtSlot(bool)
    # def on_checkBoxEditContour_toggled(self, checked):
    #     self.editContourFlag = not self.editContourFlag
    #     print(self.editContourFlag)

    @pyqtSlot()
    def on_pushButtonComputeSpectra_clicked(self):
        """Compute spectral parameters used for ML input. Computes spectra for all cases in specified folder"""

        rfAvg = []

        if self.sliceWidgetBMode.imageBMode is None:
            QMessageBox.warning(self, 'Unable to compute spectra',
                                'Must load any IQ image first',
                                QMessageBox.Ok)
            return

        roiSize = f'{self.sliceWidgetIQ.ROIChannelNum}'

        ## used to loop thru all data in folder
        path = fr'{self.SAVEPATH}\ROI_masks\channel\{roiSize}'

        if not os.path.exists(path):
            print(f'{path} does not exist')
            return

        numFolders = len(os.listdir(path))

        casesRemoved = 0
        # if channel contour, loop thru all case folders
        if path.find("channel") != 0:
            for j, folder in enumerate(os.listdir(path)):
                folder = folder[0:2].upper() + folder[2:]
                splitFolder = folder.split('_')
                caseName = splitFolder[0]
                caseNum = splitFolder[1]
                frame = splitFolder[3]
                case = f'{caseName}_{caseNum}'

                channel = self.sliceWidgetBMode.ContourThickness
                fat1 = str(self.mriWindow.get_binary_fat_thresh_mm()).split('.')[0]
                fat2 = str(self.mriWindow.get_binary_fat_thresh_mm()).split('.')[1]
                signalThresh = self.roiAvgThresh
                roiNum = self.sliceWidgetBMode.ROIChannelNum

                ##################################
                # if case.find('MF0502PRE_05') == -1 or frame.find('ES') == -1:
                #     continue
                # if case.find('MF0326PRE_7') == -1 or frame.find('ES') == -1:
                #     continue

                # skip if spectra exists
                # spectraPath = rf'{self.dataTopLevelPath}/channelspectra/thresh_{signalThresh}/fat_{fat1}_{fat2}/thickness_{channel}/yule/Channel/{roiNum}/{case}_{frame}_lists'
                # if os.path.exists(spectraPath):
                #     continue
                ##################################
                # initialize lists for storing iq data and rois for all sections of channel
                roiDataList = []
                roiFileNameList = []

                # get each channel slice in folder
                for i, file in enumerate(sorted(os.listdir(rf'{path}\{folder}'), key=lambda s: int(re.findall(r'\d+', s)[4]))):
                    fileNameFullPathROI = [fr'{self.SAVEPATH}\ROI_masks\channel\{roiSize}\{folder}\{file[:-4]}.npy']
                    fileNameFullPathData = [
                        fr'{self.SAVEPATH}\data\IQ\channel\channelContours\{folder}.npy']
                    iqPhantomPath = rf'Phantom.iq'

                    # get roi for sector
                    for file in fileNameFullPathROI:
                        # cleaning up file name
                        roiFileNamePath = util.cleanPath(file)
                        roiFileName = util.splitext(os.path.basename(roiFileNamePath))[0]

                        # extract the text data from the ROI file
                        roiData = np.load(roiFileNamePath)

                        roiDataList.append(roiData)
                        roiFileNameList.append(roiFileName)

                ### just use single iq file rather than one for each slice
                iqDataList = [np.load(fileNameFullPathData[0])]
                caseList = [f'{case}_{frame}']

                # classify each ROI as in the anterior side groove, inferior side groove, free wall of right ventricle,
                # or inferior wall of left ventricle
                locationList, antList, infList = classifier.location_flags(caseList, self.SAVEPATH, self.usDataPath,
                                                                           roiSize, self.ant_inf_size, self.ptSettings)

                # calculate spectral parameters using yule-walker AR
                self.paramList, self.paramArray, self.normParamArray, n, avgPow = classifier.spectral_parameters_full_database(
                    roiDataList, iqDataList, iqPhantomPath, self.dataTopLevelPath, locationList, antList, infList,
                    frameFlag=self.mriWindow.get_frame_flag(), saveLists=True,
                    roiFileNames=roiFileNameList, psd_type='yule', channel_roi_num=roiSize,
                    channel_thickness=self.sliceWidgetBMode.ContourThickness, signalThresh=self.roiAvgThresh,
                    fatThresh=self.mriWindow.get_binary_fat_thresh_mm(), psd_plot=False, showPlot=False)

                print(f'done with {folder} ({j + 1}/{numFolders})\n')

                casesRemoved += n

                rfAvg.append(avgPow)

            print(f'Cases Removed: {casesRemoved}')

            print(np.average(rfAvg))


    @pyqtSlot(int)
    def on_spinBoxROIAvgThresh_valueChanged(self, value):
        # signal threshold for removing ROIs from dataset
        print(f'ROI threshold changed: {value}')
        self.roiAvgThresh = value

    @pyqtSlot()
    def on_pushButtonCreateClassifier_clicked(self):
        """Create random forest classifier model"""
        self.classifierTypeML = self.machineLearningListWidget.selectedItems()[0].text()
        print(f'\nClassifier: {self.classifierTypeML}\n')

        # folderPath = FileDialog.getExistingDirectory(None, f"Select folder with params lists")
        # labelsPath = FileDialog.getOpenFileName(None, f"Select file with labels")

        ################ features to be removed ################
        features_removed = []

        # either channel rois or fat/nonfat rois
        type = "Channel"

        # switch to plot ROC curve
        ROC = False

        # set up timestamp for model and train/test data file names
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # if channel add parameters lists and tissue type labels to array
        if type == "Channel":
            psd_type = 'yule'
            roi_size = self.sliceWidgetIQ.ROIChannelNum

            binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

            fat1 = str(binaryFatThreshmm).split('.')[0]
            fat2 = str(binaryFatThreshmm).split('.')[1]

            # load all spectra lists from folderpath
            folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\thickness_{self.sliceWidgetBMode.ContourThickness}\{psd_type}\Channel\{roi_size}'

            if not os.path.exists(folderPath):
                print(f'{folderPath} does not exist')
                return

            caseList = []

            # loop thru all case folders
            for i, folder in enumerate(os.listdir(folderPath)):
                n = 0
                numFiles = len(next(os.walk(folderPath))[1])

                # get case name and save to list
                case = folder[:-6]
                caseList.append(case)

                # should be same number of files as roi size
                while n < int(roi_size):

                    # specify whether to use normalized or unnormalized data
                    # grab parameters from file n
                    paramtype = f'norm_param_list_{roi_size}_{n}.npy'
                    currParams = np.load(fr'{folderPath}\{folder}\{paramtype}')
                    currLabel = np.load(fr'{folderPath}\{folder}\labels_{roi_size}_{n}.npy')

                    if np.argwhere(np.isnan(currParams)).size != 0:
                        pass

                    # if first file in case, create array - else add current params to array
                    if n == 0:
                        if np.all(currParams == 0):
                            currParams = np.zeros(12)
                        params = np.zeros(np.shape(currParams)[0])
                        params[:] = currParams[:]

                        labels = []
                        labels.append(currLabel)
                    else:
                        # if np.all(currParams == 0):
                        #     currParams = np.zeros(np.shape(params[0]))
                        try:
                            params = np.row_stack((params, currParams))
                        except ValueError:
                            currParams = np.zeros(12)
                            params = np.row_stack((params, currParams))
                        labels.append(currLabel)

                    n += 1

                if i == 0:
                    caseParams = np.zeros((numFiles, np.shape(params)[0], np.shape(params)[1]))
                    caseParams[0, :, :] = params[:, :]

                    caseLabels = np.zeros((numFiles, len(labels)))
                    caseLabels[0, :] = labels[:]
                else:
                    caseParams[i, :, :] = params[:, :]
                    caseLabels[i, :] = labels[:]

            num_params = len(caseParams[0, 0])
            params_temp = np.copy(caseParams)
            caseParams = None
            for i in range(num_params):
                if i not in features_removed:
                    if caseParams is None:
                        caseParams = np.copy(params_temp[:, :, i])
                    else:
                        caseParams = np.dstack((caseParams, params_temp[:, :, i]))

            # names, results = classifier.classifier_comparison(caseParams, caseLabels, caseList, timestamp)

            success, feature_importances, train_cases, test_cases, model, fat_train, nonfat_train, fat_test, nonfat_test, rand_state \
                = classifier.create_classifier(caseParams, caseLabels, caseList, model_save=[], ROC=ROC,
                                               prob_thresh=self.prob_thresh, shuffleFlag=self.classifierShuffleFlag,
                                               type=self.classifierTypeML)

            self.classifierAcc = success.fat_acc_test
            self.classifierSens = success.fat_sens_test
            self.classifierSpec = success.fat_spec_test
            self.classifierPrec = success.fat_prec_test
            self.classifierRec = success.fat_rec_test
            self.classifierFscore = success.fat_fsc_test

            print(
                f'\nAccuracy: {success.fat_acc_test}\nSensitivity: {success.fat_sens_test}\nSpecificity: {success.fat_spec_test}\nYouden\'s Index: {success.fat_youdens_test}\nPrecision: {success.fat_prec_test}\nRecall: {success.fat_rec_test}\nF-score: {success.fat_fsc_test}\n')

            if self.modelSaveFlag:
                msgB = QMessageBox()
                val = msgB.question(self, '',
                                    f'Save Model?\n\nTesting:\nAccuracy: {success.fat_acc_test}\nSensitivity: {success.fat_sens_test}\nSpecificity: {success.fat_spec_test}\nYouden\'s Index: {success.fat_youdens_test}\nPrecision: {success.fat_prec_test}\nRecall: {success.fat_rec_test}\nF-score: {success.fat_fsc_test}\n',
                                    msgB.Yes | msgB.No)

                if val == msgB.Yes:

                    # need to change the model folder in on_pushButtonCreateClassifierThickness_clicked
                    # if variable not set then can save in the normal location
                    if self.modelSaveFolder == '':
                        saveFolder = fr'{self.classifierModelsPath}\model_thickness_{self.sliceWidgetBMode.ContourThickness}_list_{roi_size}_{timestamp}'
                    else:
                        saveFolder = self.modelSaveFolder

                    if not os.path.exists(saveFolder):
                        os.makedirs(saveFolder)

                    thickness = self.mriWindow.get_binary_fat_thresh_mm()
                    fat1 = str(thickness).split('.')[0]
                    fat2 = str(thickness).split('.')[1]
                    modelSavePath = fr'{saveFolder}\model_thickness_{self.sliceWidgetBMode.ContourThickness}_fat_{fat1}_{fat2}_{timestamp}.pkl'

                    model_save = [modelSavePath]
                    if model_save:
                        saveFolder = model_save[0][:model_save[0].rfind('\\')]
                        if not os.path.exists(saveFolder):
                            os.mkdir(saveFolder)
                        classifier_save = open(model_save[0], 'wb')
                        pickle.dump(model, classifier_save)
                        classifier_save.close()

                    if modelSavePath:
                        # save training and testing cases to text file
                        with open(
                                fr'{saveFolder}\model_thickness_{self.sliceWidgetBMode.ContourThickness}_fat_{fat1}_{fat2}_{timestamp}.txt',
                                'w') as f:
                            f.write('Training Cases \t\t Testing Cases\n')
                            for i in range(len(train_cases)):
                                if i < len(test_cases):
                                    if len(train_cases[i]) > 15:
                                        f.write(f'{train_cases[i]}\t{test_cases[i]}\n')
                                    else:
                                        f.write(f'{train_cases[i]}\t\t{test_cases[i]}\n')
                                else:
                                    f.write(f'{train_cases[i]}\n')
                            f.write(f'\n\nTESTING:\n\nAccuracy: {success.fat_acc_test}\nSensitivity: {success.fat_sens_test}\nSpecificity:'
                                    f' {success.fat_spec_test}\nYouden\'s Index: {success.fat_youdens_test}\nPrecision: '
                                    f'{success.fat_prec_test}\nRecall: {success.fat_rec_test}\nF-score: {success.fat_fsc_test}\n\n'
                                    f'Training ROIs:\n\tFat: {fat_train}\n\tNonfat: {nonfat_train}\n'
                                    f'Testing ROIs:\n\tFat: {fat_test}\n\tNonfat: {nonfat_test}\n\nShuffle state: {rand_state}')
                            f.close()

            else:
                if self.modelSaveFolder == '':
                    saveFolder = fr'{self.classifierModelsPath}\model_thickness_{self.sliceWidgetBMode.ContourThickness}_list_{roi_size}_{timestamp}'
                else:
                    saveFolder = self.modelSaveFolder

                if not os.path.exists(saveFolder):
                    os.makedirs(saveFolder)

                thickness = self.mriWindow.get_binary_fat_thresh_mm()
                fat1 = str(thickness).split('.')[0]
                fat2 = str(thickness).split('.')[1]
                modelSavePath = fr'{saveFolder}\model_thickness_{self.sliceWidgetBMode.ContourThickness}_fat_{fat1}_{fat2}_{timestamp}.pkl'

                model_save = [modelSavePath]
                if model_save:
                    saveFolder = model_save[0][:model_save[0].rfind('\\')]
                    if not os.path.exists(saveFolder):
                        os.mkdir(saveFolder)
                    classifier_save = open(model_save[0], 'wb')
                    pickle.dump(model, classifier_save)
                    classifier_save.close()

                if modelSavePath:
                    # save training and testing cases to text file
                    with open(
                            fr'{saveFolder}\model_thickness_{self.sliceWidgetBMode.ContourThickness}_fat_{fat1}_{fat2}_{timestamp}.txt',
                            'w') as f:
                        f.write('Training Cases \t\t Testing Cases\n')
                        for i in range(len(train_cases)):
                            if i < len(test_cases):
                                if len(train_cases[i]) > 15:
                                    f.write(f'{train_cases[i]}\t{test_cases[i]}\n')
                                else:
                                    f.write(f'{train_cases[i]}\t\t{test_cases[i]}\n')
                            else:
                                f.write(f'{train_cases[i]}\n')
                        f.write(f'\nAccuracy: {success.fat_acc_test}\nSensitivity: {success.fat_sens_test}\nSpecificity:'
                                f' {success.fat_spec_test}\nYouden\'s Index: {success.fat_youdens_test}\nPrecision: '
                                f'{success.fat_prec_test}\nRecall: {success.fat_rec_test}\nF-score: {success.fat_fsc_test}\n\n'
                                f'Training ROIs:\n\tFat: {fat_train}\n\tNonfat: {nonfat_train}\n'
                                f'Testing ROIs:\n\tFat: {fat_test}\n\tNonfat: {nonfat_test}\n\nShuffle state: {rand_state}')
                        f.close()

        # else fat/nonfat rois
        else:
            psd_type = 'yule'
            roi_size = '10'

            # fat folder and nonfat folder
            folderPath = fr'C:\Users\lgillet\School\BIRL\channelSpectra\{psd_type}\Fat'
            folderPath2 = fr'C:\Users\lgillet\School\BIRL\channelSpectra\{psd_type}\Nonfat'

            labelList = []
            caseList = []

            paramtype = f'norm_param_list_{roi_size}x{roi_size}.npy'

            numFiles = len(next(os.walk(folderPath))[1])
            numfat = 0
            numnonfat = 0

            # loop thru fat folder and nonfat folder and add to arrays
            for i, folder in enumerate(os.listdir(folderPath)):
                currParams = np.load(fr'{folderPath}\{folder}\{paramtype}')
                currLabel = np.load(fr'{folderPath}\{folder}\labels_{roi_size}x{roi_size}.npy')

                case = f'{folder[:-6]}_Fat'
                caseList.append(case)

                if i == 0:
                    paramsList = np.zeros((np.shape(currParams)[0], np.shape(currParams)[1]))
                    paramsList[:, :] = currParams[:, :]

                    caseParams = [currParams]
                    caseLabels = [currLabel]
                    # caseParams = np.zeros((numFiles, np.shape(currParams)[0], np.shape(currParams)[1]))
                    # caseParams[0, :, :] = currParams[:, :]
                    #
                    # caseLabels = np.zeros((numFiles, len(currLabel)))
                    # caseLabels[0, :] = currLabel[:]
                else:
                    caseParams.append(currParams[:, :])
                    caseLabels.append(currLabel[:])

                numfat += sum(currLabel)

            for folder in os.listdir(folderPath2):
                currParams = np.load(fr'{folderPath2}\{folder}\{paramtype}')
                currLabel = np.load(fr'{folderPath2}\{folder}\labels_{roi_size}x{roi_size}.npy')
                case = f'{folder[:-6]}_Nonfat'

                caseList.append(case)
                caseParams.append(currParams[:, :])
                caseLabels.append(currLabel[:])

                numnonfat += sum(currLabel) / 4

            paramsList = np.array(paramsList)

            caseParams = np.array(caseParams, dtype=object)
            caseLabels = np.array(caseLabels, dtype=object)
            caseList = np.array(caseList, dtype=object)

            # paramsList, labelList = shuffle(paramsList, labelList)
            # success, feature_importances = classifier.create_classifier(paramsList, labelList, model_save=[fr'C:\Users\lgillet\School\BIRL\classifierModels\model_{paramtype[:-4]}.pkl'])

            binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

            fat1 = str(binaryFatThreshmm).split('.')[0]
            fat2 = str(binaryFatThreshmm).split('.')[1]
            saveFolder = fr'C:\Users\lgillet\School\BIRL\classifierModels\fat_{fat1}_{fat2}\model_{paramtype.split("list_")[0]}list_{roi_size}x{roi_size}_{timestamp}'

            modelSavePath = fr'{saveFolder}\model_{paramtype.split("list_")[0]}list_{roi_size}x{roi_size}_{timestamp}.pkl'
            success, feature_importances, train_cases, test_cases, model = classifier.create_classifier(caseParams,
                                                                                                        caseLabels,
                                                                                                        caseList,
                                                                                                        model_save=[])

            msgB = QMessageBox()
            val = msgB.question(self, '',
                                f'Save Model?\n\nAccuracy: {success.fat_acc}\nSensitivity: {success.fat_sens}\nSpecificity: {success.fat_spec}\nYouden\'s Index: {success.fat_youdens}\n',
                                msgB.Yes | msgB.No)

            if val == msgB.Yes:

                model_save = [modelSavePath]
                if model_save:
                    saveFolder = model_save[0][:model_save[0].rfind('\\')]
                    if not os.path.exists(saveFolder):
                        os.mkdir(saveFolder)
                    classifier_save = open(model_save[0], 'wb')
                    pickle.dump(model, classifier_save)
                    classifier_save.close()

                if modelSavePath:
                    # save training and testing cases to text file
                    with open(
                            fr'{saveFolder}\model_{paramtype.split("list_")[0]}list_{roi_size}x{roi_size}_{timestamp}.txt',
                            'w') as f:
                        f.write('Training Cases \t\t\t Testing Cases\n')
                        for i in range(len(train_cases)):
                            if i < len(test_cases):
                                f.write(f'{train_cases[i][0]}\t\t{test_cases[i][0]}\n')
                            else:
                                f.write(f'{train_cases[i][0]}\n')
                        f.close()

    @pyqtSlot()
    def on_pushButtonPredict_clicked(self):
        """Classify and plot ROIs for currently loaded US case using selected model"""
        self.tissueTypeList.clear()
        self.sliceWidgetBMode.tissueTypeList.clear()
        self.sliceWidgetIQ.tissueTypeList.clear()
        self.sliceWidgetBMode.correctList.clear()

        if not self.autoLoadFlag:
            fileNameFullPathModel, filterReturn = FileDialog.getOpenFileName(self, 'Select Model',
                                                                             self.defaultOpenPath, '*.pkl')
            if not fileNameFullPathModel:
                return
        else:
            fileNameFullPathModel = self.modelPath

        # get case name, roi size, and channel thickness currently set
        CASE = f'{self._iqFileName}_E{self.mriWindow.get_frame_flag()}'
        type = 'Channel'
        roiSize = self.sliceWidgetBMode.ROIChannelNum
        thickness = self.sliceWidgetBMode.ContourThickness

        # get fat threshold thickness
        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

        fat1 = str(binaryFatThreshmm).split('.')[0]
        fat2 = str(binaryFatThreshmm).split('.')[1]

        if type == 'Channel':
            # path to spectra and rois
            folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\thickness_{thickness}\yule\Channel\{roiSize}\{CASE}_lists'
            roiPath = fr'{self.SAVEPATH}\ROI_masks\channel\{roiSize}\{CASE[:-2]}Channel_{CASE[-2:]}'
            roiPathBMode = fr'{self.SAVEPATH}\ROI_masks_bmode\channel\{roiSize}\{CASE[:-2]}Channel_{CASE[-2:]}'

            labelList = []

            # get spectra and add to array
            n = 0
            while n < int(roiSize):
                param = f'norm_param_list_{roiSize}_{n}'
                currParams = np.load(fr'{folderPath}\{param}.npy')

                if n == 0:
                    paramsList = np.zeros(np.shape(currParams)[0])
                    paramsList[:] = currParams[:]
                else:
                    if np.all(currParams == 0):
                        currParams = np.zeros(np.shape(paramsList[0]))
                    paramsList = np.row_stack((paramsList, currParams))

                # load tissue type labels and add to list
                currLabel = np.load(fr'{folderPath}\labels_{roiSize}_{n}.npy')
                labelList = np.append(labelList, currLabel)

                n += 1

            paramsList = np.array(paramsList)

            iqROIs = []
            bModeROIs = []

            # get rois
            for n, roiFile in enumerate(sorted(os.listdir(roiPath), key=lambda s: int(re.findall(r'\d+', s)[4]))):
                if labelList[n] == 0:
                    continue
                # cleaning up file name
                self._roiFileNamePath = util.cleanPath(rf'{roiPath}\{roiFile}')
                self._roiFileName = util.splitext(os.path.basename(self._roiFileNamePath))[0]

                roiMaskIQ = np.load(self._roiFileNamePath)

                roiMaskBMode = convert_mask_from_polar_to_cartesian(np.flipud(roiMaskIQ))
                roiMaskBMode = np.flipud(roiMaskBMode)
                roiMaskBMode = roiMaskBMode.astype('uint8')

                roiEdgeIQ = cv2.Canny(image=roiMaskIQ, threshold1=100, threshold2=200)
                kernel = np.ones((2, 2))
                roiEdgeIQ = cv2.dilate(roiEdgeIQ, kernel)
                roiEdgeIQ = np.ma.masked_where(roiEdgeIQ == 0, roiEdgeIQ)
                iqROIs.append(roiEdgeIQ)

                roiEdgeBMode = cv2.Canny(image=roiMaskBMode, threshold1=100, threshold2=200)
                kernel = np.ones((2, 2))
                roiEdgeBMode = cv2.dilate(roiEdgeBMode, kernel)
                roiEdgeBMode = np.ma.masked_where(roiEdgeBMode == 0, roiEdgeBMode)
                bModeROIs.append(roiEdgeBMode)

            # remove all 0 spectra and labels
            paramsList = paramsList[~np.all(paramsList == 0, axis=1)]
            labelList = [i for i in labelList if i != 0]

            # testim = plt.figure()
            # for ii in range(len(labelList)):
            #     plt.imshow(bModeROIs[ii], cmap='RdYlGn', interpolation='nearest', origin='lower')

            # predict fat/nonfat based on spectral parameters - return classifications and succes params
            self.predictedLabels, correct, success = classifier.predict(paramsList, labelList, fileNameFullPathModel)

            # plot each roi with its respective tissue type - 1 = fat (red), 4 = nonfat (white)
            for ii in range(len(labelList)):
                self.tissueType = self.predictedLabels[ii]
                self.sliceWidgetIQ.IQROIMasks.append(iqROIs[ii])
                self.sliceWidgetBMode.BModeROIMasks.append(bModeROIs[ii])

                # ROI = [self.sliceWidgetIQ.firstPointIQ, self.sliceWidgetIQ.secondPointIQ]
                # self.manualIQROIs.append(ROI)
                self.tissueTypeList.append(self.tissueType)
                self.sliceWidgetIQ.tissueTypeList.append(self.tissueType)
                self.sliceWidgetBMode.tissueTypeList.append(self.tissueType)

                self.sliceWidgetBMode.correctList.append(correct[ii])

            print(self._roiFileName)

            self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                               self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
            self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                            self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

            if not self.autoLoadFlag:
                # display success metrics in GUI popup box
                msgB = QMessageBox()
                val = msgB.question(self, 'Results',
                                    f'Accuracy: {success.fat_acc}\nSensitivity: {success.fat_sens}\nSpecificity: {success.fat_spec}\nYouden\'s Index: {success.fat_youdens}\n',
                                    msgB.Ok)

            else:
                self.classifierAcc = success.fat_acc
                self.classifierSens = success.fat_sens
                self.classifierSpec = success.fat_spec
                # self.classifierPrec = success.fat_prec
                # self.classifierRec = success.fat_rec
                # self.classifierFscore = success.fat_fsc

        else:
            binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

            fat1 = str(binaryFatThreshmm).split('.')[0]
            fat2 = str(binaryFatThreshmm).split('.')[1]
            folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\yule\Fat\{CASE}_lists'
            folderPath2 = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\yule\Nonfat\{CASE}_lists'

            # fileNameFullPath, filterReturn = FileDialog.getOpenFileName(self, 'Select ROI Data text file',
            #                                                             self.defaultOpenPath, '*.txt')
            # if not fileNameFullPath:
            #     return

            roiPathFat = fr'{self.SAVEPATH}\ROIs\fat\15x15\{CASE[:-3]}_Fat_{CASE[-2:]}.txt'

            # cleaning up file name
            self._roiFileNamePath = util.cleanPath(roiPathFat)
            self._roiFileName = util.splitext(os.path.basename(self._roiFileNamePath))[0]

            # extract the text data from the ROI file
            roiData = open(self._roiFileNamePath, "r")
            self.roiDataList = roiData.read().splitlines()

            self.tissueTypeList.clear()
            self.manualBModeROIs.clear()
            self.manualIQROIs.clear()

            alinestart = []
            alinestop = []
            samplestart = []
            samplestop = []

            # First 2 lines are Case and labels
            # Need tissue type(3), aline start(4), aline stop(5), sample start(6), sample stop(7)
            # loop thru rows and grab needed data - discard first 2 rows
            for i in range(2, len(self.roiDataList)):
                row = self.roiDataList[i].split(", ")

                # self.tissueType = int(row[3])
                alinestart.append(int(row[4]))
                alinestop.append(int(row[5]))
                samplestart.append(int(row[6]))
                samplestop.append(int(row[7]))

            roiPathFat = fr'{self.SAVEPATH}\ROIs\nonfat\15x15\{CASE[:-3]}_Nonfat_{CASE[-2:]}.txt'
            # cleaning up file name
            self._roiFileNamePath = util.cleanPath(roiPathFat)
            self._roiFileName = util.splitext(os.path.basename(self._roiFileNamePath))[0]

            # extract the text data from the ROI file
            roiData = open(self._roiFileNamePath, "r")
            self.roiDataList = roiData.read().splitlines()

            # First 2 lines are Case and labels
            # Need tissue type(3), aline start(4), aline stop(5), sample start(6), sample stop(7)
            # loop thru rows and grab needed data - discard first 2 rows
            for i in range(2, len(self.roiDataList)):
                row = self.roiDataList[i].split(", ")

                # self.tissueType = int(row[3])
                alinestart.append(int(row[4]))
                alinestop.append(int(row[5]))
                samplestart.append(int(row[6]))
                samplestop.append(int(row[7]))

            param = 'norm_param_list'
            roiSize = '15'
            labelList = []
            for file in os.listdir(folderPath):
                if file.find(f'labels_{roiSize}') != -1:
                    currLabel = np.load(fr'{folderPath}\{file}')
                    labelList = np.append(labelList, currLabel)

                if file.find(f'{param}_{roiSize}') != -1:
                    currParams = np.load(fr'{folderPath}\{file}')
                    paramsList = np.zeros((np.shape(currParams)[0], np.shape(currParams)[1]))
                    paramsList[:, :] = currParams[:, :]

            for file in os.listdir(folderPath2):
                if file.find(f'labels_{roiSize}') != -1:
                    currLabel = np.load(fr'{folderPath2}\{file}')
                    labelList = np.append(labelList, currLabel)

                if file.find(f'{param}_{roiSize}') != -1:
                    currParams = np.load(fr'{folderPath2}\{file}')
                    paramsList = np.row_stack((paramsList, currParams))

            binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

            fat1 = str(binaryFatThreshmm).split('.')[0]
            fat2 = str(binaryFatThreshmm).split('.')[1]
            predictedLabels, _, _ = classifier.predict(paramsList, labelList,
                                                          fr'C:\Users\lgillet\School\BIRL\classifierModels\fat_{fat1}_{fat2}\model_norm_param_list_15x15.pkl')

            for ii in range(len(labelList)):
                self.tissueType = predictedLabels[ii]
                self.sliceWidgetIQ.firstPointIQ = [alinestart[ii], samplestart[ii]]
                self.sliceWidgetIQ.secondPointIQ = [alinestop[ii], samplestop[ii]]

                ROI = [self.sliceWidgetIQ.firstPointIQ, self.sliceWidgetIQ.secondPointIQ]
                self.manualIQROIs.append(ROI)
                self.tissueTypeList.append(self.tissueType)

            print(self._roiFileName)

            self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                               self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
            self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                            self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonClassifyTraining_clicked(self):
        # classify the listed training data from model's .txt file
        self.defaultOpenPath = '..\data\classifierModels'
        fileNameFullPathModel, filterReturn = FileDialog.getOpenFileName(self, 'Select Model',
                                                                         self.defaultOpenPath, '*.pkl')
        if not fileNameFullPathModel:
            return

        folderPathList = fileNameFullPathModel.split('/')[:-1]
        modelFolderPath = '/'.join(folderPathList)

        txtFilePath = f'{fileNameFullPathModel.split(".")[0]}.txt'

        type = 'Channel'
        roiSize = self.sliceWidgetIQ.ROIChannelNum

        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

        fat1 = str(binaryFatThreshmm).split('.')[0]
        fat2 = str(binaryFatThreshmm).split('.')[1]

        with open(txtFilePath) as f:
            file = f.read()
            f.close()

        fLines = file.split('\n')[1:-7]

        for i, line in enumerate(fLines):
            loc = line.find('\t')
            if loc != -1:
                fLines[i] = line[:loc]

        labelList = []
        for i, case in enumerate(fLines):
            if case.find('MF') == -1:
                break

            folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\thickness_{self.sliceWidgetBMode.ContourThickness}\yule\Channel\{roiSize}\{case}_lists'

            n = 0
            while n < int(roiSize):
                param = f'norm_param_list_{roiSize}_{n}'
                currParams = np.load(fr'{folderPath}\{param}.npy')

                if np.all(currParams == 0):
                    n += 1
                    continue

                if i == 0 and n == 0:
                    paramsList = np.zeros(np.shape(currParams)[0])
                    paramsList[:] = currParams[:]
                else:
                    paramsList = np.row_stack((paramsList, currParams))

                # load tissue type labels and add to list
                currLabel = np.load(fr'{folderPath}\labels_{roiSize}_{n}.npy')
                labelList = np.append(labelList, currLabel)

                n += 1

        paramsList = np.array(paramsList)
        # paramsList = standardize_features([paramsList])
        labelList[labelList == 4] = 0
        #save arrays in excel file
        # df = pd.DataFrame(paramsList)
        # labelList[labelList > 2] = 0
        # df1 = pd.DataFrame(labelList)
        # df = pd.concat([df, df1], axis=1)
        # ## save to xlsx file
        # filepath = self.defaultOpenPath + '\DataTrain.xlsx'
        # df.to_excel('DataTest.xlsx', index=False)

        predictedLabels, _, success = classifier.predict(paramsList, labelList,
                                                                              fileNameFullPathModel)

        msgB = QMessageBox()
        val = msgB.question(self, '',
                            f'Training Data Results\n\nAccuracy: {success.fat_acc}\nSensitivity: {success.fat_sens}\nSpecificity: {success.fat_spec}\nYouden\'s Index: {success.fat_youdens}\n\nSave?\n',
                            msgB.Yes | msgB.No)

        if val == msgB.Yes:

            save = f'{modelFolderPath}/train_results_{folderPathList[-1]}.txt'

            if save:
                # save training and testing cases to text file
                with open(save, 'w') as f:
                    f.write(f'Training data success:\nAccuracy: {success.fat_acc}\nSensitivity: {success.fat_sens}\n'
                            f'Specificity: {success.fat_spec}\nYouden\'s Index: {success.fat_youdens}\n')
                    f.close()

    @pyqtSlot()
    def on_pushButtonClassifyTesting_clicked(self):
        # classify the listed testing data from model's .txt file
        self.defaultOpenPath = '..\data\classifierModels'
        fileNameFullPathModel, filterReturn = FileDialog.getOpenFileName(self, 'Select Model',
                                                                         self.defaultOpenPath, '*.pkl')
        if not fileNameFullPathModel:
            return

        folderPathList = fileNameFullPathModel.split('/')[:-1]
        modelFolderPath = '/'.join(folderPathList)

        txtFilePath = f'{fileNameFullPathModel.split(".")[0]}.txt'

        type = 'Channel'
        roiSize = self.sliceWidgetBMode.ROIChannelNum
        thickness = self.sliceWidgetIQ.ContourThickness

        with open(txtFilePath) as f:
            file = f.read()
            f.close()

        fLines = file.split('\n')[1:-7]
        test = []

        for i, line in enumerate(fLines):
            loc = line.rfind('\t')
            if loc != -1:
                test.append(line[loc + 1:])

        labelList = []
        for i, case in enumerate(test):
            binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

            fat1 = str(binaryFatThreshmm).split('.')[0]
            fat2 = str(binaryFatThreshmm).split('.')[1]
            folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\thickness_{self.sliceWidgetBMode.ContourThickness}\yule\Channel\{roiSize}\{case}_lists'

            n = 0
            while n < int(roiSize):
                param = f'norm_param_list_{roiSize}_{n}'
                currParams = np.load(fr'{folderPath}\{param}.npy')

                if np.all(currParams == 0):
                    n += 1
                    continue

                if i == 0 and n == 0:
                    paramsList = np.zeros(np.shape(currParams)[0])
                    paramsList[:] = currParams[:]
                else:
                    # check if all zeros before appending
                    if not np.any(currParams):
                        currParams = np.zeros(len(paramsList[0]))
                        paramsList = np.row_stack((paramsList, currParams))
                    else:
                        try:
                            paramsList = np.row_stack((paramsList, currParams))
                        except ValueError:
                            pass

                # load tissue type labels and add to list
                currLabel = np.load(fr'{folderPath}\labels_{roiSize}_{n}.npy')
                labelList = np.append(labelList, currLabel)

                n += 1

        paramsList = np.array(paramsList)
        # paramsList = standardize_features([paramsList])
        labelList[labelList == 4] = 0

        predictedLabels, _, success = classifier.predict(paramsList, labelList,
                                                                              fileNameFullPathModel)

        msgB = QMessageBox()
        val = msgB.question(self, 'Results',
                            f'Testing Data Results\n\nAccuracy: {success.fat_acc}\nSensitivity: {success.fat_sens}\nSpecificity: {success.fat_spec}\nYouden\'s Index: {success.fat_youdens}\n\nSave?\n',
                            msgB.Yes | msgB.No)

        if val == msgB.Yes:

            save = f'{modelFolderPath}/test_results_{folderPathList[-1]}.txt'

            if save:
                # save training and testing cases to text file
                with open(save, 'w') as f:
                    f.write(f'Testing data success:\nAccuracy: {success.fat_acc}\nSensitivity: {success.fat_sens}\n'
                            f'Specificity: {success.fat_spec}\nYouden\'s Index: {success.fat_youdens}\n')
                    f.close()

    @pyqtSlot()
    def on_pushButtonClassifierTest_clicked(self):
        """
        Rerandomize data and create classifier number of times specified by iterations variable.
        Calculate average accuracy, sensitivity, and specificity
        """
        acc = []
        sens = []
        spec = []
        prec = []
        rec = []
        fscore = []
        iterations = 50

        self.modelSaveFlag = 0
        for i in range(iterations):
            MainWindow.on_pushButtonCreateClassifier_clicked(self)
            acc.append(self.classifierAcc)
            sens.append(self.classifierSens)
            spec.append(self.classifierSpec)
            prec.append(self.classifierPrec)
            rec.append(self.classifierRec)
            print(f'\nDone with {i + 1} out of {iterations}')

        self.modelSaveFlag = 1

        accuracy = sum(acc) / len(acc)
        sensitivity = sum(sens) / len(sens)
        specificity = sum(spec) / len(spec)

        self.accAvg = accuracy
        self.sensAvg = sensitivity
        self.specAvg = specificity

        self.accSpread = max(acc) - min(acc)
        self.sensSpread = max(sens) - min(sens)
        self.specSpread = max(spec) - min(spec)

        self.accStd = np.std(acc)
        self.sensStd = np.std(sens)
        self.specStd = np.std(spec)

        print(
            f'\nTesting Data Results\n\nAccuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\n')
        # msgB = QMessageBox()
        # val = msgB.question(self, '',
        #                     f'Testing Data Results\n\nAccuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\n',
        #                     msgB.Yes | msgB.No)

    @pyqtSlot()
    def on_pushButtonRunClassifierAll_clicked(self):
        # run classifier on all data and save to excel sheet

        # set up timestamp for file name
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        thicknessList = self.thicknessList
        roiSizeList = self.roiNumList
        fatThreshList = self.fatThreshList
        roiThreshList = self.roiAvgThreshList
        probThreshList = self.probThreshList
        clfList = self.classifiersList

        clfsAll = ["RF", "QDA", "SVM", "AdaBoost", "KNN", "MLP", "XGBoost", "LGBM", "VotingClassifier", "StackingClassifier"]

        dfList = []
        sheetList = []

        imPredictList = ["MF0303POST_4_ED", "MF0313PRE_3_ED", "MF0504PRE_2_ES", "MF0313PRE_7_ES", "MF0309PRE_5_ED", "MF0511PRE_09_ES"]
        clfPredictList = [r"D:\school\BIRL\sp2024_paper\best_RF\model_thickness_10_fat_5_5_20240624-130933.pkl"]
        # clfPredictList = [r"D:\school\BIRL\EchoFatSegmentationGUI\data\classifierModels\model_thickness_16_list_40_20240218-173812\model_thickness_16_fat_7_5_20240218-173812.pkl"]
        clfPredictListNames = ["RF"]

        # don't want dialogue box asking to save each model
        self.modelSaveFlag = 0

        # MainWindow.on_pushButtonGetAllData_clicked(self)

        for clf in clfList:
            for prob_thresh in probThreshList:
                for fatThresh in fatThreshList:
                    for thickness in thicknessList:
                        for roiNum in roiSizeList:
                            for roiThresh in roiThreshList:
                                self.on_doubleSpinBoxBinFatThresh_valueChanged(fatThresh)
                                self.on_ContourThickness_valueChanged(thickness)
                                self.on_ROIChannelNum_valueChanged(roiNum)
                                self.on_spinBoxROIAvgThresh_valueChanged(roiThresh)

                                clfInd = clfsAll.index(clf)
                                self.machineLearningListWidget.setCurrentRow(clfInd)

                                self.prob_thresh = prob_thresh

                                MainWindow.on_pushButtonComputeSpectra_clicked(self)
        #                         MainWindow.on_pushButtonClassifierTest_clicked(self)
        #
        #                         df = pd.DataFrame(
        #                             [[self.accAvg, self.accSpread, self.accStd], [self.sensAvg, self.sensSpread,
        #                                                                           self.sensStd], [self.specAvg,
        #                                                                                           self.specSpread,
        #                                                                                           self.specStd]],
        #                             index=['Accuracy', 'Sensitivity', 'Specificity'],
        #                             columns=['Value', 'Spread', 'Std'])
        #
        #                         dfList.append(df)
        #                         sheetList.append(f'{clf}-{fatThresh}-{thickness}-{roiNum}-{roiThresh}')
        #
        # with pd.ExcelWriter(f'{self.dataTopLevelPath}\classifierResultsSheets\{timestamp}.xlsx') as writer:
        #     for n, df in enumerate(dfList):
        #         df.to_excel(writer, sheetList[n])
        #
        self.modelSaveFlag = 1
        #
        # caseName = [x.split('P')[0] for x in imPredictList]
        # prePost = [('P' + x.split('P')[1]).split('_')[0] for x in imPredictList]
        # caseNumbers = [x.split('_')[1] for x in imPredictList]
        # frameType = [x.split('_')[2] for x in imPredictList]
        #
        # # set ultrasound and mri to load from saved path rather than ask
        # self.autoLoadFlag = 1
        #
        # pred_acc = []
        #
        # for n in range(len(imPredictList)):
        #         MainWindow.on_pushButtonClearAll_clicked(self)
        #         caseDigits = re.findall(r'\d', imPredictList[n])
        #
        #         if frameType[n] == 'ED':
        #             frame = 'EndDiastole'
        #         else:
        #             frame = 'EndSystole'
        #
        #         # load ultrasound image
        #         if prePost[n] == 'POST2':
        #             self.iqPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n][:-1].capitalize()}-Training2\IQ\{caseName[n]}{prePost[n]}_{caseNumbers[n]}.iq'
        #         else:
        #             self.iqPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n].capitalize()}-Training\IQ\{caseName[n]}{prePost[n]}_{caseNumbers[n]}.iq'
        #             if not os.path.exists(self.iqPath):
        #                 self.iqPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n].capitalize()}-Training\IQ\{caseName[n]}_{caseNumbers[n]}.iq'
        #
        #         MainWindow.on_pushButtonLoadIQ_clicked(self)
        #
        #         if frameType[n] == 'ED':
        #             MainWindow.on_pushButtonGoToED_clicked(self)
        #             frameNum = self.EDFrame
        #         else:
        #             MainWindow.on_pushButtonGoToES_clicked(self)
        #             frameNum = self.ESFrame
        #
        #         for nn in range(len(clfPredictList)):
        #             # clfInd = clfsAll.index(clfPredictListNames[nn])
        #             # self.machineLearningListWidget.setCurrentRow(clfInd)
        #
        #             MainWindow.on_pushButtonClearROIs_clicked(self)
        #
        #             self.modelPath = clfPredictList[nn]
        #             MainWindow.on_pushButtonPredict_clicked(self)
        #
        #             impath = rf"D:\school\BIRL\EchoFatSegmentationGUI\data\classifierResultsSheets\predictions-{timestamp}\{imPredictList[n]}"
        #             if not os.path.exists(impath):
        #                 os.makedirs(impath)
        #
        #             fig = self.sliceWidgetBMode.axes.get_figure()
        #             fig.savefig(rf"{impath}\{clfPredictListNames[nn]}.png")
        #
        #             pred_acc.append(self.classifierAcc)
        # print(f"\nPredicted accuracy: {np.average(pred_acc)}")
        #
        # self.autoLoadFlag = 0

    @pyqtSlot()
    def on_pushButtonGetAllData_clicked(self):
        """
        Function to automatically collect IQ data and ROI locations from ultrasound
        Loads corresponding MRI and ultrasound images and aligns them using information from
        excel sheet.
        """

        fatAvgs = []
        maxAvgs = []

        roiSizes = self.roiNumList
        fatThresholds = self.fatThreshList
        contourThicknesses = self.thicknessList

        # read excel file
        df = pd.read_excel(rf'{self.dataTopLevelPath}\us_mri_match_shift_amts.xlsx')

        cases = df['Case'].tolist()
        mriSlices = df['MRI Slice'].tolist()
        mriShift = df['MRI Shift'].tolist()
        usShift = df['US Shift'].tolist()
        mriRotation = df['MRI Rotate'].tolist()
        usObserver = df['US Obs'].tolist()
        mriObserver = df['MRI Obs'].tolist()

        caseName = []
        # [x.split('P')[0] for x in cases]
        prePost = []
        for x in cases:
            sp1 = x.split('P')
            if len(sp1) > 1:
                prePost.append((('P' + sp1[1]).split('_')[0]).upper())
                caseName.append(sp1[0].upper())
            else:
                prePost.append('')
                caseName.append(sp1[0].split('_')[0].upper())
        # [('P' + x.split('P')[1]).split('_')[0] for x in cases]
        caseNumbers = [x.split('_')[1] for x in cases]
        frameType = [x.split('_')[2].upper() for x in cases]

        # set ultrasound and mri to load from saved path rather than ask
        self.autoLoadFlag = 1

        # open mri window
        # mriWindow = gui.mriWindow.MRIWindow()
        # mriWindow.show()

        self.IQRadioButton.setChecked(True)

        for fatThresh in fatThresholds:
            self.doubleSpinBoxBinFatThresh.setValue(fatThresh)
            for thickness in contourThicknesses:
                self.ContourThickness.setValue(thickness)
                for n in range(len(cases)):

                    ###### uncomment this section to skip cases that are already saved ######
                    path = fr'{self.SAVEPATH}\data\IQ\channel\channelContours\{caseName[n]}{prePost[n]}_{caseNumbers[n]}_Channel_{frameType[n]}.npy'
                    path2 = fr'{self.SAVEPATH}\data\IQ\channel\channelContours\{caseName[n]}_{caseNumbers[n]}_Channel_{frameType[n]}.npy'

                    roiPath = fr'{self.SAVEPATH}\ROI_masks\channel'
                    if os.path.exists(path) or os.path.exists(path2):
                        rois = os.listdir(roiPath)
                        rois = [int(x) for x in rois]
                        if len(roiSizes) > len(rois):
                            if set(rois).issubset(roiSizes):
                                continue
                        else:
                            if set(roiSizes).issubset(rois):
                                continue

                    # if cases[n].find('MF0326PRE_7_ES') == -1:
                    #     continue
                    # if cases[n].find('MF0502PRE_05_ES') == -1:
                    #     continue

                    MainWindow.on_pushButtonClearAll_clicked(self)
                    caseDigits = re.findall(r'\d', cases[n])

                    if frameType[n].upper() == 'ED':
                        frame = 'EndDiastole'
                    else:
                        frame = 'EndSystole'

                    # load ultrasound and mri image
                    if prePost[n].upper() == 'POST2':
                        self.iqPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n][:-1].capitalize()}-Training2\IQ\{caseName[n]}{prePost[n]}_{caseNumbers[n]}.iq'
                    elif prePost[n] == '':
                        self.iqPath = rf'{self.usDataPath}\{caseName[n]}\Pre-Training\IQ\{caseName[n]}{prePost[n]}_{caseNumbers[n]}.iq'
                    else:
                        self.iqPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n].capitalize()}-Training\IQ\{caseName[n]}{prePost[n]}_{caseNumbers[n]}.iq'
                        if not os.path.exists(self.iqPath):
                            self.iqPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n].capitalize()}-Training\IQ\{caseName[n]}_{caseNumbers[n]}.iq'

                    if caseDigits[1] == '3':
                        self.mriPath = rf'{self.mriDataPath}\MF{caseDigits[0]}{caseDigits[1]}{caseDigits[2]}{caseDigits[3]}-{prePost[n]}\{caseDigits[1]}{caseDigits[2]}{caseDigits[3]}-{prePost[n]}-{frame}-Cropped-Segmentation-{mriObserver[n]}.seg.nrrd'
                    else:
                        if prePost[n] == '':
                            self.mriPath = rf'{self.mriDataPath}\MF{caseDigits[0]}{caseDigits[1]}{caseDigits[2]}{caseDigits[3]}-PRE\{caseDigits[1]}{caseDigits[2]}{caseDigits[3]}-{frame}-Cropped-Segmentation-{mriObserver[n]}.seg.nrrd'
                        else:
                            self.mriPath = rf'{self.mriDataPath}\MF{caseDigits[0]}{caseDigits[1]}{caseDigits[2]}{caseDigits[3]}-{prePost[n]}\{caseDigits[1]}{caseDigits[2]}{caseDigits[3]}-{frame}-Cropped-Segmentation-{mriObserver[n]}.seg.nrrd'
                    if caseName[n] == 'MF0313' and prePost[n] == 'POST2':
                        self.mriPath = rf'{self.mriDataPath}\MF{caseDigits[0]}{caseDigits[1]}{caseDigits[2]}{caseDigits[3]}-{prePost[n][:-1]}\{caseDigits[1]}{caseDigits[2]}{caseDigits[3]}-{prePost[n][:-1]}-{frame}-Cropped-Segmentation-{mriObserver[n]}.seg.nrrd'
                    if cases[n - 1][:-1] != cases[n][:-1] or self._iqFileName == '':
                        MainWindow.on_pushButtonLoadIQ_clicked(self)

                    if frameType[n].upper() == 'ED':
                        MainWindow.on_pushButtonGoToED_clicked(self)
                        # frameNum = self.EDFrame
                    else:
                        MainWindow.on_pushButtonGoToES_clicked(self)
                        # frameNum = self.ESFrame

                    # get frame num based on observer
                    if prePost[n] == '':
                        cntFile = rf'{self.usDataPath}\{caseName[n]}\Pre-Training\Traced Contour\{frame}'
                        cntFormat = f'{caseName[n]}_{caseNumbers[n]}_{frameType[n]}'

                    elif prePost[n] == 'POST2':
                        cntFile = rf'{self.usDataPath}\{caseName[n]}\Post-Training2\Traced Contour\{frame}'
                        cntFormat = f'{caseName[n]}{prePost[n]}_{caseNumbers[n]}_{frameType[n]}'

                    else:
                        cntFile = rf'{self.usDataPath}\{caseName[n]}\{prePost[n]}-Training\Traced Contour\{frame}'
                        cntFormat = f'{caseName[n]}{prePost[n]}_{caseNumbers[n]}_{frameType[n]}'

                    for file in os.listdir(cntFile):
                        if file.lower().startswith(cntFormat.lower()) and file.lower().endswith(
                                f'{usObserver[n]}.nrrd'.lower()):
                            frameNum = int(file.split('_')[-2])

                    contourPath = fr'{self.SAVEPATH}\contours\thickness_{self.sliceWidgetBMode.ContourThickness}\{self._iqFileName}_{frameType[n]}'
                    if os.path.exists(contourPath):
                        MainWindow.on_pushButtonLoadContour_clicked(self)
                    else:
                        self.mriWindow.on_pushButtonLoadImage_clicked()

                        self.mriWindow.on_horizontalSliderMRISlice_valueChanged(
                            mriSlices[n] - self.mriWindow.get_segment_offset())

                        self.mriWindow.spinBoxShift.setValue(mriShift[n])
                        self.spinBoxShift.setValue(usShift[n])

                        if mriRotation[n] == 'CW':
                            self.mriWindow.on_pushButtonRotateCW_clicked()
                        elif mriRotation[n] == 'CCW':
                            self.mriWindow.on_pushButtonRotateCCW_clicked()

                        self.mriWindow.on_pushButtonGetOutsidePoints_clicked()

                        if prePost[n] == 'POST2':
                            self.contourPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n][:-1].capitalize()}-Training2\Traced Contour\{frame}\{cases[n]}_{frameNum}_{usObserver[n]}.nrrd'
                        else:
                            self.contourPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n].capitalize()}-Training\Traced Contour\{frame}\{cases[n]}_{frameNum}_{usObserver[n]}.nrrd'
                            if not os.path.exists(self.contourPath):
                                self.contourPath = rf'{self.usDataPath}\{caseName[n]}\Pre-Training\Traced Contour\{frame}\{caseName[n]}_{caseNumbers[n]}_{frameType[n]}_{frameNum}_{usObserver[n]}.nrrd'

                        MainWindow.on_pushButtonLoadPoints_clicked(self)
                        MainWindow.on_pushButtonGetOutsidePoints_clicked(self)

                        ################ TEMPORARY CODE: used for getting averge fat thickness over dataset ###############
                        thickness_mm = self.mriWindow.get_thickness_by_point()[0]
                        avg = np.average(thickness_mm)
                        max_thickness = np.max(thickness_mm)
                        fatAvgs.append(avg)
                        maxAvgs.append(max_thickness)
                        MainWindow.on_pushButtonClearAll_clicked(self)

                        self.mriWindow.on_horizontalSliderMRISlice_valueChanged(1)
                        continue
                        ###################################################################################################

                        MainWindow.on_pushButtonBinaryFat_clicked(self)
                        MainWindow.on_pushButtonNewContour_clicked(self)
                        MainWindow.on_pushButtonClearAll_clicked(self)
                    MainWindow.on_pushButtonLoadContour_clicked(self)

                    for roi in roiSizes:
                        self.ROIChannelNum.setValue(roi)

                        MainWindow.on_pushButtonDrawChannelROIs_clicked(self)
                        MainWindow.on_pushButtonExportChannel_clicked(self)
                        MainWindow.on_pushButtonExportROI_clicked(self)

                    MainWindow.on_pushButtonClearAll_clicked(self)

                    self.mriWindow.on_horizontalSliderMRISlice_valueChanged(1)

        self.autoLoadFlag = 0

        print(f'Average fat: {np.average(fatAvgs)}\n')
        print(f'Average maximum: {np.average(maxAvgs)}\n')
        print(f'Highest fat thickness: {np.max(maxAvgs)}\n')

    @pyqtSlot()
    def on_pushButtonInterpolateFat_clicked(self):
        """plot approximate fat mapping based on predicted ROIs"""

        # clear fat map points
        self.sliceWidgetBMode.bModeFatMap = []

        # clear predicted fat points
        self.predictedThickness = []

        # check for predicted labels
        if len(self.predictedLabels) == 0 or len(self.sliceWidgetBMode.bModeOutsidePoints) == 0:
            print('Load predicted ROIs and Bspline first!')
            return

        CASE = f'{self._iqFileName}_E{self.mriWindow.get_frame_flag()}'
        roiSize = self.sliceWidgetBMode.ROIChannelNum
        thickness = self.sliceWidgetBMode.ContourThickness

        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

        fat1 = str(binaryFatThreshmm).split('.')[0]
        fat2 = str(binaryFatThreshmm).split('.')[1]

        folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\thickness_{thickness}\yule\Channel\{roiSize}\{CASE}_lists'
        roiPath = fr'{self.SAVEPATH}\ROI_masks_bmode\channel\{roiSize}\{CASE[:-2]}Channel_{CASE[-2:]}'

        labelList = []

        # get labels
        for n in range(int(roiSize)):
            # load tissue type labels and add to list
            currLabel = np.load(fr'{folderPath}\labels_{roiSize}_{n}.npy')
            labelList = np.append(labelList, currLabel)

        bModeROIs = []
        roiCentroids = []

        for n, roiFile in enumerate(sorted(os.listdir(roiPath), key=lambda s: int(re.findall(r'\d+', s)[4]))):
            if labelList[n] == 0:
                continue
            # cleaning up file name
            self._roiFileNamePath = util.cleanPath(rf'{roiPath}\{roiFile}')
            self._roiFileName = util.splitext(os.path.basename(self._roiFileNamePath))[0]

            roiMaskBMode = np.load(self._roiFileNamePath)
            roiMaskBMode = np.flipud(roiMaskBMode)
            roiMaskBMode = roiMaskBMode.astype('uint8')

            bModeROIs.append(roiMaskBMode)

            centroid = scipy.ndimage.measurements.center_of_mass(roiMaskBMode)
            roiCentroids.append(centroid)

        # draw normal vectors and find which ROI the point is within - if fat ROI then add normal vector point to
        # list, else add bspline point
        outsidePoints = np.array(self.sliceWidgetBMode.bModeOutsidePoints)

        outsidePoints_iq = []
        currImageHeightIQ = self.iqImageHeight
        for j in range(len(outsidePoints)):
            (x, y) = polarTransform.getPolarPointsImage(outsidePoints[j], self.ptSettings)
            outsidePoints_iq.append([y, currImageHeightIQ - x])

        outsidePoints_iq = np.array(outsidePoints_iq)

        # get fat thickness and convert to pixels in ultrasound
        conversion = (self.bmodeFinalRadius - self.bmodeInitialRadius) / self.bmodeImageDepthMM  # [pixels/mm]
        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

        length = binaryFatThreshmm * conversion

        # length for testing which ROI point belongs to
        testLength = 3.5    # [mm]
        testLength *= conversion

        crossX, crossY = fatMapping.getNormalVectors(outsidePoints, length=length)
        crossXTest, crossYTest = fatMapping.getNormalVectors(outsidePoints, length=testLength)

        fatContourPoints = []
        for i in range(0, len(crossX[0])):
            normalPoint = (outsidePoints[i, 0] - crossY[0, i], outsidePoints[i, 1] + crossX[0, i])
            splinePoint = (outsidePoints[i, 0], outsidePoints[i, 1])

            roiNum = 0
            lowestDist = 9999
            for j in range(len(bModeROIs)):
                # if normal point within ROI
                # if bModeROIs[j][int(outsidePoints[i, 1] + crossXTest[0, i])][int(outsidePoints[i, 0] - crossYTest[0, i])] == 255:
                #     roiNum = j
                #     break

                dist = spatial.distance.euclidean(splinePoint, [roiCentroids[j][1], roiCentroids[j][0]])
                if dist < lowestDist:
                    roiNum = j
                    lowestDist = dist

            # if ROI that point is within is fat add normal vector point to list
            if self.predictedLabels[roiNum] == 1:
                # fat
                fatContourPoints.append(np.array(normalPoint))
                self.predictedThickness.append(self.mriWindow.get_binary_fat_thresh_mm())
            else:
                # not fat
                fatContourPoints.append(np.array((outsidePoints[i, 0], outsidePoints[i, 1])))
                self.predictedThickness.append(0)

        # make sure fat map is list then clear
        self.sliceWidgetBMode.bModeFatMap = list(self.sliceWidgetBMode.bModeFatMap)
        self.sliceWidgetBMode.bModeFatMap.clear()

        self.sliceWidgetBMode.bModeFatMap.append(fatContourPoints)
        self.sliceWidgetBMode.bModeFatMap = self.sliceWidgetBMode.bModeFatMap[0]

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonFatMapDSC_clicked(self):
        """Calculate dice similarity coefficient of predicted fat map compared
        to ground truth fat map from MRI. Both fat maps must be loaded in US window"""

        if len(self.sliceWidgetBMode.bModeFatMap) == 0:
            print('Load fat interpolation first!')

        if len(self.sliceWidgetBMode.bModeFatPoints) == 0:
            MainWindow.on_pushButtonBinaryFat_clicked(self)

        # get outside points spline and make this the inside of the contour
        points = np.array(self.sliceWidgetBMode.bModeOutsidePoints)
        points = np.around(points, 0)
        points = points.astype(int)
        xPts = points[:, 1]
        yPts = points[:, 0]

        # fill mask with outside points
        originalarray = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))
        originalarray[xPts, yPts] = 255
        fillcontour = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth, 3))
        fillcontour[:, :, 1] = originalarray[:, :]
        fillcontour = fillcontour.astype('uint8')

        # slightly dilate contour to make sure it's closed
        kernel = np.ones((2, 2))
        fillcontour = cv2.dilate(fillcontour, kernel)

        # fill contour
        originalarray = floodFillContour(fillcontour)[:, :, 0]

        # get fat thickness points and make this the outside of the contour
        mriFatArray = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))

        # convert fat thickness from mm to pixels
        conversion = self.bmodeImageDepthMM / (self.bmodeFinalRadius - self.bmodeInitialRadius)
        fatThickness = np.array(self.fatPoints)
        fatThickness = np.around(fatThickness, 0)
        fatThickness = fatThickness.astype(int)

        xFat = []
        yFat = []

        # generate normal vector for each outside point, place a point at each location along this vector up until
        # fat thickness
        for i in range(len(fatThickness)):
            for j in range(fatThickness[i]):
                crossX, crossY = fatMapping.getNormalVectors(self.sliceWidgetBMode.bModeOutsidePoints,
                                                             length=j)
                xFat.append(self.sliceWidgetBMode.bModeOutsidePoints[i, 1] + crossX[0, i])
                yFat.append(self.sliceWidgetBMode.bModeOutsidePoints[i, 0] - crossY[0, i])

        xFat = np.around(np.array(xFat), 0)
        xFat = xFat.astype(int)
        yFat = np.around(np.array(yFat), 0)
        yFat = yFat.astype(int)

        # fill mask with fat thickness points
        mriFatArray[xFat, yFat] = 255
        mriFatArray = mriFatArray.astype('uint8')
        originalarray = originalarray.astype('uint8')
        dilatedarray = cv2.bitwise_or(mriFatArray, originalarray)

        filldilated = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth, 3))
        filldilated[:, :, 1] = dilatedarray[:, :]
        filldilated = filldilated.astype('uint8')
        mriFatArray = floodFillContour(filldilated)[:, :, 0]

        # initialized array for predicted fat contour
        predictedFatArray = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth))

        # get coordinates for predicted fat map
        fatThicknessPredicted = np.array(self.predictedThickness)
        fatThicknessPredicted /= conversion
        fatThicknessPredicted = np.around(fatThicknessPredicted, 0)
        fatThicknessPredicted = fatThicknessPredicted.astype(int)

        xFatPredicted = []
        yFatPredicted = []

        # generate normal vector for each outside point, place a point at each location along this vector up until
        # fat thickness
        for i in range(len(fatThicknessPredicted)):
            for j in range(fatThicknessPredicted[i]):
                crossX, crossY = fatMapping.getNormalVectors(self.sliceWidgetBMode.bModeOutsidePoints,
                                                             length=j)
                xFatPredicted.append(self.sliceWidgetBMode.bModeOutsidePoints[i, 1] + crossX[0, i])
                yFatPredicted.append(self.sliceWidgetBMode.bModeOutsidePoints[i, 0] - crossY[0, i])

        xFatPredicted = np.around(np.array(xFatPredicted), 0)
        xFatPredicted = xFatPredicted.astype(int)
        yFatPredicted = np.around(np.array(yFatPredicted), 0)
        yFatPredicted = yFatPredicted.astype(int)

        # fill mask with fat thickness points
        predictedFatArray[xFatPredicted, yFatPredicted] = 255
        predictedFatArray = predictedFatArray.astype('uint8')
        originalarray = originalarray.astype('uint8')
        dilatedarray = cv2.bitwise_or(predictedFatArray, originalarray)

        filldilated = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth, 3))
        filldilated[:, :, 1] = dilatedarray[:, :]
        filldilated = filldilated.astype('uint8')
        predictedFatArray = floodFillContour(filldilated)[:, :, 0]

        # subtract inside contour from both fat contours
        mriFatArray = mriFatArray - originalarray
        predictedFatArray = predictedFatArray - originalarray

        overlap = cv2.bitwise_and(mriFatArray, predictedFatArray)
        overlap[overlap < 0] = 0
        overlap[overlap > 0] = 255

        mriFatArray[mriFatArray > 0] = 255
        predictedFatArray[predictedFatArray > 0] = 255

        overlapSum = np.count_nonzero(overlap)
        mriSum = np.count_nonzero(mriFatArray)
        predictedSum = np.count_nonzero(predictedFatArray)

        dsc = (2 * overlapSum) / (mriSum + predictedSum)

        if not self.autoLoadFlag:
            QMessageBox().about(self, 'DSC', f'DSC of fat map: {dsc}')
        else:
            self.fatDSCList.append(dsc)

        overlapIm = np.zeros((self.bmodeImageHeight, self.bmodeImageWidth, 3))
        overlapIm[:, :, 0] = overlap
        self.sliceWidgetBMode.fatOverlapIm = overlapIm

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonFatMapMSE_clicked(self):
        """Calculate mean squared error of predicted fat map compared
        to ground truth fat map from MRI. Both fat maps must be loaded in US window"""

        if len(self.sliceWidgetBMode.bModeFatMap) == 0:
            print('Load fat interpolation first!')

        if len(self.sliceWidgetBMode.bModeFatPoints) == 0:
            MainWindow.on_pushButtonBinaryFat_clicked(self)

        # actual fat thickness in mm
        fatThickness = np.array(self.fatPoints)

        # predicted fat thickness in mm
        fatThicknessPredicted = np.array(self.predictedThickness)

        diff = 0
        r2_denom = 0
        denom_mean = np.mean(fatThickness)
        for i in range(len(fatThickness)):
            diff += (fatThickness[i] - fatThicknessPredicted[i]) ** 2
            r2_denom += (fatThickness[i] - denom_mean) ** 2
        MSE = diff / len(fatThickness)
        r2 = 1 - diff / r2_denom

        if not self.autoLoadFlag:
            QMessageBox().about(self, 'Mean Squared Error and R2', f'MSE of fat map: {MSE}\nR2 of fat map: {r2}')
        else:
            self.fatMSEList.append(MSE)

    @pyqtSlot()
    def on_pushButtonFatMapEvalAll_clicked(self):
        """get DSC and MSE for all testing cases for specific model"""

        fileNameFullPathModel, filterReturn = FileDialog.getOpenFileName(self, 'Select Model',
                                                                         self.defaultOpenPath, '*.pkl')
        if not fileNameFullPathModel:
            return

        directoryList = fileNameFullPathModel.split('/')
        directory = '/'.join(directoryList[:-1])

        txtPath = f'{directory}/{directoryList[-2]}.txt'

        f = open(txtPath, 'r').read()

        # get testing cases from file
        # testing cases listed after two tabs - first two in list will be headers
        cases = f.split('\t\t')[2:]

        # list contains testing case followed by \n
        for i in range(len(cases)):
            cases[i] = cases[i].split('\n')[0]

        # set ultrasound and mri to load from saved path rather than ask
        self.autoLoadFlag = 1

        # open mri window
        mriWindow = gui.mriWindow.MRIWindow()
        mriWindow.show()

        # read excel file
        df = pd.read_excel(rf'{self.dataTopLevelPath}\us_mri_match_shift_amts.xlsx')

        caseList = df['Case'].tolist()
        mriSlices = df['MRI Slice'].tolist()
        mriShift = df['MRI Shift'].tolist()
        usShift = df['US Shift'].tolist()
        mriRotation = df['MRI Rotate'].tolist()

        caseName = [x.split('P')[0] for x in cases]
        prePost = [('P' + x.split('P')[1]).split('_')[0] for x in cases]
        caseNumbers = [x.split('_')[1] for x in cases]
        frameType = [x.split('_')[2] for x in cases]

        # set path to classifier model
        self.modelPath = fileNameFullPathModel

        acc = []
        sens = []
        spec = []

        # loop thru testing cases
        for n in range(len(cases)):

            # clear image
            MainWindow.on_pushButtonClearContourMorph_clicked(self)
            MainWindow.on_pushButtonClearAll_clicked(self)

            # get all digits that are in the name of the case
            caseDigits = re.findall(r'\d', cases[n])

            # find index of current case in excel file
            ind = caseList.index(cases[n])

            # mri slice, mri shift, us shift, and mri rotation of current case
            mriCurrentSlice = mriSlices[ind]
            mriCurrentShift = mriShift[ind]
            usCurrentShift = usShift[ind]
            mriCurrentRotation = mriRotation[ind]

            if frameType[n] == 'ED':
                frame = 'EndDiastole'
            else:
                frame = 'EndSystole'

            # load ultrasound and mri image
            if prePost[n] == 'POST2':
                self.iqPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n][:-1].capitalize()}-Training2\IQ\{caseName[n]}{prePost[n]}_{caseNumbers[n]}.iq'
            else:
                self.iqPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n].capitalize()}-Training\IQ\{caseName[n]}{prePost[n]}_{caseNumbers[n]}.iq'
            self.mriPath = rf'{self.mriDataPath}\PATS_MF03_cMRI_Results_Comparison\MF{caseDigits[0]}{caseDigits[1]}{caseDigits[2]}{caseDigits[3]}-{prePost[n]}\{caseDigits[1]}{caseDigits[2]}{caseDigits[3]}-{prePost[n]}-{frame}-Cropped-Segmentation-JK.seg.nrrd'

            MainWindow.on_pushButtonLoadIQ_clicked(self)
            mriWindow.on_pushButtonLoadImage_clicked()

            if frameType[n] == 'ED':
                MainWindow.on_pushButtonGoToED_clicked(self)
                frameNum = self.EDFrame
            else:
                MainWindow.on_pushButtonGoToES_clicked(self)
                frameNum = self.ESFrame

            mriWindow.on_horizontalSliderMRISlice_valueChanged(mriCurrentSlice - self.mriWindow.get_segment_offset())

            mriWindow.spinBoxShift.setValue(mriCurrentShift)
            self.spinBoxShift.setValue(usCurrentShift)

            if mriCurrentRotation == 'CW':
                mriWindow.on_pushButtonRotateCW_clicked()
            elif mriCurrentRotation == 'CCW':
                mriWindow.on_pushButtonRotateCCW_clicked()

            # load MRI Bspline
            mriWindow.on_pushButtonGetOutsidePoints_clicked()

            if prePost[n] == 'POST2':
                self.contourPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n][:-1].capitalize()}-Training2\Traced Contour\{frame}\{cases[n]}_{frameNum}.nrrd'
            else:
                self.contourPath = rf'{self.usDataPath}\{caseName[n]}\{prePost[n].capitalize()}-Training\Traced Contour\{frame}\{cases[n]}_{frameNum}.nrrd'

            # load US Bspline
            MainWindow.on_pushButtonLoadPoints_clicked(self)
            MainWindow.on_pushButtonGetOutsidePoints_clicked(self)

            # predict fat and draw fat contour
            MainWindow.on_pushButtonPredict_clicked(self)
            MainWindow.on_pushButtonInterpolateFat_clicked(self)

            MainWindow.on_pushButtonFatMapDSC_clicked(self)
            MainWindow.on_pushButtonFatMapMSE_clicked(self)

            acc.append(self.classifierAcc)
            sens.append(self.classifierSens)
            spec.append(self.classifierSpec)

        self.autoLoadFlag = 1

        # get folder of classifier model
        modelFolder = self.modelPath.split('/')
        modelFolder = '/'.join(modelFolder[:-1])

        # save training and testing cases to text file
        with open(fr'{modelFolder}\classifier_DSE_MSE.txt', 'w') as f:
            for i in range(len(cases)):
                f.write(
                    f'Case: {cases[i]}\n\tDSC: {self.fatDSCList[i]}\n\tMSE: {self.fatMSEList[i]}\n\tAccuracy: {acc[i]}\n\tSensitivity: {sens[i]}\n\tSpecificity: {spec[i]}\n')
            f.close()

    @pyqtSlot()
    def on_pushButtonUNetClassifier_clicked(self):
        """Classify ROIs using CNN classifier"""
        self.classifierTypeDL = self.deepLearningListWidget.selectedItems()[0].text()
        lossFunction = self.lossFunctionListWidget.selectedItems()[0].text()

        alpha = self.get_ftl_alpha()
        gamma = self.get_ftl_gamma()

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        verbose = self.DLVerbose

        # initial_learning_rate = 0.000025
        initial_learning_rate = self.DLLearningRate
        learning_rate_patience = self.DLLearningRatePatience
        early_stopping_patience = 50
        learning_rate_drop = self.DLLearningRateDrop

        buffer_size = 100
        batch_size = self.DLBatchSize

        psd_type = 'yule'
        roi_size = self.sliceWidgetIQ.ROIChannelNum

        if self.modelSaveFlag:
            modelPath = rf'{self.classifierModelsPath}\{timestamp}'
        else:
            # modelPath = self.modelSaveFolder
            modelPath = rf'{self.classifierModelsPath}\{timestamp}'

        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

        fat1 = str(binaryFatThreshmm).split('.')[0]
        fat2 = str(binaryFatThreshmm).split('.')[1]
        modelName = f'{self.classifierTypeDL}_fat_{fat1}_{fat2}.hdf5'
        folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\thickness_{self.sliceWidgetBMode.ContourThickness}\{psd_type}\Channel\{roi_size}'

        if not os.path.exists(folderPath):
            print(f'{folderPath} does not exist')
            return

        # compute initial bias
        # num_pos = 777
        # num_neg = 3903
        # bias = np.log([num_pos/num_neg])

        trainData, validationData, testData, train_cases, validation_cases, test_cases, num_params, numTrain, \
            numVal, numTest = generate_data(folderPath, roi_size, shuffle=self.classifierShuffleFlag)

        trainData = trainData.shuffle(buffer_size).repeat().batch(batch_size)
        validationData = validationData.batch(batch_size)
        # testData = testData.shuffle(buffer_size).batch(batch_size)

        steps_per_epoch = numTrain // batch_size
        validation_steps = numVal // batch_size

        print(f'\nClassifier Type: {self.classifierTypeDL}\nLoss function: {lossFunction}\n')

        if self.classifierTypeDL == "VuNet1D":
            model = uNetModel1D(num_filters=128, kernel=5, input_size=(64, 12), binary=True, num_classes=1,
                                dropout_rate=0.5)

            initial_learning_rate = 1.893171814906959e-05
            learning_rate_patience = 6.0
            early_stopping_patience = 30
            learning_rate_drop = 0.2

            # lossFunction = 'Tversky'
            alpha = 0.7
            gamma = 1.5

        elif self.classifierTypeDL == "SimpleUNet1D":
            model = simpleModel1D(num_filters=32, input_size=(64, 12))

            # initial_learning_rate = 4.615794009633788e-07
            initial_learning_rate = 0.00300500155092942
            learning_rate_patience = 13.0
            early_stopping_patience = 30
            learning_rate_drop = 0.3

            # lossFunction = 'Tversky'
            alpha = 0.7
            gamma = 1

        elif self.classifierTypeDL == "VuNet1D_tune":

            tuner = kt.Hyperband(
                hypermodel=uNetModel1D_tune,
                objective="val_accuracy",
                max_epochs=200,
                factor=3,
                hyperband_iterations=3,
                overwrite=True,
                project_name="VuNet1D_tune"
            )

            tuner.search_space_summary()


            tuner.search(trainData, validation_data=validationData, epochs=100, steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps, shuffle=True, verbose=1)

            model = tuner.get_best_models(num_models=1)

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            print(f'\nCLASSIFIER TYPE: {self.classifierTypeDL}\n')
            print(f'\n BEST HYPERPARAMETERS:\n Kernel: {best_hps.get("kernel")}\nNumber of filters: {best_hps.get("num_filters")}\n'
                  f'Dropout Rate: {best_hps.get("dropout_rate")}\nLearning Rate: {best_hps.get("lr")}\n'
                  f'Learning Rate Drop: {best_hps.get("lr_drop")}\nLearning Rate Patience: {best_hps.get("lr_patience")}\n')
        elif self.classifierTypeDL == "SimpleUNet1D_tune":

            tuner = kt.Hyperband(
                hypermodel=simpleModel1D_tune,
                objective="val_accuracy",
                max_epochs=200,
                factor=3,
                hyperband_iterations=3,
                overwrite=True,
                project_name="SimpleUNet1D_tune"
            )

            tuner.search_space_summary()

            tuner.search(trainData, validation_data=validationData, epochs=100, steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps, shuffle=True, verbose=1)

            model = tuner.get_best_models(num_models=1)

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            print(f'\nCLASSIFIER TYPE: {self.classifierTypeDL}\n')
            print(f'\n BEST HYPERPARAMETERS:\n Number of Filters: {best_hps.get("num_filters")}\nKernel 1: {best_hps.get("kernel_1")}\nKernel 2: {best_hps.get("kernel_2")}\n'
                  f'Kernel 3: {best_hps.get("kernel_3")}\nKernel 4: {best_hps.get("kernel_4")}\nLearning Rate: {best_hps.get("lr")}\n'
                  f'Learning Rate Drop: {best_hps.get("lr_drop")}\nLearning Rate Patience: {best_hps.get("lr_patience")}\n')

        elif self.classifierTypeDL == "SimpleRNN_tune":

            tuner = kt.Hyperband(
                hypermodel=simpleRNN_tune,
                objective="val_accuracy",
                max_epochs=200,
                factor=3,
                hyperband_iterations=3,
                overwrite=True,
                project_name="SimpleRNN_tune"
            )

            tuner.search_space_summary()

            tuner.search(trainData, validation_data=validationData, epochs=100, steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps, shuffle=True, verbose=1)

            model = tuner.get_best_models(num_models=1)

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            print(f'\nCLASSIFIER TYPE: {self.classifierTypeDL}\n')
            print(f'\n\tBEST HYPERPARAMETERS:\nFilters 1: {best_hps.get("filters_1")}\n'
                  f'Dropout: {best_hps.get("dropout")}\nRecurrent Dropout: {best_hps.get("recurrent_dropout")}\nLearning Rate: {best_hps.get("lr")}\n'
                  f'Learning Rate Drop: {best_hps.get("lr_drop")}\nLearning Rate Patience: {best_hps.get("lr_patience")}\n')


        else:
            model = simpleRNN(input_size=(64, 12))

            initial_learning_rate = 1.537102369204225e-05
            learning_rate_patience = 6.0
            early_stopping_patience = 30
            learning_rate_drop = 0.1

            alpha = 0.5
            gamma = 2

            # lossFunction = 'Weighted BCE'

        callbacks = get_callbacks(f'{modelPath}/{modelName}', initial_learning_rate=initial_learning_rate,
                                  learning_rate_patience=learning_rate_patience,
                                  early_stopping_patience=early_stopping_patience,
                                  learning_rate_drop=learning_rate_drop, ftlalpha=alpha, ftlgamma=gamma)

        plot = True
        if self.modelSaveFlag == 0:
            plot = False

        if lossFunction=='BCE' or lossFunction=='Undersampling' or lossFunction=='Oversampling':
            lossFunction = tf.keras.losses.binary_crossentropy
        elif lossFunction=='Tversky':
            lossFunction = tversky_loss
        elif lossFunction=='Weighted BCE':
            lossFunction = weighted_binary_crossentropy
        else:
            print('\nLoss function not recognized\n')
            lossFunction = core.CNNClassifier.weighted_binary_crossentropy

        metrics = trainModel(model, trainData, validationData,
                   steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                   initial_learning_rate=initial_learning_rate, validation_steps=validation_steps, binary=True,
                   num_epochs=200, plot=plot, verbose=verbose, lossFunc=lossFunction)

        acc = metrics['accuracy'][-1]
        sens = metrics['sensitivity'][-1]
        spec = metrics['specificity'][-1]
        val_acc = metrics['val_accuracy'][-1]
        val_sens = metrics['val_sensitivity'][-1]
        val_spec = metrics['val_specificity'][-1]

        # self.CNNClassifierAcc = val_acc
        # self.CNNClassifierSens = val_sens
        # self.CNNClassifierSpec = val_spec
        # self.CNNClassifierPrec = metrics['val_precision'][-1]
        # self.CNNClassifierRec = metrics['val_recall'][-1]

        # ask to save model - if not delete directory
        if self.modelSaveFlag:
            msgB = QMessageBox()
            val = msgB.question(self, 'Results',
                                f'Model Results\n\nAccuracy: {acc}\nSensitivity: {sens}\nSpecificity:'
                                f' {spec}\n\nValidation Accuracy: {val_acc}\nValidation Sensitivity: {val_sens}'
                                f'\nValidation Specificity: {val_spec}\n\n\nSave?\n',
                                msgB.Yes | msgB.No)

            if val == msgB.Yes:
                if not os.path.exists(modelPath):
                    os.makedirs(modelPath)
                # save training and testing cases to text file
                with open(
                        fr'{modelPath}\{modelName}.txt',
                        'w') as f:
                    f.write('Training Cases \t\t Validation Cases \t\t Testing Cases\n')
                    for i in range(len(train_cases)):
                        if i < len(test_cases) and i < len(validation_cases):
                            if len(train_cases[i]) > 15:
                                f.write(f'{train_cases[i]}\t{validation_cases[i]}\t\t{test_cases[i]}\n')
                            else:
                                f.write(f'{train_cases[i]}\t\t{validation_cases[i]}\t\t{test_cases[i]}\n')
                        elif i < len(validation_cases):
                            if len(train_cases[i]) > 15:
                                f.write(f'{train_cases[i]}\t{validation_cases[i]}\n')
                            else:
                                f.write(f'{train_cases[i]}\t\t{validation_cases[i]}\n')
                        elif i < len(test_cases):
                            if len(train_cases[i]) > 15:
                                f.write(f'{train_cases[i]}\t{test_cases[i]}\n')
                            else:
                                f.write(f'{train_cases[i]}\t\t{test_cases[i]}\n')
                        else:
                            f.write(f'{train_cases[i]}\n')
                    f.write(f'\n\nAccuracy: {acc}\nSensitivity: {sens}\nSpecificity:'
                            f' {spec}\n\nValidation Accuracy: {val_acc}\nValidation Sensitivity: {val_sens}'
                            f'\nValidation Specificity: {val_spec}\n')
                    f.close()

                model.save(f'{modelPath}/{modelName}', overwrite=True)

        else:
            if not os.path.exists(modelPath):
                os.makedirs(modelPath)
            # save training and testing cases to text file
            with open(
                    fr'{modelPath}\{modelName}.txt',
                    'w') as f:
                f.write('Training Cases \t\t Validation Cases \t\t Testing Cases\n')
                for i in range(len(train_cases)):
                    if i < len(test_cases) and i < len(validation_cases):
                        if len(train_cases[i]) > 15:
                            f.write(f'{train_cases[i]}\t{validation_cases[i]}\t\t{test_cases[i]}\n')
                        else:
                            f.write(f'{train_cases[i]}\t\t{validation_cases[i]}\t\t{test_cases[i]}\n')
                    elif i < len(validation_cases):
                        if len(train_cases[i]) > 15:
                            f.write(f'{train_cases[i]}\t{validation_cases[i]}\n')
                        else:
                            f.write(f'{train_cases[i]}\t\t{validation_cases[i]}\n')
                    elif i < len(test_cases):
                        if len(train_cases[i]) > 15:
                            f.write(f'{train_cases[i]}\t{test_cases[i]}\n')
                        else:
                            f.write(f'{train_cases[i]}\t\t{test_cases[i]}\n')
                    else:
                        f.write(f'{train_cases[i]}\n')
                f.write(f'\n\nAccuracy: {acc}\nSensitivity: {sens}\nSpecificity:'
                        f' {spec}\n\nValidation Accuracy: {val_acc}\nValidation Sensitivity: {val_sens}'
                        f'\nValidation Specificity: {val_spec}\n')
                f.close()

            model.save(f'{modelPath}/{modelName}', overwrite=True)

            self.modelPath = f'{modelPath}/{modelName}'

            self.on_pushButtonCNNClassifyTesting_clicked()

            if self.saveCNNFlag == 0:
                shutil.rmtree(modelPath)


    @pyqtSlot()
    def on_pushButtonUNetPredict_clicked(self):
        """Classify ROIs of currently loaded case using CNN"""
        self.tissueTypeList.clear()
        self.sliceWidgetBMode.tissueTypeList.clear()
        self.sliceWidgetIQ.tissueTypeList.clear()
        self.sliceWidgetBMode.correctList.clear()

        if not self.autoLoadFlag:
            fileNameFullPathModel, filterReturn = FileDialog.getOpenFileName(self, 'Select Model',
                                                                             self.defaultOpenPath, '*.hdf5')
            if not fileNameFullPathModel:
                return
        else:
            fileNameFullPathModel = self.modelPath

        # get case name, roi size, and channel thickness currently set
        CASE = f'{self._iqFileName}_E{self.mriWindow.get_frame_flag()}'
        roiSize = self.sliceWidgetBMode.ROIChannelNum
        thickness = self.sliceWidgetBMode.ContourThickness
        batch_size = 4

        # get fat threshold thickness
        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

        fat1 = str(binaryFatThreshmm).split('.')[0]
        fat2 = str(binaryFatThreshmm).split('.')[1]

        # path to spectra and rois
        folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\thickness_{thickness}\yule\Channel\{roiSize}\{CASE}_lists'
        roiPath = fr'{self.SAVEPATH}\ROI_masks\channel\{roiSize}\{CASE[:-2]}Channel_{CASE[-2:]}'

        # get spectra and add to array
        n = 0
        params = None
        labelList = []
        # should be same number of files as roi size
        while n < int(roiSize):
            param = f'norm_param_list_{roiSize}_{n}'
            currParams = np.load(fr'{folderPath}\{param}.npy')

            num_params = np.shape(currParams)[0]

            if np.all(currParams == 0):
                n += 1
                continue

            if n == 0:
                params = np.zeros(np.shape(currParams)[0])
                params[:] = currParams[:]

                labels = []
            else:
                if params is None:
                    params = np.zeros(np.shape(currParams)[0])
                    params[:] = currParams[:]

                    labels = []
                else:
                    params = np.row_stack((params, currParams))

            # load tissue type labels and add to list
            currLabel = np.load(fr'{folderPath}\labels_{roiSize}_{n}.npy')
            labels = np.append(labels, currLabel)
            labelList = np.append(labelList, currLabel)

            n += 1

        params = np.array(params)
        labels = np.array(labels)

        iqROIs = []
        bModeROIs = []

        # get rois
        for n, roiFile in enumerate(sorted(os.listdir(roiPath), key=lambda s: int(re.findall(r'\d+', s)[4]))):
            param = f'norm_param_list_{roiSize}_{n}'
            currParams = np.load(fr'{folderPath}\{param}.npy')

            if np.all(currParams == 0):
                n += 1
                continue
            # cleaning up file name
            self._roiFileNamePath = util.cleanPath(rf'{roiPath}\{roiFile}')
            self._roiFileName = util.splitext(os.path.basename(self._roiFileNamePath))[0]

            roiMaskIQ = np.load(self._roiFileNamePath)

            roiMaskBMode = convert_mask_from_polar_to_cartesian(np.flipud(roiMaskIQ))
            roiMaskBMode = np.flipud(roiMaskBMode)
            roiMaskBMode = roiMaskBMode.astype('uint8')

            roiEdgeIQ = cv2.Canny(image=roiMaskIQ, threshold1=100, threshold2=200)
            kernel = np.ones((2, 2))
            roiEdgeIQ = cv2.dilate(roiEdgeIQ, kernel)
            roiEdgeIQ = np.ma.masked_where(roiEdgeIQ == 0, roiEdgeIQ)
            iqROIs.append(roiEdgeIQ)

            roiEdgeBMode = cv2.Canny(image=roiMaskBMode, threshold1=100, threshold2=200)
            kernel = np.ones((2, 2))
            roiEdgeBMode = cv2.dilate(roiEdgeBMode, kernel)
            roiEdgeBMode = np.ma.masked_where(roiEdgeBMode == 0, roiEdgeBMode)
            bModeROIs.append(roiEdgeBMode)

        # remove all 0 spectra and labels
        params = params[~np.all(params == 0, axis=1)]
        labels = [i for i in labels if i != 0]

        params = standardize_features([params])
        params = params[0]

        params = np.array(params)
        labels = np.array(labels)

        labels[labels == 4] = 0
        labelsLen = len(labels)
        indices = [*range(0, labelsLen)]
        indices = np.array(indices)

        labels_original = np.copy(labels)

        params = params.take(range(-32, 32), mode='wrap', axis=0)
        labels = labels.take(range(-32, 32), mode='wrap', axis=0)
        indices = indices.take(range(-32, 32), mode='wrap', axis=0)

        data = tf.data.Dataset.from_tensor_slices(([params], [labels]))
        data = data.batch(batch_size)

        model = keras.models.load_model(fileNameFullPathModel, compile=False)
        pred = model.predict(data)[0]

        self.predictedLabels = np.zeros(labelsLen)

        for i, ind in enumerate(indices):
            if np.count_nonzero(self.predictedLabels) == labelsLen:
                break

            self.predictedLabels[ind] = pred[i][0]

        # threshold predictions
        self.predictedLabels[self.predictedLabels >= 0.5] = 1
        self.predictedLabels[self.predictedLabels < 0.5] = 0

        total = 0
        correct = 0
        correctList = []
        # plot each roi with its respective tissue type - 1 = fat (red), 0 = nonfat (white)
        for ii in range(len(self.predictedLabels)):
            self.tissueType = self.predictedLabels[ii]
            self.sliceWidgetIQ.IQROIMasks.append(iqROIs[ii])
            self.sliceWidgetBMode.BModeROIMasks.append(bModeROIs[ii])

            self.tissueTypeList.append(self.tissueType)
            self.sliceWidgetIQ.tissueTypeList.append(self.tissueType)
            self.sliceWidgetBMode.tissueTypeList.append(self.tissueType)

            total += 1

            if self.predictedLabels[ii] == labels_original[ii]:
                correct += 1
                self.sliceWidgetBMode.correctList.append(1)
            else:
                self.sliceWidgetBMode.correctList.append(0)

        print(self._roiFileName)
        print(f"\nAccuracy: {correct/total}\n")

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                        self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        # if not self.autoLoadFlag:
        #     # display success metrics in GUI popup box
        #     msgB = QMessageBox()
        #     val = msgB.question(self, 'Results', f'Accuracy: {success.fat_acc}\nSensitivity: {success.fat_sens}\nSpecificity: {success.fat_spec}\nYouden\'s Index: {success.fat_youdens}\n',
        #                         msgB.Ok)
        #
        # else:
        #     self.classifierAcc = success.fat_acc
        #     self.classifierSens = success.fat_sens
        #     self.classifierSpec = success.fat_spec

    @pyqtSlot()
    def on_pushButtonCNNClassifyTraining_clicked(self):
        prob_thresh = 0.5

        if not self.autoLoadFlag:
            fileNameFullPathModel, filterReturn = FileDialog.getOpenFileName(self, 'Select Model',
                                                                             self.defaultOpenPath, '*.hdf5')
            if not fileNameFullPathModel:
                return
        else:
            fileNameFullPathModel = self.modelPath

        folderPathList = fileNameFullPathModel.split('/')[:-1]
        modelFolderPath = '/'.join(folderPathList)

        txtFilePath = f'{modelFolderPath}/UNET.hdf5.txt'

        type = 'Channel'
        roiSize = self.sliceWidgetIQ.ROIChannelNum
        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

        fat1 = str(binaryFatThreshmm).split('.')[0]
        fat2 = str(binaryFatThreshmm).split('.')[1]
        batch_size = 4

        # get training cases from .txt file
        with open(txtFilePath) as f:
            file = f.read()
            f.close()

        fLines = file.split('\n')[1:-10]

        for i, line in enumerate(fLines):
            loc = line.find('\t')
            if loc != -1:
                fLines[i] = line[:loc]

        labelList = []
        predList = []
        labelProbs = []

        for i, case in enumerate(fLines):
            if case.find('MF') == -1:
                break

            folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\thickness_{self.sliceWidgetBMode.ContourThickness}\yule\Channel\{roiSize}\{case}_lists'

            n = 0
            while n < int(roiSize):
                param = f'norm_param_list_{roiSize}_{n}'
                currParams = np.load(fr'{folderPath}\{param}.npy')

                num_params = np.shape(currParams)[0]

                if np.all(currParams == 0):
                    n += 1
                    continue

                if n == 0:
                    params = np.zeros(np.shape(currParams)[0])
                    params[:] = currParams[:]

                    labels = []
                else:
                    params = np.row_stack((params, currParams))

                # load tissue type labels and add to list
                currLabel = np.load(fr'{folderPath}\labels_{roiSize}_{n}.npy')
                labels = np.append(labels, currLabel)
                labelList = np.append(labelList, currLabel)

                n += 1

            params = np.array(params)
            labels = np.array(labels)

            params = standardize_features([params])
            params = params[0]

            # remove all 0 spectra and labels
            params = params[~np.all(params == 0, axis=1)]
            labels = [i for i in labels if i != 0]

            params = np.array(params)
            labels = np.array(labels)

            labels[labels == 4] = 0
            labelsLen = len(labels)
            indices = [*range(0, labelsLen)]
            indices = np.array(indices)

            params = params.take(range(-32, 32), mode='wrap', axis=0)
            labels = labels.take(range(-32, 32), mode='wrap', axis=0)
            indices = indices.take(range(-32, 32), mode='wrap', axis=0)

            data = tf.data.Dataset.from_tensor_slices(([params], [labels]))
            data = data.batch(batch_size)

            model = simpleModel1D(num_filters=32, input_size=(64, num_params), binary=True, num_classes=1,
                                  dropout_rate=0.4, dilation_rate=4)
            model.load_weights(fileNameFullPathModel)
            pred = model.predict(data)[0]

            self.predictedLabels = np.zeros(labelsLen)

            for i, ind in enumerate(indices):
                if np.count_nonzero(self.predictedLabels) == labelsLen:
                    break

                self.predictedLabels[ind] = pred[i][0]

            # save probabilities
            for prob in self.predictedLabels:
                labelProbs.append(prob)

            # threshold predictions
            self.predictedLabels[self.predictedLabels >= 0.5] = 1
            self.predictedLabels[self.predictedLabels < 0.5] = 0

            # save probabilities
            for prob in self.predictedLabels:
                predList.append(prob)

        labelList[labelList == 4] = 0

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i, label in enumerate(labelList):
            if labelProbs[i] < prob_thresh and (1 - labelProbs[i]) < prob_thresh:
                continue

            # is fat
            if label == 1:
                # guess fat
                if predList[i] == 1:
                    TP += 1
                # guess not fat
                else:
                    FN += 1
            # is not fat
            else:
                # guess nonfat
                if predList[i] == 0:
                    TN += 1
                # guess fat
                else:
                    FP += 1

        fat_sensitivity = TP / (TP + FN)

        fat_specificity = TN / (TN + FP)

        fat_accuracy = (TP + TN) / (TP + TN + FP + FN)

        fat_Youdens = fat_sensitivity + fat_specificity - 1

        # save training and testing cases to text file
        with open(fr'{modelFolderPath}\TrainingResults_thresh{prob_thresh}.txt', 'w') as f:
            f.write(
                f'Training Data Results:\n\nAccuracy: {fat_accuracy}\nSensitivity: {fat_sensitivity}\nSpecificity: {fat_specificity}\n\nProbability Threshold: {prob_thresh}')
            f.close()

        print('\nDone')

    @pyqtSlot()
    def on_pushButtonCNNClassifyTesting_clicked(self):

        if self.modelSaveFlag:
            fileNameFullPathModel, filterReturn = FileDialog.getOpenFileName(self, 'Select Model',
                                                                             self.defaultOpenPath, '*.hdf5')
            if not fileNameFullPathModel:
                return
        else:
            fileNameFullPathModel = self.modelPath

        folderPathList = fileNameFullPathModel.split('/')[:-1]
        modelFolderPath = '/'.join(folderPathList)

        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()
        fat1 = str(binaryFatThreshmm).split('.')[0]
        fat2 = str(binaryFatThreshmm).split('.')[1]

        txtFilePath = f'{fileNameFullPathModel}.txt'

        type = 'Channel'
        roiSize = self.sliceWidgetIQ.ROIChannelNum

        batch_size = 2

        folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\thickness_{self.sliceWidgetBMode.ContourThickness}\yule\Channel\{roiSize}'

        model = keras.models.load_model(fileNameFullPathModel, compile=False)

        fat_accuracy, fat_sensitivity, fat_specificity, fat_precision, fat_recall\
            = classify_testing(model,
                            txtFilePath,
                            folderPath,
                            roiSize,
                            batch_size)

        self.CNNClassifierAcc = fat_accuracy
        self.CNNClassifierSens = fat_sensitivity
        self.CNNClassifierSpec = fat_specificity
        self.CNNClassifierPrec = fat_precision
        self.CNNClassifierRec = fat_recall

        # save training and testing cases to text file
        with open(fr'{modelFolderPath}\TestingResults.txt', 'w') as f:
            f.write(
                f'Testing Data Results:\n\nAccuracy: {fat_accuracy}\nSensitivity: {fat_sensitivity}\nSpecificity: {fat_specificity}')
            f.close()

        print('\nDone')

    @pyqtSlot()
    def on_pushButtonCNNClassifierTest_clicked(self):
        """
        Rerandomize data and create classifier number of times specified by iterations variable.
        Calculate average accuracy, sensitivity, and specificity
        """
        acc = []
        sens = []
        spec = []
        prec = []
        rec = []
        fscore = []
        iterations = 10

        self.modelSaveFlag = 0
        for i in range(iterations):
            MainWindow.on_pushButtonUNetClassifier_clicked(self)
            acc.append(self.CNNClassifierAcc)
            sens.append(self.CNNClassifierSens)
            spec.append(self.CNNClassifierSpec)
            prec.append(self.CNNClassifierPrec)
            rec.append(self.CNNClassifierRec)
            print(f'\nDone with {i + 1} out of {iterations}')

            K.clear_session()

        self.modelSaveFlag = 1

        accuracy = sum(acc) / len(acc)
        sensitivity = sum(sens) / len(sens)
        specificity = sum(spec) / len(spec)
        precision = sum(prec) / len(prec)
        recall = sum(rec) / len(rec)

        self.accAvg = accuracy
        self.sensAvg = sensitivity
        self.specAvg = specificity
        self.precAvg = precision
        self.recAvg = recall

        self.accSpread = max(acc) - min(acc)
        self.sensSpread = max(sens) - min(sens)
        self.specSpread = max(spec) - min(spec)
        self.precSpread = max(prec) - min(prec)
        self.recSpread = max(rec) - min(rec)

        self.accStd = np.std(acc)
        self.sensStd = np.std(sens)
        self.specStd = np.std(spec)
        self.precStd = np.std(prec)
        self.recStd = np.std(rec)

        print(
            f'\nTesting Data Results\n\nAccuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\n')

    @pyqtSlot()
    def on_pushButtonRunCNNClassifierAll_clicked(self):
        # run classifier on all data and save to excel sheet

        # set up timestamp for file name
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        thicknessList = self.thicknessList
        roiSizeList = self.roiNumList
        fatThreshList = self.fatThreshList
        roiThreshList = self.roiAvgThreshList
        probThreshList = self.probThreshList
        learningRateList = self.learningRateList
        learningRatePatienceList = self.learningRatePatienceList
        learningRateDropList = self.learningRateDropList
        batchSizeList = self.batchSizeList

        clfs = ['VuNet1D', 'SimpleUNet1D', 'SimpleRNN']     # list of all classifiers
        clfList = ['SimpleUNet1D']  # list of classifier to be used
        losses = ['BCE', 'Tversky', 'Weighted BCE']         # list of all losses
        lossList = ['Weighted BCE']       # list of all losses to be used

        FTL_alpha = [0.5]
        FTL_gamma = [1.5]

        dfList = []
        sheetList = []

        # don't want dialogue box asking to save each model
        self.modelSaveFlag = 0
        self.saveCNNFlag = 1

        it = 0

        num_its = len(clfList) * len(probThreshList) * len(fatThreshList) * len(thicknessList) * len(roiSizeList) * \
                  len(roiThreshList) * len(learningRateList) * len(learningRatePatienceList) * len(
            learningRateDropList) * \
                  len(batchSizeList) * len(lossList)

        print(f'\nNUMBER OF ITERATIONS: {num_its}')

        # MainWindow.on_pushButtonGetAllData_clicked(self)

        for clf in clfList:
            for prob_thresh in probThreshList:
                for fatThresh in fatThreshList:
                    for thickness in thicknessList:
                        for roiNum in roiSizeList:
                            for roiThresh in roiThreshList:
                                for lr in learningRateList:
                                    for lrPatience in learningRatePatienceList:
                                        for lrDrop in learningRateDropList:
                                            for batch in batchSizeList:
                                                for loss in lossList:
                                                    for alpha in FTL_alpha:
                                                        for gamma in FTL_gamma:
                                                            clfInd = clfs.index(clf)
                                                            lossInd = losses.index(loss)
                                                            self.deepLearningListWidget.setCurrentRow(clfInd)
                                                            self.lossFunctionListWidget.setCurrentRow(lossInd)

                                                            self.doubleSpinBoxFTLAlpha.setValue(alpha)
                                                            self.doubleSpinBoxFTLGamma.setValue(gamma)

                                                            self.on_doubleSpinBoxBinFatThresh_valueChanged(fatThresh)
                                                            self.on_ContourThickness_valueChanged(thickness)
                                                            self.on_ROIChannelNum_valueChanged(roiNum)
                                                            self.on_spinBoxROIAvgThresh_valueChanged(roiThresh)

                                                            self.classifierTypeDL = self.deepLearningListWidget.selectedItems()[0].text()
                                                            self.DLLearningRate = lr
                                                            self.DLLearningRatePatience = lrPatience
                                                            self.DLLearningRateDrop = lrDrop
                                                            self.DLBatchSize = batch

                                                            self.prob_thresh = prob_thresh

                                                            # MainWindow.on_pushButtonComputeSpectra_clicked(self)
                                                            MainWindow.on_pushButtonCNNClassifierTest_clicked(self)

                                                            df = pd.DataFrame([[self.accAvg, self.accSpread, self.accStd],
                                                                               [self.sensAvg, self.sensSpread, self.sensStd],
                                                                               [self.specAvg, self.specSpread, self.specStd],
                                                                               [self.precAvg, self.precSpread, self.precStd],
                                                                               [self.recAvg, self.recSpread, self.recStd],
                                                                               [lr],
                                                                               [lrPatience],
                                                                               [lrDrop],
                                                                               [batch]],
                                                                              index=['Accuracy', 'Sensitivity', 'Specificity',
                                                                                     'Precision', 'Recall',
                                                                                     'lr', 'lr Patience', 'lr Drop', 'Batch Size'],
                                                                              columns=['Value', 'Spread', 'Std'])

                                                            # dfList.append(df)
                                                            sheet = (f'{clf}_{lr}_{lrPatience}_{lrDrop}_{batch}')
                                                            # sheet = f'{clf}_{loss}_{alpha}_{gamma}'

                                                            fname = f'{self.dataTopLevelPath}\classifierResultsSheets\CNN_{timestamp}.xlsx'

                                                            if it == 0:
                                                                writer = pd.ExcelWriter(fname)
                                                                df.to_excel(writer, sheet)
                                                                writer.close()
                                                            else:
                                                                book = load_workbook(fname)
                                                                writer = pd.ExcelWriter(fname, engine='openpyxl')
                                                                writer.book = book

                                                                df.to_excel(writer, sheet)
                                                                writer.close()

                                                            print(f'Done with iteration {it+1} out of {num_its}')
                                                            it += 1


        self.modelSaveFlag = 1

    @pyqtSlot()
    def on_pushButtonCreateClassifierThickness_clicked(self):
        """Create multiple classifier with varying fat threshold for predicition of fat thickness"""

        # set up timestamp for model and train/test data file names
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        fatThicknesses = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        self.fatThreshList = []

        self.modelSaveFolder = fr'{self.classifierModelsPath}\Thickness Classifier RF\model_FAT_THICKNESS_PREDICTION_{timestamp}'
        channelThickness = self.sliceWidgetBMode.ContourThickness
        roiNum = self.sliceWidgetBMode.ROIChannelNum
        signalThresh = self.roiAvgThresh

        # check if iq data with each fat threshold exist, if not, get data
        for thickness in fatThicknesses:
            fat1 = str(thickness).split('.')[0]
            fat2 = str(thickness).split('.')[1]
            path = rf'{self.iqChannelROIsPath}\fat_{fat1}_{fat2}\thickness_{channelThickness}\ROI_masks\channel\{roiNum}'

            # add to list of thicknesses to get
            if not os.path.exists(path):
                self.fatThreshList.append(thickness)

        self.on_pushButtonGetAllData_clicked()

        # check if spectra exist
        for thickness in fatThicknesses:
            fat1 = str(thickness).split('.')[0]
            fat2 = str(thickness).split('.')[1]
            path = rf'{self.channelSpectraPath}\thresh_{signalThresh}\fat_{fat1}_{fat2}\thickness_{channelThickness}\yule\Channel\{roiNum}'

            if not os.path.exists(path):
                self.mriWindow.binary_fat_thresh_changed(thickness)
                self.on_pushButtonComputeSpectra_clicked()

        # don't shuffle train/test split
        self.classifierShuffleFlag = False

        # don't want dialogue box asking to save each model
        self.modelSaveFlag = 0

        for thickness in fatThicknesses:
            self.mriWindow.binary_fat_thresh_changed(thickness)

            self.on_pushButtonCreateClassifier_clicked()

        # reset flags
        self.classifierShuffleFlag = True
        self.modelSaveFlag = 1

    @pyqtSlot()
    def on_pushButtonPredictThickness_clicked(self):
        """Predict fat/not fat for currently loaded file with each model to get approx fat thickness"""

        if len(self.sliceWidgetBMode.bModeOutsidePoints) == 0:
            print('Load Ultrasound BSpline First!')
            return

        # ask for folder containing models
        modelFolderPath = FileDialog.getExistingDirectory(None, f"Select models folder")
        # ####### hardcoded temporarily #########
        # modelFolderPath = r'C:\Users\lgillet\School\BIRL\Python\EchoFatSegmentationGUI\data\classifierModels\Thickness Classifier RF\model_FAT_THICKNESS_PREDICTION_20230822-153619'

        models = []
        for file in os.listdir(modelFolderPath):
            if file.find('.pkl') != -1:
                models.append(file)

        # get case name, roi size, and channel thickness currently set
        CASE = f'{self._iqFileName}_E{self.mriWindow.get_frame_flag()}'
        type = 'Channel'
        roiSize = self.sliceWidgetBMode.ROIChannelNum
        thickness = self.sliceWidgetBMode.ContourThickness

        fatThicknessList = []
        predList = []
        probsList = []
        successList = []

        for model in models:
            modelFullPath = fr'{modelFolderPath}\{model}'

            fat_ind = model.find('fat')

            fat1 = model[fat_ind + 4]
            fat2 = model[fat_ind + 6]

            fatThickness = f'{fat1}.{fat2}'
            fatThicknessList.append(float(fatThickness))

            # path to spectra and rois
            folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\thickness_{thickness}\yule\Channel\{roiSize}\{CASE}_lists'
            roiPath = fr'{self.SAVEPATH}\ROI_masks\channel\{roiSize}\{CASE[:-2]}Channel_{CASE[-2:]}'

            labelList = []

            # get spectra and add to array
            n = 0
            while n < int(roiSize):
                param = f'norm_param_list_{roiSize}_{n}'
                currParams = np.load(fr'{folderPath}\{param}.npy')

                if n == 0:
                    paramsList = np.zeros(np.shape(currParams)[0])
                    paramsList[:] = currParams[:]
                else:
                    paramsList = np.row_stack((paramsList, currParams))

                # load tissue type labels and add to list
                currLabel = np.load(fr'{folderPath}\labels_{roiSize}_{n}.npy')
                labelList = np.append(labelList, currLabel)

                n += 1

            paramsList = np.array(paramsList)

            bModeROIs = []
            roiCentroids = []

            # get rois
            for n, roiFile in enumerate(sorted(os.listdir(roiPath), key=lambda s: int(re.findall(r'\d+', s)[4]))):
                if labelList[n] == 0:
                    continue
                # cleaning up file name
                self._roiFileNamePath = util.cleanPath(rf'{roiPath}\{roiFile}')
                self._roiFileName = util.splitext(os.path.basename(self._roiFileNamePath))[0]

                roiMaskIQ = np.load(self._roiFileNamePath)

                roiMaskBMode = convert_mask_from_polar_to_cartesian(np.flipud(roiMaskIQ))
                roiMaskBMode = np.flipud(roiMaskBMode)
                roiMaskBMode = roiMaskBMode.astype('uint8')

                centroid = scipy.ndimage.measurements.center_of_mass(roiMaskBMode)
                roiCentroids.append(centroid)

                bModeROIs.append(roiMaskBMode)

            # remove all 0 spectra and labels
            paramsList = paramsList[~np.all(paramsList == 0, axis=1)]
            labelList = [i for i in labelList if i != 0]

            # predict fat/nonfat based on spectral parameters - return classifications and success params
            pred, probs, success = classifier.predict(paramsList, labelList, modelFullPath)

            predList.append(pred)
            probsList.append(probs)
            successList.append(success)

        currImageHeightIQ = 596

        roi_conf = np.zeros((len(bModeROIs)))
        roi_labels = np.zeros((len(bModeROIs)))
        roi_thickness = np.zeros((len(bModeROIs)))

        fig = plt.figure()
        ax = fig.subplots(nrows=2, ncols=3)

        for k, (thickness, ((x, y), value)) in enumerate(zip(fatThicknessList, np.ndenumerate(ax))):
            # loop thru ROIs and plot
            predictions = predList[k]
            probabilities = probsList[k]
            successes = successList[k]

            patches_fat = []
            patches_nonfat = []

            ax[x, y].set_title(f'Fat Threshold {thickness}')
            ax[x, y].set_xlim((0, 800))
            ax[x, y].set_ylim((0, 600))

            roi_image = np.zeros((600, 800, 3))

            # create image showing predictions of all classifiers
            for i in range(len(bModeROIs)):

                # if not fat
                if predictions[i][0] == 0:
                    conf = min(probabilities[i][0])
                    roi = bModeROIs[i] * conf
                    roi = roi.astype('uint8')

                    roi_image[:, :, 0] += roi
                    roi_image[:, :, 1] += roi
                    roi_image[:, :, 2] += roi
                    # roi = matplotlib.patches.Polygon(bModeContourPoints, closed=True, fill=False,
                    #                                  edgecolor=(conf, 0, 0), zorder=3)
                    # roi_image += [bModeROIs[i][0], bModeROIs[i][1], 255*conf]
                    text = f'{conf}'
                    ax[x, y].text(roiCentroids[i][1]-12, roiCentroids[i][0], text, fontsize='small')

                    # patches_nonfat.append(roi)

                    # if confidence higher than the previous, replace and change label
                    if 1 - conf > roi_conf[i]:
                        roi_conf[i] = conf
                        roi_labels[i] = 0
                        roi_thickness[i] = thickness

                # if fat
                elif predictions[i][0] == 1:
                    conf = max(probabilities[i][0])
                    # roi_image += [bModeROIs[i][0], bModeROIs[i][1], 255*conf]
                    text = f'{conf}'
                    ax[x, y].text(roiCentroids[i][1]-12, roiCentroids[i][0], text, fontsize='small')

                    # patches_fat.append(roi)

                    roi = bModeROIs[i] * conf
                    roi = roi.astype('uint8')

                    roi_image[:, :, 0] += roi

                    # if confidence higher than the current, replace and change label
                    if conf > roi_conf[i]:
                        roi_conf[i] = conf
                        roi_labels[i] = 1
                        roi_thickness[i] = thickness

                # ax[x, y].add_patch(roi)
                # ax[x, y].add_image(roi_image)
                roi_image = roi_image.astype('uint8')
                ax[x, y].imshow(roi_image)

        self.predictedLabels = list(roi_labels)

        plt.show()

        # for ii in range(len(self.predictedLabels)):
        #     self.tissueType = self.predictedLabels[ii]
        #     # self.sliceWidgetIQ.IQROIMasks.append(iqROIs[ii])
        #     self.sliceWidgetBMode.BModeROIMasks.append(bModeROIs[ii])
        #
        #     self.tissueTypeList.append(self.tissueType)
        #     self.sliceWidgetIQ.tissueTypeList.append(self.tissueType)
        #     self.sliceWidgetBMode.tissueTypeList.append(self.tissueType)

        # self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
        #                                    self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)
        # self.sliceWidgetIQ.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
        #                                 self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

        # plot ROIs of best confidences and labels

        # draw normal vectors and find which ROI the point is within - if fat ROI then add normal vector point to
        # list, else add bspline point
        outsidePoints = np.array(self.sliceWidgetBMode.bModeOutsidePoints)

        outsidePoints_iq = []
        currImageHeightIQ = self.iqImageHeight
        for j in range(len(outsidePoints)):
            (x, y) = polarTransform.getPolarPointsImage(outsidePoints[j], self.ptSettings)
            outsidePoints_iq.append([y, currImageHeightIQ - x])

        outsidePoints_iq = np.array(outsidePoints_iq)

        # get fat thickness and convert to pixels in ultrasound
        conversion = (self.bmodeFinalRadius - self.bmodeInitialRadius) / self.bmodeImageDepthMM  # [pixels/mm]
        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

        # length for testing which ROI point belongs to

        fatContourPoints = []
        for i in range(0, len(outsidePoints)):
            splinePoint = (outsidePoints[i, 0], outsidePoints[i, 1])

            roiNum = 0
            lowestDist = 9999
            for j in range(len(bModeROIs)):

                dist = spatial.distance.euclidean(splinePoint, [roiCentroids[j][1], roiCentroids[j][0]])
                if dist < lowestDist:
                    roiNum = j
                    lowestDist = dist

            # if ROI that point is within is fat add normal vector point to list
            predictionThreshold = 0.6
            if roi_conf[roiNum] > predictionThreshold:
                # fat
                length = roi_thickness[roiNum] * conversion
                crossX, crossY = fatMapping.getNormalVectors(outsidePoints, length=length)
                normalPoint = (outsidePoints[i, 0] - crossY[0, i], outsidePoints[i, 1] + crossX[0, i])

                fatContourPoints.append(np.array(normalPoint))
                self.predictedThickness.append(length / conversion)
            else:
                # not fat
                fatContourPoints.append(np.array((outsidePoints[i, 0], outsidePoints[i, 1])))
                self.predictedThickness.append(0)

        # make sure fat map is list then clear
        self.sliceWidgetBMode.bModeFatMap = list(self.sliceWidgetBMode.bModeFatMap)
        self.sliceWidgetBMode.bModeFatMap.clear()

        self.sliceWidgetBMode.bModeFatMap.append(fatContourPoints)
        self.sliceWidgetBMode.bModeFatMap = self.sliceWidgetBMode.bModeFatMap[0]

        self.manualIQROIs.clear()

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonCompareMRIThickness_clicked(self):
        """Plot MRI fat and compare to predicted fat thickness, compute MSE"""

        if len(self.sliceWidgetBMode.bModeFatMap) == 0:
            "Load fat thickness predicition first!"
            return

        # ask for folder containing models
        modelFolderPath = FileDialog.getExistingDirectory(None, f"Select models folder")
        if modelFolderPath is None:
            return
        ####### hardcoded temporarily #########
        # modelFolderPath = r'C:\Users\lgillet\School\BIRL\Python\EchoFatSegmentationGUI\data\classifierModels\Thickness Classifier RF\model_FAT_THICKNESS_PREDICTION_20230822-153619'

        models = []
        for file in os.listdir(modelFolderPath):
            if file.find('.pkl') != -1 or file.find('.txt') == -1:
                models.append(file)

        fatThicknessList = []
        for model in models:
            modelFullPath = fr'{modelFolderPath}\{model}'

            fat_ind = model.find('fat')

            fat1 = model[fat_ind + 4]
            fat2 = model[fat_ind + 6]

            fatThickness = f'{fat1}.{fat2}'
            fatThicknessList.append(float(fatThickness))

        # round mri fat thicknesses to nearest threshold used in prediction or to 0
        thickness = self.mriWindow.get_thickness_by_point()
        mriPossibleThicknesses = fatThicknessList
        mriPossibleThicknesses.insert(0, 0)
        if thickness:

            for i in range(len(thickness[0])):
                val = thickness[0][i]

                new_val = min(mriPossibleThicknesses, key=lambda x: abs(x - val))

                thickness[0][i] = new_val

        else:
            print("No MRI fat thickness loaded")
            return

        self.mriWindow.thickness_by_point_changed(thickness)

        self.on_pushButtonFatThickness_clicked()

        self.on_pushButtonFatMapMSE_clicked()

    @pyqtSlot()
    def on_pushButtonCreateCNNThickness_clicked(self):
        """Create multiple CNN classifiers with varying fat threshold for predicition of fat thickness"""

        # set up timestamp for model and train/test data file names
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        fatThicknesses = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        self.fatThreshList = []

        modelName = "UNet1D"

        self.modelSaveFolder = fr'{self.classifierModelsPath}\Thickness Classifier CNN\model_{modelName}_FAT_THICKNESS_PREDICTION_{timestamp}'
        channelThickness = self.sliceWidgetBMode.ContourThickness
        roiNum = self.sliceWidgetBMode.ROIChannelNum
        signalThresh = self.roiAvgThresh

        # check if iq data with each fat threshold exist, if not, get data
        for thickness in fatThicknesses:
            fat1 = str(thickness).split('.')[0]
            fat2 = str(thickness).split('.')[1]
            path = rf'{self.iqChannelROIsPath}\fat_{fat1}_{fat2}\thickness_{channelThickness}\ROI_masks\channel\{roiNum}'

            # add to list of thicknesses to get
            if not os.path.exists(path):
                self.fatThreshList.append(thickness)

        self.on_pushButtonGetAllData_clicked()

        # check if spectra exist
        for thickness in fatThicknesses:
            fat1 = str(thickness).split('.')[0]
            fat2 = str(thickness).split('.')[1]
            path = rf'{self.channelSpectraPath}\thresh_{signalThresh}\fat_{fat1}_{fat2}\thickness_{channelThickness}\yule\Channel\{roiNum}'

            if not os.path.exists(path):
                self.mriWindow.binary_fat_thresh_changed(thickness)
                self.on_pushButtonComputeSpectra_clicked()

        # don't shuffle train/test split
        self.classifierShuffleFlag = False

        # don't want dialogue box asking to save each model
        self.modelSaveFlag = 0
        self.saveCNNFlag = 1

        for thickness in fatThicknesses:
            self.mriWindow.binary_fat_thresh_changed(thickness)

            self.on_pushButtonUNetClassifier_clicked()

        # reset flags
        self.classifierShuffleFlag = True
        self.modelSaveFlag = 1
        self.saveCNNFlag = 0

    @pyqtSlot()
    def on_pushButtonPredictThicknessCNN_clicked(self):
        """Predict fat/not fat for currently loaded file with each model to get approx fat thickness"""

        if len(self.sliceWidgetBMode.bModeOutsidePoints) == 0:
            print('Load Ultrasound BSpline First!')
            return

        # ask for folder containing models
        modelFolderPath = FileDialog.getExistingDirectory(None, f"Select models folder")
        ####### hardcoded temporarily #########
        # modelFolderPath = r'C:\Users\lgillet\School\BIRL\Python\EchoFatSegmentationGUI\data\classifierModels\Thickness Classifier CNN\model_simpleModel1D_FAT_THICKNESS_PREDICTION_20230907-102442'

        models = []
        for file in os.listdir(modelFolderPath):
            if file.find('.hdf5') != -1 and file.find('.txt') == -1:
                models.append(file)

        # get case name, roi size, and channel thickness currently set
        CASE = f'{self._iqFileName}_E{self.mriWindow.get_frame_flag()}'
        type = 'Channel'
        roiSize = self.sliceWidgetBMode.ROIChannelNum
        thickness = self.sliceWidgetBMode.ContourThickness

        fatThicknessList = []
        predList = []
        probsList = []
        successList = []
        batch_size = 4

        for model in models:
            modelFullPath = fr'{modelFolderPath}\{model}'

            fat_ind = model.find('fat')

            fat1 = model[fat_ind + 4]
            fat2 = model[fat_ind + 6]

            fatThickness = f'{fat1}.{fat2}'
            fatThicknessList.append(float(fatThickness))

            # path to spectra and rois
            folderPath = fr'{self.channelSpectraPath}\thresh_{self.roiAvgThresh}\fat_{fat1}_{fat2}\thickness_{thickness}\yule\Channel\{roiSize}\{CASE}_lists'
            roiPath = fr'{self.SAVEPATH}\ROI_masks\channel\{roiSize}\{CASE[:-2]}Channel_{CASE[-2:]}'

            labelList = []

            # get spectra and add to array
            n = 0
            while n < int(roiSize):
                param = f'norm_param_list_{roiSize}_{n}'
                currParams = np.load(fr'{folderPath}\{param}.npy')

                num_params = np.shape(currParams)[0]

                if n == 0:
                    paramsList = np.zeros(np.shape(currParams)[0])
                    paramsList[:] = currParams[:]
                else:
                    paramsList = np.row_stack((paramsList, currParams))

                # load tissue type labels and add to list
                currLabel = np.load(fr'{folderPath}\labels_{roiSize}_{n}.npy')
                labelList = np.append(labelList, currLabel)

                n += 1

            paramsList = np.array(paramsList)

            bModeROIs = []
            roiCentroids = []

            # get rois
            for n, roiFile in enumerate(sorted(os.listdir(roiPath), key=lambda s: int(re.findall(r'\d+', s)[4]))):
                if labelList[n] == 0:
                    continue
                # cleaning up file name
                self._roiFileNamePath = util.cleanPath(rf'{roiPath}\{roiFile}')
                self._roiFileName = util.splitext(os.path.basename(self._roiFileNamePath))[0]

                roiMaskIQ = np.load(self._roiFileNamePath)

                roiMaskBMode = convert_mask_from_polar_to_cartesian(np.flipud(roiMaskIQ))
                roiMaskBMode = np.flipud(roiMaskBMode)
                roiMaskBMode = roiMaskBMode.astype('uint8')

                centroid = scipy.ndimage.measurements.center_of_mass(roiMaskBMode)
                roiCentroids.append(centroid)

                bModeROIs.append(roiMaskBMode)

            # remove all 0 spectra and labels
            paramsList = paramsList[~np.all(paramsList == 0, axis=1)]
            labelList = [i for i in labelList if i != 0]

            paramsList = standardize_features([paramsList])
            paramsList = paramsList[0]

            params = np.array(paramsList)
            labels = np.array(labelList)

            labels[labels == 4] = 0
            labelsLen = len(labels)
            indices = [*range(0, labelsLen)]
            indices = np.array(indices)

            params = params.take(range(-32, 32), mode='wrap', axis=0)
            labels = labels.take(range(-32, 32), mode='wrap', axis=0)
            indices = indices.take(range(-32, 32), mode='wrap', axis=0)

            data = tf.data.Dataset.from_tensor_slices(([params], [labels]))
            data = data.batch(batch_size)

            model = keras.models.load_model(modelFullPath, compile=False)
            probs_wrapped = model.predict(data)[0]

            probs = np.zeros(labelsLen)

            for i, ind in enumerate(indices):
                if np.count_nonzero(probs) == labelsLen:
                    break

                probs[ind] = probs_wrapped[i][0]

            pred = np.copy(probs)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            predList.append(pred)
            probsList.append(probs)
            # successList.append(success)

        currImageHeightIQ = 596

        roi_conf = np.zeros((len(bModeROIs)))
        roi_labels = np.zeros((len(bModeROIs)))
        roi_thickness = np.zeros((len(bModeROIs)))

        fig = plt.figure()
        ax = fig.subplots(nrows=3, ncols=2)

        for k, (thickness, ((x, y), value)) in enumerate(zip(fatThicknessList, np.ndenumerate(ax))):
            # loop thru ROIs and plot
            predictions = predList[k]
            probabilities = probsList[k]

            patches_fat = []
            patches_nonfat = []

            ax[x, y].set_title(f'Fat Threshold {thickness}')
            ax[x, y].set_xlim((0, 800))
            ax[x, y].set_ylim((0, 600))

            roi_image = np.zeros((600, 800, 3))

            # create image showing predictions of all classifiers
            for i in range(len(bModeROIs)):

                # if not fat
                if predictions[i] == 0:
                    conf = probabilities[i]
                    # roi = matplotlib.patches.Polygon(bModeContourPoints, closed=True, fill=False,
                    #                                  edgecolor=(conf, 0, 0), zorder=3)
                    # roi_image += [bModeROIs[i][0], bModeROIs[i][1], 255*conf]
                    # text = f'{conf}'
                    # ax[x, y].text(roiCentroids[i][0], roiCentroids[i][1], text)

                    # patches_nonfat.append(roi)

                    # if confidence higher than the previous, replace and change label
                    if 1 - conf > roi_conf[i]:
                        roi_conf[i] = conf
                        roi_labels[i] = 0
                        roi_thickness[i] = thickness

                # if fat
                elif predictions[i] == 1:
                    conf = probabilities[i]
                    # roi_image += [bModeROIs[i][0], bModeROIs[i][1], 255*conf]
                    # text = f'{conf}'
                    # ax[x, y].text(roiCentroids[i][0], roiCentroids[i][1], text)

                    # patches_fat.append(roi)

                    # if confidence higher than the current, replace and change label
                    if conf > roi_conf[i]:
                        roi_conf[i] = conf
                        roi_labels[i] = 1
                        roi_thickness[i] = thickness

                # ax[x, y].add_patch(roi)
                # ax[x, y].add_image(roi_image)

        self.predictedLabels = list(roi_labels)

        # plot ROIs of best confidences and labels

        # draw normal vectors and find which ROI the point is within - if fat ROI then add normal vector point to
        # list, else add bspline point
        outsidePoints = np.array(self.sliceWidgetBMode.bModeOutsidePoints)

        outsidePoints_iq = []
        currImageHeightIQ = self.iqImageHeight
        for j in range(len(outsidePoints)):
            (x, y) = polarTransform.getPolarPointsImage(outsidePoints[j], self.ptSettings)
            outsidePoints_iq.append([y, currImageHeightIQ - x])

        outsidePoints_iq = np.array(outsidePoints_iq)

        # get fat thickness and convert to pixels in ultrasound
        conversion = (self.bmodeFinalRadius - self.bmodeInitialRadius) / self.bmodeImageDepthMM  # [pixels/mm]
        binaryFatThreshmm = self.mriWindow.get_binary_fat_thresh_mm()

        fatContourPoints = []
        for i in range(0, len(outsidePoints)):
            splinePoint = (outsidePoints[i, 0], outsidePoints[i, 1])

            roiNum = 0
            lowestDist = 9999
            for j in range(len(bModeROIs)):

                dist = spatial.distance.euclidean(splinePoint, [roiCentroids[j][1], roiCentroids[j][0]])
                if dist < lowestDist:
                    roiNum = j
                    lowestDist = dist

            # if ROI that point is within is fat add normal vector point to list
            predictionThreshold = 0.6
            if roi_conf[roiNum] > predictionThreshold:
                # fat
                length = roi_thickness[roiNum] * conversion
                crossX, crossY = fatMapping.getNormalVectors(outsidePoints, length=length)
                normalPoint = (outsidePoints[i, 0] - crossY[0, i], outsidePoints[i, 1] + crossX[0, i])

                fatContourPoints.append(np.array(normalPoint))
                self.predictedThickness.append(length / conversion)
            else:
                # not fat
                fatContourPoints.append(np.array((outsidePoints[i, 0], outsidePoints[i, 1])))
                self.predictedThickness.append(0)

        # make sure fat map is list then clear
        self.sliceWidgetBMode.bModeFatMap = list(self.sliceWidgetBMode.bModeFatMap)
        self.sliceWidgetBMode.bModeFatMap.clear()

        self.sliceWidgetBMode.bModeFatMap.append(fatContourPoints)
        self.sliceWidgetBMode.bModeFatMap = self.sliceWidgetBMode.bModeFatMap[0]

        self.manualIQROIs.clear()

        self.sliceWidgetBMode.updateFigure(self.roiDataList, self.landMarkViewNumber, self.ptSettings,
                                           self.manualBModeROIs, self.manualIQROIs, self.tissueTypeList)

    @pyqtSlot()
    def on_pushButtonOpenMRIWindow_clicked(self):
        self.mriWindow.show()

    def get_bmode_morph_points(self):
        return self.bModeMorphPoints

    def get_auto_load_flag(self):
        return self.autoLoadFlag

    def get_mri_path(self):
        return self.mriPath

    def get_bmode_images(self):
        return self.bModeImages

    def get_ftl_alpha(self):
        return self.doubleSpinBoxFTLAlpha.value()

    def get_ftl_gamma(self):
        return self.doubleSpinBoxFTLGamma.value()