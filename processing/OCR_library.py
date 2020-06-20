import numpy as np
import cv2
import os

MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def readOCR():
    try:
        npaClassifications = np.loadtxt("data/classifications.txt", np.float32) # read in training classifications
    except:
        print("error, unable to open classifications.txt, exiting program")
        os.system("pause")
        return
    try:
        npaFlattenedImages = np.loadtxt("data/flattened_images.txt", np.float32) # read in training images
    except:
        print("error, unable to open flattened_images.txt, exiting program")
        os.system("pause")
        return
    return npaClassifications, npaFlattenedImages

def createValidContoursList(npaContours):
    validContoursWithData = []
    for npaContour in npaContours:
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        validContoursWithData.append(contourWithData)                                   # add contour with data object to list of all contours with data
    return validContoursWithData

def recognize(image, kNearest, validContoursWithData):
    previouspreviousChar = 'Q'
    previousChar = 'Q'
    counter = 0
    strFinalString = ""
    for contourWithData in validContoursWithData:
        counter += 1
        imgROI = image[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight, contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)
        strCurrentChar = str(chr(int(npaResults[0][0])))

        # first 2 symbols cant be numbers
        if counter == 1 or counter == 2:
            if strCurrentChar == '0':
                strCurrentChar = 'O'

        # 3rd symbol has to be 0 instead of O except for OPO, SZO, NGO, PKO, PGO, ZKO, ZLO
        elif counter == 3:
            if strCurrentChar == 'O':
                allow_change = True
                # OPO
                if previouspreviousChar == 'O':
                    if previousChar == 'P':
                        allow_change = False
                # SZO
                elif previouspreviousChar == 'S':
                    if previousChar == 'Z':
                        allow_change = False
                # NGO
                elif previouspreviousChar == 'N':
                    if previousChar == 'G':
                        allow_change = False
                # PKO and PGO
                elif previouspreviousChar == 'P':
                    if previousChar == 'K':
                        allow_change = False
                    elif previousChar == 'G':
                        allow_change = False
                # ZKO and ZLO
                elif previouspreviousChar == 'Z':
                    if previousChar == 'K':
                        allow_change = False
                    elif previousChar == 'L':
                        allow_change = False
                if allow_change:
                    strCurrentChar = '0'

        # determine between O and zero based on average ratio
        elif counter >= 4:
            if strCurrentChar == 'O' or strCurrentChar == '0':
                ratios = 0
                for c in validContoursWithData:
                    ratios = ratios + (c.intRectWidth / c.intRectHeight)
                avgRatio = ratios / len(validContoursWithData)
                ratio = contourWithData.intRectWidth / contourWithData.intRectHeight
                if ratio > avgRatio:
                    strCurrentChar = 'O'
                else:
                    strCurrentChar = '0'

        strFinalString = strFinalString + strCurrentChar
        previouspreviousChar = previousChar
        previousChar = strCurrentChar

    print(strFinalString)
    return strFinalString

class ContourWithData():
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self): # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self): # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False # much better validity checking would be necessary
        return True
