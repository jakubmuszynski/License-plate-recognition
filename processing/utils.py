from processing import library
import numpy as np
import cv2
import time
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
        if counter == 3:
            if strCurrentChar == 'O':
                if previousChar != 'P' and previousChar != 'Z' and previousChar != 'G' and previousChar != 'K' and previousChar != 'L':
                    strCurrentChar = '0'
        if counter >= 4:
            if strCurrentChar == 'O':
                strCurrentChar = '0'
        strFinalString = strFinalString + strCurrentChar
        previousChar = strCurrentChar
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

def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here

    # start measuring time
    start = time.time()

    # read OCR files
    npaClassifications, npaFlattenedImages = readOCR()

    # reshape numpy array to 1d, necessary to pass to call to train
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    # instantiate KNN object
    kNearest = cv2.ml.KNearest_create()
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    # get input image dimensions
    image_height, image_width, image_channels = image.shape

    # initial image processing
    thresh, contours = library.preliminaryProcessing(image, image_width)

    # selecting contours - width/image_width and height/image_width ratios
    contours = library.selectionContourRatios(contours, image_width)

    # selecting contours - minimum 1 close neighbour, proper angle between them
    contours = library.selectionDistanceAngle(contours, image_width)

    # selecting contours - difference from average height
    contours = library.selectionAverageHeight(contours, image_width)

    # selecting contours - discarding duplicates
    contours = library.discardDuplicates(contours)

    # selecting contours - discarding inner contours
    contours = library.discardInnerContours(contours, image_width)

    # sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # create list of lists containing groups of contours
    groups = library.grouping(contours, image_width * 0.12)

    # print info about number of contours
    print('-----OVERALL DETECTION-----')
    library.printLengthInfo(groups)

    # choose contours
    allContoursWithData, rawContoursWithData = library.chooseSymbols(groups)

    # create final contours list
    validContoursWithData = createValidContoursList(allContoursWithData)

    # recognize characters and print
    strFinalString1 = recognize(thresh, kNearest, validContoursWithData)
    print(strFinalString1)

    # drawing
    thresh_drawing = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    library.drawGroups(groups, thresh_drawing)

    # repeated detection
    if len(rawContoursWithData) <= 8 and len(rawContoursWithData) >= 6:
        # crop chosen contours group and repeat detection process
        groups_crop, thresh_crop = library.repeatOperationsOnCrop(rawContoursWithData, thresh, image_width)
        # choose contours
        allContoursWithData, rawContoursWithData = library.chooseSymbols(groups_crop)
        # create final contours list
        validContoursWithData = createValidContoursList(allContoursWithData)
        # recognize characters and print
        strFinalString2 = recognize(thresh_crop, kNearest, validContoursWithData)
        print(strFinalString2)
        # drawing
        thresh_crop_drawing = cv2.cvtColor(thresh_crop, cv2.COLOR_GRAY2BGR)
        library.drawGroups(groups_crop, thresh_crop_drawing)
        # flags
        repeated_detection = True
        repeated_detection_allowance = True
    else:
        repeated_detection = False
        repeated_detection_allowance = False
        print('WARNING: repeated detection aborted')

    # resize images
    thresh_drawing = library.resizeImage(thresh_drawing, 1000)
    if repeated_detection:
        thresh_crop_drawing = library.resizeImage(thresh_crop_drawing, 1000)
        if strFinalString2[0] != 'B' and strFinalString2[0] != 'C' and strFinalString2[0] != 'D' and strFinalString2[0] != 'E' and strFinalString2[0] != 'F' and strFinalString2[0] != 'G' and strFinalString2[0] != 'K' and strFinalString2[0] != 'L' and strFinalString2[0] != 'N' and strFinalString2[0] != 'O' and strFinalString2[0] != 'P' and strFinalString2[0] != 'R' and strFinalString2[0] != 'S' and strFinalString2[0] != 'T' and strFinalString2[0] != 'W' and strFinalString2[0] != 'Z':
            repeated_detection_allowance = False
            print('WARNING: incorrect first character in repeated detection - rejecting')
        if strFinalString2[1] == '1' or strFinalString2[1] == '2' or strFinalString2[1] == '3' or strFinalString2[1] == '4' or strFinalString2[1] == '5' or strFinalString2[1] == '6' or strFinalString2[1] == '7' or strFinalString2[1] == '8' or strFinalString2[1] == '9':
            repeated_detection_allowance = False
            print('WARNING: incorrect second character in repeated detection - rejecting')
        if len(strFinalString2) < len(strFinalString1):
            repeated_detection_allowance = False
            print('WARNING: incorrect repeated detection result size - rejecting')

    # stop measuring time and print
    end = time.time()
    print('I worked for', format((end - start), '.2f'), 'seconds\n')

    # show result images
    # cv2.imshow('threshold image', thresh_drawing)
    # if repeated_detection:
    #     cv2.imshow('crop result', thresh_crop_drawing)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if repeated_detection_allowance:
        return strFinalString2
    else:
        return strFinalString1
