from processing import library
from processing import OCR_library
import numpy as np
import cv2
import time

def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here

    # start measuring time
    start = time.time()

    #---------------------------initialize OCR---------------------------
    # read OCR files
    npaClassifications, npaFlattenedImages = OCR_library.readOCR()
    # reshape numpy array to 1d, necessary to pass to call to train
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
    # instantiate KNN object
    kNearest = cv2.ml.KNearest_create()
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    #------------------------detection/selection-------------------------
    # get input image dimensions
    image_height, image_width, image_channels = image.shape
    # initial image processing
    thresh, contours = library.preliminaryProcessing(image, image_width)
    # selecting contours
    contours = library.selection(contours, image_width)

    #------------------------------grouping------------------------------
    # create list of lists containing groups of contours
    groups = library.grouping(contours, image_width * 0.12)
    # print info about number of contours
    library.printLengthInfoOverall(groups)

    #--------------------------final selection---------------------------
    # choose contours
    allContoursWithData, rawContoursWithData = library.chooseSymbols(groups)
    # create final contours list
    validContoursWithData = OCR_library.createValidContoursList(allContoursWithData)

    #----------------------------recognition-----------------------------
    # recognize characters and print
    strFinalString1 = OCR_library.recognize(thresh, kNearest, validContoursWithData)

    #--------------------------draw and resize---------------------------
    thresh_drawing = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    library.drawGroups(groups, thresh_drawing)
    thresh_drawing = library.resizeImage(thresh_drawing, 1000)

    #--------------------repeated detection/selection--------------------
    # decide whether to repeat the process or not
    rep_det, rep_det_all = library.repeatDecision(rawContoursWithData)
    # if yes
    if rep_det:
        # repeated detection/selection
        strFinalString2, thresh_crop_drawing = library.repeatedDetection(kNearest, rawContoursWithData, thresh, image_width)
        # warn, resize, draw
        rep_det_all, thresh_crop_drawing = library.resizeAndWarn(rep_det, rep_det_all, thresh_crop_drawing, strFinalString1, strFinalString2)

    # fill final result up to 7 elements
    if rep_det_all:
        strFinalString2 = library.fill(strFinalString2)
    else:
        strFinalString1 = library.fill(strFinalString1)

    # discard elements after 7th (just to be sure)
    if rep_det_all:
        strFinalString2 = strFinalString2[:7]
    else:
        strFinalString1 = strFinalString1[:7]

    # stop measuring time and print
    end = time.time()
    print('I worked for', format((end - start), '.2f'), 'seconds\n')

    # show result images
    cv2.imshow('threshold image', thresh_drawing)
    if rep_det:
        cv2.imshow('crop result', thresh_crop_drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if rep_det_all:
        return strFinalString2
    else:
        return strFinalString1