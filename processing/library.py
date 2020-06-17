from processing import OCR_library
import cv2
import math
import numpy as np
from random import randint  # for random colors only

def calculateDistance(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    val_x = ((x1 + (w1 / 2)) - (x2 + (w2 / 2)))
    val_y = ((y1 + (h1 / 2)) - (y2 + (h2 / 2)))
    distance = math.sqrt((val_x ** 2) + (val_y ** 2))
    return distance

def calculateAngle(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    distance = calculateDistance(contour1, contour2)
    y_diff = abs((y1 + (h1 / 2)) - (y2 + (h2 / 2)))
    if distance == 0.0:
        deg = 0.0
    else:
        sin = y_diff / float(distance)
        rad = math.asin(sin)
        deg = rad * 180 / math.pi
    return deg

def calculateHeightChangeRatio1(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    difference = abs(h1 - h2)
    ratio = difference / h1
    return ratio

def calculateAverageHeight(contours):
    result = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        result = result + h
    result = result / len(contours)
    return result

def calculateMaxHeight(contours):
    result = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > result:
            result = h
    return result

def calculateAverageHeightGroups(groups):
    result = 0
    for g in groups:
        for c in g:
            x, y, w, h = cv2.boundingRect(c)
            result = result + h
    no_elem = 0
    for g in groups:
        for c in g:
            no_elem += 1
    result = result / no_elem
    return result

def calculateAverageArea(contours):
    result = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        result = result + (w * h)
    result = result / len(contours)
    return result

def calculateMaxArea(contours):
    result = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > result:
            result = (w * h)
    return result

def preliminaryProcessing(image, image_width):
    blockSize = int(image_width / 25.6)
    if blockSize % 2 == 0:
        blockSize += 1
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    gray = value
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, 5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return thresh, contours

def preliminaryProcessing2(image, image_width):
    blockSize = int(image_width / 25.6)
    if blockSize % 2 == 0:
        blockSize += 1
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, 5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return thresh, contours

def selectionContourRatios(contours, image_width):
    result = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        offset_min = 5
        offset_max = 15
        prop_w_max = image_width * 426 / 4000 + offset_max    # 0.101 + offset
        prop_h_max = image_width * 636 / 3000 + offset_max    # 0.212 + offset
        prop_w_min = image_width * 25 / 4000 - offset_max     # 0.006 - offset (letter 'I')
        prop_h_min = image_width * 0.0535 - offset_min  # 0.071 - offset
        if w > prop_w_min and h > prop_h_min:
            if w < prop_w_max and h < prop_h_max:
                if (w / h) > 0.12 and (w / h) < 0.88:
                    result.append(c)
    return result

def selectionDistanceAngle(contours, image_width):
    result = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        for c1 in contours:
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            if x != x1 and y != y1:
                distance = calculateDistance(c, c1)
                val = image_width * 120 / 4000
                max_distance = image_width * 426 / 4000 + val
                if distance > 0 and distance < max_distance:
                    angle_deg = calculateAngle(c, c1)
                    if angle_deg < 45:
                        result.append(c)
    return result

def selectionAverageHeight(contours, image_width):
    result = []
    difference = image_width * 0.034
    avg_h = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        avg_h = avg_h + h
    avg_h = avg_h / len(contours)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < (avg_h + difference) and h > (avg_h - difference):
            result.append(c)
    return result

def discardDuplicates(contours):
    result = []
    result.append(contours[0])
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        counter = 0
        for c1 in result:
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            if x == x1 and y == y1:
                counter += 1
        if counter < 1:
            result.append(c)
    return result

def discardInnerContours(contours, image_width):
    offset = image_width * 0.002
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        xo = x + (w / 2)
        yo = y + (h / 2)
        for c1 in contours:
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            xo1 = x1 + (w1 / 2)
            yo1 = y1 + (h1 / 2)
            if xo == xo1 and yo == yo1 and w == w1 and h == h1:
                pass
            elif xo > (xo1 - offset) and xo < (xo1 + offset) and yo > (yo1 - offset) and yo < (yo1 + offset):
                if w > w1:
                    contours.remove(c1)
                else:
                    contours.remove(c)
    return contours

def selection(contours, image_width):
    # selecting contours - width/image_width and height/image_width ratios
    contours = selectionContourRatios(contours, image_width)
    # selecting contours - minimum 1 close neighbour, correct angle between them
    contours = selectionDistanceAngle(contours, image_width)
    # selecting contours - difference from average height
    contours = selectionAverageHeight(contours, image_width)
    # selecting contours - discarding duplicates
    contours = discardDuplicates(contours)
    # selecting contours - discarding inner contours
    contours = discardInnerContours(contours, image_width)
    # sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    return contours

def group(contours, radius):
    # assume the first contour as first element in friends group
    friends = []
    friend = contours[0]
    friends.insert(len(friends), contours.pop(contours.index(friend)))
    i = 0
    exit = 0
    while exit != 1:
        # find another friend
        min_distance = radius
        for c in contours:
            distance = calculateDistance(friends[i], c)
            if distance < min_distance:
                min_distance = distance
                friend = c
        # if friend is eligible, move him from contours to friends
        if min_distance < radius:
            friends.insert(len(friends), contours.pop(contours.index(friend)))
            i += 1
        # if it isnt, quit looking for friends
        else:
            exit = 1
    result = []
    result.append(friends)
    return friends, contours

def grouping(contours, radius):
    result = []
    while len(contours) > 0:
        friends, contours = group(contours, radius)
        result.append(friends)
    return result

def printLengthInfoOverall(groups):
    all_contours_number = 0
    group_sizes = []
    for g in groups:
        group_sizes.append(len(g))
        all_contours_number = all_contours_number + len(g)
    print('-----OVERALL DETECTION-----')
    print("number of contours in total:", all_contours_number)
    print("groups:", group_sizes)

def printLengthInfoRepeated(groups):
    all_contours_number = 0
    group_sizes = []
    for g in groups:
        group_sizes.append(len(g))
        all_contours_number = all_contours_number + len(g)
    print('-----REPEATED DETECTION-----')
    print("number of contours in total:", all_contours_number)
    print("groups:", group_sizes)

def cropChosenContours(image, contours):
    image_height, image_width, image_channels = image.shape
    temp1_x = []
    temp1_y = []
    temp2_x = []
    temp2_y = []
    for i in range(len(contours)):
        bb_x, bb_y, bb_w, bb_h = cv2.boundingRect(contours[i])
        temp1_x.append(bb_x)
        temp1_x.append(bb_x+bb_w)
        temp1_y.append(bb_y)
        temp1_y.append(bb_y)
        temp2_x.append(bb_x)
        temp2_x.append(bb_x+bb_w)
        temp2_y.append(bb_y+bb_h)
        temp2_y.append(bb_y+bb_h)
    x1 = np.array(temp1_x, dtype=int)
    y1 = np.array(temp1_y, dtype=int)
    x2 = np.array(temp2_x, dtype=int)
    y2 = np.array(temp2_y, dtype=int)
    lin_points_up = list(zip(x1, y1))
    lin_points_down = list(zip(x2, y2))
    border = image_width * 0.007
    # uncomment to draw lines
    # up_left = (int(lin_points_up[0][0] - border), int(lin_points_up[0][1] - border))
    # up_right = (int(lin_points_up[len(lin_points_up) - 1][0] + border), int(lin_points_up[len(lin_points_up) - 1][1] - border))
    # down_left = (int(lin_points_down[0][0] - border), int(lin_points_down[0][1] + border))
    # down_right = (int(lin_points_down[len(lin_points_down) - 1][0] + border), int(lin_points_down[len(lin_points_down) - 1][1] + border))
    # cv2.line(image, (up_left), (up_right), (0, 0, 255), thickness=20)
    # cv2.line(image, (down_left), (down_right), (255, 0, 0), thickness=20)
    src_pts = np.array([[int(lin_points_up[0][0] - border), int(lin_points_up[0][1] - border)], [int(lin_points_up[len(lin_points_up) - 1][0] + border), int(lin_points_up[len(lin_points_up) - 1][1] - border)], [int(lin_points_down[0][0] - border), int(lin_points_down[0][1] + border)], [int(lin_points_down[len(lin_points_down) - 1][0] + border), int(lin_points_down[len(lin_points_down) - 1][1] + border)]], dtype="float32")
    width = int(np.sqrt((lin_points_up[0][0] - lin_points_up[len(lin_points_up) - 1][0]) ** 2 + (lin_points_up[0][1] - lin_points_up[len(lin_points_up) - 1][1]) ** 2))
    height = calculateMaxHeight(contours)
    dst_pts = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(image, M, (width, height), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,(255, 255, 255))
    return result

def repeatOperationsOnCrop(contours, thresh, image_width):
    # crop chosen contours from thresholded image
    thresh_to_crop = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cropped = cropChosenContours(thresh_to_crop, contours)
    # repeat the detection/selection process
    thresh_crop, contours_crop = preliminaryProcessing(cropped, image_width)
    contours_crop = selectionContourRatios(contours_crop, image_width)
    contours_crop = selectionDistanceAngle(contours_crop, image_width)
    contours_crop = selectionAverageHeight(contours_crop, image_width)
    contours_crop = discardDuplicates(contours_crop)
    contours_crop = discardInnerContours(contours_crop, image_width)
    contours_crop = sorted(contours_crop, key=lambda c: cv2.boundingRect(c)[0])
    groups_crop = grouping(contours_crop, image_width * 0.12)
    # print info about number of contours
    printLengthInfoRepeated(groups_crop)
    return groups_crop, thresh_crop

def chooseSymbols(groups):
    allContoursWithData = []
    group2 = []
    group3 = []
    group4 = []
    group5 = []
    group7 = []
    for i in range(len(groups)):
        if len(groups[i]) == 8:
            # discard left contour (probably EU/PL symbol)
            groups[i].remove(groups[i][0])
            # and add it to the list of potential plates
            group7.append(groups[i])
            print('removed 1 contour - probably EU/PL')
        elif len(groups[i]) == 7:
            group7.append(groups[i])
        if len(groups[i]) == 2:
            group2.append(groups[i])
        if len(groups[i]) == 5:
            group5.append(groups[i])
        if len(groups[i]) == 4:
            group4.append(groups[i])
        if len(groups[i]) == 3:
            group3.append(groups[i])
    # calculate total number of symbols
    no = 0
    for g in groups:
        no = no + len(g)
    # 1 plate with 7 symbols
    if len(group7) == 1:
        for c in group7[0]:
            allContoursWithData.append(c)
    # 1 plate in parts of 2 and 5
    elif len(group2) == 1 and len(group5) == 1:
        for c in group2[0]:
            allContoursWithData.append(c)
        for c in group5[0]:
            allContoursWithData.append(c)
    # 1 plate in parts of 3 and 4
    elif len(group3) == 1 and len(group4) == 1:
        for c in group3[0]:
            allContoursWithData.append(c)
        for c in group4[0]:
            allContoursWithData.append(c)
    # 1 plate in parts of 2 and 5 with PL/EU on the left
    elif len(group3) == 1 and len(group5) == 1:
        group3[0].remove(group3[0][0])
        for c in group3[0]:
            allContoursWithData.append(c)
        for c in group5[0]:
            allContoursWithData.append(c)
    # 7 symbols in total
    elif no == 7:
        for g in groups:
            for c in g:
                allContoursWithData.append(c)
    # 8 symbols in total
    elif no == 8:
        groups[0].remove(groups[0][0])
        for g in groups:
            for c in g:
                allContoursWithData.append(c)
    # 9 symbols in total
    elif no == 9:
        groups[0].remove(groups[0][0])
        for g in groups:
            for c in g:
                allContoursWithData.append(c)
    # only one group
    elif len(groups) == 1:
        for c in groups[0]:
            allContoursWithData.append(c)
    # if there is still a mess
    else:
        max_area = 0
        for g in groups:
            max_area_temp = calculateMaxArea(g)
            if max_area_temp > max_area:
                max_area = max_area_temp
        for g in groups:
            max_area_temp = calculateMaxArea(g)
            if max_area_temp > max_area * 0.75:
                for c in g:
                    allContoursWithData.append(c)
    # every contour
    rawContoursWithData = allContoursWithData
    # discard symbols after 7th
    allContoursWithData = allContoursWithData[:7]
    return allContoursWithData, rawContoursWithData

def repeatDecision(rawContoursWithData):
    if len(rawContoursWithData) <= 8 and len(rawContoursWithData) >= 6:
        rep_det = True
        rep_det_all = True
    else:
        rep_det = False
        rep_det_all = False
        print('WARNING: repeated detection aborted')
    return rep_det, rep_det_all

def repeatedDetection(kNearest, rawContoursWithData, thresh, image_width):
    # crop chosen contours group and repeat detection process
    groups_crop, thresh_crop = repeatOperationsOnCrop(rawContoursWithData, thresh, image_width)
    # choose contours
    allContoursWithData, rawContoursWithData = chooseSymbols(groups_crop)
    # create final contours list
    validContoursWithData = OCR_library.createValidContoursList(allContoursWithData)
    # recognize characters and print
    strFinalString2 = OCR_library.recognize(thresh_crop, kNearest, validContoursWithData)
    # drawing
    thresh_crop_drawing = cv2.cvtColor(thresh_crop, cv2.COLOR_GRAY2BGR)
    drawGroups(groups_crop, thresh_crop_drawing)
    return strFinalString2, thresh_crop_drawing

def resizeAndWarn(repeated_detection, repeated_detection_allowance, thresh_crop_drawing, strFinalString1, strFinalString2):
    if repeated_detection:
        thresh_crop_drawing = resizeImage(thresh_crop_drawing, 1000)
        repeated_detection_allowance = RDA(repeated_detection_allowance, strFinalString1, strFinalString2)
    return repeated_detection_allowance, thresh_crop_drawing

def RDA(repeated_detection_allowance, strFinalString1, strFinalString2):
    if strFinalString2[0] != 'B' and strFinalString2[0] != 'C' and strFinalString2[0] != 'D' and strFinalString2[
        0] != 'E' and strFinalString2[0] != 'F' and strFinalString2[0] != 'G' and strFinalString2[0] != 'K' and \
            strFinalString2[0] != 'L' and strFinalString2[0] != 'N' and strFinalString2[0] != 'O' and strFinalString2[
        0] != 'P' and strFinalString2[0] != 'R' and strFinalString2[0] != 'S' and strFinalString2[0] != 'T' and \
            strFinalString2[0] != 'W' and strFinalString2[0] != 'Z':
        repeated_detection_allowance = False
        print('WARNING: incorrect first character in repeated detection - rejecting')
    if strFinalString2[1] == '1' or strFinalString2[1] == '2' or strFinalString2[1] == '3' or strFinalString2[
        1] == '4' or strFinalString2[1] == '5' or strFinalString2[1] == '6' or strFinalString2[1] == '7' or \
            strFinalString2[1] == '8' or strFinalString2[1] == '9':
        repeated_detection_allowance = False
        print('WARNING: incorrect second character in repeated detection - rejecting')
    if len(strFinalString2) < len(strFinalString1):
        repeated_detection_allowance = False
        print('WARNING: incorrect repeated detection result size - rejecting')
    return repeated_detection_allowance

def drawContours(contours, image):
    for c in contours:
        R = randint(1, 255)
        G = randint(1, 255)
        B = randint(1, 255)
        cv2.drawContours(image, c, -1, (B, G, R), 5)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (B, G, R), 3)
        cv2.circle(image, (int(x), int(y)), 20, (255, 0, 0), 3)
        cv2.circle(image, (int(x) + int(w), int(y) + int(h)), 20, (255, 0, 0), 3)
        cv2.circle(image, (int(x) + int(w / 2), int(y) + int(h / 2)), 20, (0, 0, 255), 3)

def drawGroups(groups, image):
    for g in groups:
        R = randint(1, 255)
        G = randint(1, 255)
        B = randint(1, 255)
        for c in g:
            cv2.drawContours(image, c, -1, (B, G, R), 5)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (B, G, R), 3)
            cv2.circle(image, (int(x), int(y)), 20, (255, 0, 0), 3)
            cv2.circle(image, (int(x) + int(w), int(y) + int(h)), 20, (255, 0, 0), 3)
            cv2.circle(image, (int(x) + int(w / 2), int(y) + int(h / 2)), 20, (0, 0, 255), 3)

def resizeImage(image, width):
    image_height, image_width, image_channels = image.shape
    width = int(width)
    height = int(width * image_height / image_width)
    dsize = (width, height)
    result = cv2.resize(image, dsize)
    return result
