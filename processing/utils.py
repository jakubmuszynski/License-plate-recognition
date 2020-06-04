import numpy as np
import cv2
import math
import time

from random import randint

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

def preliminaryProcessing(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    gray = value
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return thresh, contours

def selectionContourRatios(image, contours):
    result = []
    image_height, image_width, image_channels = image.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        offset_min = 5
        offset_max = 15
        prop_w_max = image_width * 426 / 4000 + offset_max    # 0.101 + offset
        prop_h_max = image_width * 636 / 3000 + offset_max    # 0.212 + offset
        prop_w_min = image_width * 25 / 4000 - offset_max     # 0.006 - offset (letter 'I')
        prop_h_min = image_height * 214 / 3000 - offset_min   # 0.071 - offset
        if w > prop_w_min and h > prop_h_min:
            if w < prop_w_max and h < prop_h_max:
                if (w / h) > 0.12 and (w / h) < 0.88:
                    result.append(c)
    return result

def selectionDistanceAngle(image, contours):
    result = []
    image_height, image_width, image_channels = image.shape
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

def selectionAverageHeight(contours, difference):
    result = []
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

def discardInnerContours(contours):
    offset = 5
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

def group(contours, radius):
    friends = [contours[0]]
    friend = contours[0]
    contours.remove(contours[0])
    i = 0
    exit = 0
    while exit != 1:
        min_distance = radius * 10
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            distance = calculateDistance(friends[i], c)
            if distance < min_distance:
                min_distance = distance
                friend = c
        if min_distance < radius:
            friends.append(friend)
            contours.remove(friend)
            i += 1
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

def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here

    # start measuring time
    start = time.time()

    # get input image dimensions
    image_height, image_width, image_channels = image.shape

    # initial image processing
    thresh, contours = preliminaryProcessing(image)

    # selecting contours - width/image_width and height/image_height ratios
    contours = selectionContourRatios(image, contours)

    # selecting contours - minimum 1 close neighbour, angle between them
    contours = selectionDistanceAngle(image, contours)

    # selecting contours - difference from average height
    contours = selectionAverageHeight(contours, image_height * 0.045)

    # selecting contours - discarding duplicates
    contours = discardDuplicates(contours)

    # selecting contours - discarding inner contours
    contours = discardInnerContours(contours)

    # sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    print("number of contours:", len(contours))

    contours = grouping(contours, image_width * 0.12)

    print("number of groups:", len(contours))

    leng = 0
    for c in contours:
        for c1 in c:
            leng += 1
    print("number of contours in groups:", leng)

    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    for c in contours:
        r = randint(1, 255)
        g = randint(1, 255)
        b = randint(1, 255)
        for c1 in c:
            cv2.drawContours(thresh, c1, -1, (b, g, r), 5)
            x, y, w, h = cv2.boundingRect(c1)
            cv2.rectangle(thresh, (int(x), int(y)), (int(x + w), int(y + h)), (b, g, r), 3)
            cv2.circle(thresh, (int(x), int(y)), 20, (255, 0, 0), 3)
            cv2.circle(thresh, (int(x) + int(w), int(y) + int(h)), 20, (255, 0, 0), 3)
            cv2.circle(thresh, (int(x) + int(w / 2), int(y) + int(h / 2)), 20, (0, 0, 255), 3)

    end = time.time()
    print('I worked for', format((end - start), '.2f'), 'seconds')
    print(' ')

    scale_percent = 40
    width = int(image_width * scale_percent / 100)
    height = int(image_height * scale_percent / 100)
    dsize = (width, height)

    thresh = cv2.resize(thresh, dsize)

    cv2.imshow('image', thresh)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 'XD'
