import cv2
import math
from random import randint # for random colors only

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

def preliminaryProcessing(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    gray = value
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 5)
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

def printLengthInfo(groups):
    all_contours_number = 0
    group_sizes = []
    for g in groups:
        group_sizes.append(len(g))
        all_contours_number = all_contours_number + len(g)
    print("number of contours in total:", all_contours_number)
    print("groups:", group_sizes)

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