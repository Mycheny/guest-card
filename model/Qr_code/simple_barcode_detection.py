import numpy as np
import cv2
import os
import math


def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    # define local variables for readability

    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer

def cv_distance(P, Q):
    return int(math.sqrt(pow((P[0] - Q[0]), 2) + pow((P[1] - Q[1]), 2)))

def isTimingPattern(line):
    # 除去开头结尾的白色像素点
    while line[0] != 0:
        line = line[1:]
    while line[-1] != 0:
        line = line[:-1]
    # 计数连续的黑白像素点
    c = []
    count = 1
    l = line[0]
    for p in line[1:]:
        if p == l:
            count = count + 1
        else:
            c.append(count)
            count = 1
        l = p
    c.append(count)
    # 如果黑白间隔太少，直接排除
    if len(c) < 5:
        return False
    # 计算方差，根据离散程度判断是否是 Timing Pattern
    threshold = 5
    result = np.var(c)
    return result < threshold

def detect(image):
    # 把图像从RGB装换成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(gray, 100, 200)
    # cv2.imshow('img',edges)
    # cv2.waitKey(0)
    img_fc, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]
    found = []
    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c + 1
        if c >= 5:
            found.append(i)

    boxes = []
    for i in found:
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = tuple(box)
        boxes.append(box)


    valid = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if len(box)>3:
                valid.add(i)
                valid.add(j)

    valid = list(valid)
    if len(valid)>2:
        contour_all = list()
        for i in range(len(valid)):
            c = found[valid.pop()]
            for su in contours[c]:
                contour_all.append(su)
        contour_all= np.array(contour_all)
        rect = cv2.minAreaRect(contour_all)
        box = cv2.boxPoints(rect)
        box = np.array(box)
        return box
    else:
        return None


def open_image():
    img = cv2.imread(r"C:\Users\User\Desktop\demo\0.jpg")
    # img = cv2.resize(img,(399,260))
    if img is not None:
        cropImg = detect(img)
        # cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
        # cv2.imshow("Image", cropImg)
        # cv2.imwrite("contoursImage2.jpg", img)
        # cv2.waitKey(0)
    else:
        print("图片无法打开！")


if __name__ == '__main__':
    open_image()