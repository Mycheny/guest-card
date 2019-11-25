import platform
system = platform.system()
print(system)
if system == "Windows":
    import pyzbar.pyzbar as zbar
elif system == "Linux":
    import zbar as zbar
import cv2


# 定义旋转rotate函数
def rotate(image, angle, center=None, scale=0.9):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated


image = cv2.imread('timg2.jpg')
image = cv2.resize(image, (224, 224))
# ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# win10
if system == "Windows":
    results = zbar.decode(image)
    for result in results:
        a = result.data
        b = a.decode()
        print(result.data.decode())

# linux
if system == "Linux":
    ret, image = cv2.threshold(image, 12, 255, cv2.THRESH_BINARY)
    scanner = zbar.Scanner()
    results = scanner.scan(image)
    for result in results:
        a = result.data
        b = a.decode()
        print(result.data.decode())


