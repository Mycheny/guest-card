import platform

import cv2
import numpy as np
import base64
import urllib.parse, urllib.request

from model.Qr_code import simple_barcode_detection
from service import predictions, sess, x, client, options


def judge(image):
    """
    处理图片
    :param image:
    :return:
    """
    text = get_qr(image)
    if not text:
        box = simple_barcode_detection.detect(image)
        if box is not None:
            xs = box[:, 0]
            ys = box[:, 1]
            x_max = int(min(np.max(xs) + 10, image.shape[1]))
            x_min = int(max(np.min(xs) - 10, 0))
            y_max = int(min(np.max(ys) + 10, image.shape[0]))
            y_min = int(max(np.min(ys) - 10, 0))

            # cv2.polylines(image, np.int32([box]), True, (0, 0, 255), 10)
            crop_img = image[y_min:y_max, x_min:x_max]

            # cv2.imshow("Frame", crop_img)
            # cv2.waitKey(1)
            text = get_qr(crop_img)

            if text:
                return {'result': 0, 'text': text}
            else:
                return {'error_code': 103, 'error_msg': 'Invalid qr code'}
        else:
            fream = cv2.resize(image, (200, 130))
            fream = cv2.cvtColor(fream, cv2.COLOR_BGR2GRAY)
            fream = fream.reshape((1, fream.shape[0], fream.shape[1], 1)) / 255
            p = sess.run(predictions, feed_dict={x: fream})
            p = np.reshape(p, (2,))
            print(p)
            if p[0:1] > 0.5:
                data = get_text(image)
                return {'result': 1, 'text': data}
            elif p[0:1] > 0.2:
                res = find_card(image)
                return {'result': -1, 'text': res}
            else:
                return {'result': -1, 'text': 'No qr code or certificate'}
    else:
        return {'result': 0, 'text': text}


def find_card(image):
    # 红色和紫色范围
    lower1 = np.array([0, 48, 45])
    upper1 = np.array([5, 255, 255])
    lower2 = np.array([125, 48, 45])
    upper2 = np.array([180, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(hsv, lower1, upper1)
    mask_2 = cv2.inRange(hsv, lower2, upper2)
    output = cv2.bitwise_and(hsv, hsv, mask=mask_1 + mask_2)
    # 根据阈值找到对应颜色
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    Matrix = np.ones((20, 20), np.uint8)
    img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)

    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
    # print(np.max(img_edge2))
    ret, binary = cv2.threshold(img_edge2, 97, 255, cv2.THRESH_BINARY)

    # 找到轮廓
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        res = contours[0]
        if res.shape[0] > 30:
            res = res.reshape((res.shape[0], res.shape[2]))
            x_min = np.max(res[:, 0], axis=1)
            return 11
        else:
            return 1
    else:
        return 1


def get_qr(image):
    """
    解析二维码
    :param image:
    :return:
    """
    system = platform.system()
    results, qr_result = [], []
    if system == "Windows":
        import pyzbar.pyzbar as zbar
        results = zbar.decode(image)
    elif system == "Linux":
        import zbar as zbar
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scanner = zbar.Scanner()
        results = scanner.scan(image)

    for result in results:
        qr_result.append(result.data.decode())
    return qr_result


def get_text(image):
    """
    文字识别
    :param image:
    :return:
    """
    ret, flow = cv2.imencode('.jpg', image)
    text = client.basicGeneral(flow.tobytes(), options)
    # text = '这是证件'
    return text


def get_qr_by_api(image):
    """
    通过api解析二维码
    :param image:
    :return:
    """
    host = 'http://qrapi.market.alicloudapi.com'
    path = '/yunapi/qrdecode.html'
    appcode = '685bae52043442348e056aec5ca7b919'
    bodys = {}
    url = host + path

    # fream = cv2.imread('timg1.jpg')
    ret, image = cv2.imencode('.jpg', image)
    image = image.tobytes()
    image = base64.b64encode(image).decode()

    bodys['imgurl'] = 'http://www.wwei.cn/static/images/qrcode.jpg'
    bodys['imgdata'] = 'data:image/jpeg;base64,' + image
    bodys['version'] = '1.1'
    post_data = urllib.parse.urlencode(bodys).encode(encoding='UTF8')
    request = urllib.request.Request(url, post_data)
    request.add_header('Authorization', 'APPCODE ' + appcode)
    request.add_header('Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8')
    response = urllib.request.urlopen(request)
    content = response.read()
    return content
