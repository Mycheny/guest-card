# -*- coding=utf-8 -*-
"""
api接口控制
"""
import base64
import re

import cv2
import numpy as np
from flask import Flask, request, json, render_template

from service.discern import judge
from util.config import logger

app = Flask(__name__, static_folder='./templates', static_url_path='')


@app.route('/')
def hello_world():
    ip = request.remote_addr
    return render_template('index.html', user_ip=ip)


def get_data(strs):
    if 'image' in strs.keys():
        image = strs['image']
        image = re.sub(r'data(.|\n)*base64,', "", image)
        if image and isinstance(image, str):
            image = base64.b64decode(image)
            image = np.asarray(bytearray(image), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            result = judge(image)
            logger.info(result)
            return json.dumps(result, ensure_ascii=False)
        else:
            message = 'The value is null or invalid'
            logger.error(message)
            return json.dumps({'error_code': 102, 'error_msg': message}, ensure_ascii=False)
    else:
        message = 'KeyError: The key is image instead of {0}'.format(list(strs.keys())[0], )
        logger.error(message)
        return json.dumps({'error_code': 101, 'error_msg': message}, ensure_ascii=False)


@app.route('/xiaoi/identify', methods=['POST'])
def dispose():
    if request.data:
        strs = json.loads(request.data)
        return get_data(strs)
    elif request.values:
        strs = request.values
        return get_data(strs)
    else:
        message = 'The data is invalid'
        logger.error(message)
        return json.dumps({'error_code': 100, 'error_msg': message}, ensure_ascii=False)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8094, ssl_context='adhoc')
    app.run(host='0.0.0.0', port=8093)
