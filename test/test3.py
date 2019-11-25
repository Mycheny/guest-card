import base64
import urllib.parse, urllib.request

import cv2

host = 'http://qrapi.market.alicloudapi.com'
path = '/yunapi/qrdecode.html'
method = 'POST'
appcode = '685bae52043442348e056aec5ca7b919'
querys = ''
bodys = {}
url = host + path

fream = cv2.imread('timg1.jpg')
ret, fream = cv2.imencode('.jpg', fream)
fream = fream.tobytes()
image = base64.b64encode(fream).decode()

bodys['imgurl'] = 'http://www.wwei.cn/static/images/qrcode.jpg'
bodys['imgdata'] = 'data:image/jpeg;base64,'+image
bodys['version'] = '1.1'
post_data = urllib.parse.urlencode(bodys).encode(encoding='UTF8')
request = urllib.request.Request(url, post_data)
request.add_header('Authorization', 'APPCODE ' + appcode)
request.add_header('Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8')
response = urllib.request.urlopen(request)
content = response.read()
if (content):
    print(content)