# from __future__ import absolute_import
import tensorflow as tf
from aip import AipOcr

from model.card.net import Net

net = Net()
x = net.x
predictions = net.predictions
save_path = 'resource/model/model.ckp'
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, save_path)

APP_ID = '11186890'
API_KEY = 'WXi67Buab6z5WxNi88haL3KZ'
SECRET_KEY = 'hNgmYCs8RIXFqeR32nqwFe8NKaPiqsuU'
options = {"language_type": "CHN_ENG", "detect_direction": "true", "detect_language": "true", "probability": "true"}

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)  # 实例化AipFace
