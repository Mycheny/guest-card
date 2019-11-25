import sys

sys.path.append("..")
from socket import *
import json, threading
from websocket import create_connection


# config = {
#     'HOST': '127.0.0.1',
#     'PORT': 10010
# }
# pip install websocket-client

class Client():
    def __init__(self):
        # 调用create_connection方法，建立一个websocket链接,链接是自己的链接
        self.ws = create_connection("ws://127.0.0.1:10000/websocket")
        # 建一个线程，监听服务器发送给客户端的数据
        self.trecv = threading.Thread(target=self.recv)
        self.trecv.start()

    # 发送方法，聊天输入语句时调用，此处默认为群聊ALL
    def send(self, content):
        # 这里的msg要根据实际需要自己写
        msg = {
            "type": "POST",
            "content": content
        }
        msg = json.dumps(msg)
        self.ws.send(msg)

    # 接收服务端发送给客户的数据，只要ws处于连接状态，则一直接收数据
    def recv(self):
        try:
            while self.ws.connected:
                result = self.ws.recv()
                print("received msg:" + str(result))
        except :
            pass


if __name__ == '__main__':
    content = ""
    c = Client()
    # 建立链接后，就可以按照需要自己send了
    c.send(content)