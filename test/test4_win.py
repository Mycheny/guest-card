#! /usr/bin/python
# -*- coding: utf8 -*-
# The QR reader module to decode QR codes.
#
import pyzbar.pyzbar as pyzbar

import time
import cv2
import os
import re
import sys

class qr_reader():
    '''
    class to match the QR image with
    specific information.
    '''
    def __init__(self):
        pass

    def image_qr_read(self, img): # filename):
        qr_result = ""
        '''
        Read the QR image data from the image.
        '''
        decodedObjects = pyzbar.decode(img)
        # Print results
        for obj in decodedObjects:
           print('Type : ', obj.type)
           qr_result = obj.data
           print('Data : ', obj.data,'\n')
        '''
        qr_data = QR(filename=filename)
        if qr_data.decode():
            print(qr_data.data)
            #print(qr_data.data_type)
            qr_result = qr_data.data_to_string().decode("utf-8-sig")
		'''
        return qr_result

    def webcam_qr_read(self):
        cv2.namedWindow("neoQR-preview")
        vc = cv2.VideoCapture(0)

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False

        qr_result = ""
        while rval:
            cv2.imshow("neoQR-preview", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
            tempfile = "temp.jpg" #self.os_lib.getImageName()
            cv2.imwrite(tempfile, frame)
            img = cv2.imread(tempfile) 
            qr_result = self.image_qr_read(img) #cache)
            time.sleep(0.1)
            #self.os_lib.remove_file(cache)
            if qr_result:
                # Exit on valid QR code result
                continue
        cv2.destroyWindow("neoQR-preview")
        # Bug in opencv, need to call waitkey with imshow to kill the window
        # completely. Otherwise the window get closed only when program exits.
        cv2.waitKey(-1)
        cv2.imshow("neoQR-preview", frame)
        #self.os_lib.remove_file_matched_ext(curr_dir, 'jpg')
        return qr_result

if __name__ == "__main__":
    #qr_reader().image_qr_read("/home/sugesh/sree.jpg")
    qr_reader().webcam_qr_read()

