import cv2
import numpy as np


def imageBlockTrans_Debug(scr, DCBias=0):
    rowblocknum = (int)(scr.shape[0] / 8)
    colblocknum = (int)(scr.shape[1] / 8)

    des = np.zeros((rowblocknum * 8, colblocknum * 8))
    for i in range(rowblocknum):
        for j in range(colblocknum):
            des[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = cv2.dct(scr[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8])
            # ignore the DC coefficients
            # des[i*8,j*8] = des[i*8,j*8] + DCBias
            des[i * 8, j * 8] = 0
    return des


# - scr: a 1-channel 2D numpy array,
def imageBlockTrans(scr, DCBias=0):
    rowblocknum = (int)(scr.shape[0] / 8)
    colblocknum = (int)(scr.shape[1] / 8)

    channel = []
    for i in range(rowblocknum):
        for j in range(colblocknum):
            temp = cv2.dct(scr[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8])
            temp[0,0] = 0.
            channel = channel + [temp]

    return np.stack(channel)
    # return channels


def imageTrans(scr, DCBias=0):
    des = cv2.dct(scr)
    # des[0, 0] = des[0, 0] + DCBias
    des[0, 0] = 0
    return des

