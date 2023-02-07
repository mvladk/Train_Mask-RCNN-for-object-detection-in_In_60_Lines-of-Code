# import the necessary packages
import numpy as np
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# file_name = "fonts/SynthText_train.h5"

# db = h5py.File(file_name, 'r')
# im_names = list(db['data'].keys())
# im = im_names[0]
# # img = cv2.imread("fonts/images/ant+hill_102_bb.jpg")
# img = cv2.imread("fonts/images/ant+hill_102.jpg")
# # print(im)
# # img = db['data'][im][:]
# font = db['data'][im].attrs['font']
# charBB = db['data'][im].attrs['charBB']

# xId = 0
# yId = 1
# boxId = 1

# points = [
#     [int(db['data'][im].attrs['charBB'][xId][0][boxId]), int(db['data'][im].attrs['charBB'][yId][0][boxId])],
#         [int(db['data'][im].attrs['charBB'][xId][1][boxId]), int(db['data'][im].attrs['charBB'][yId][1][boxId])], 
#     [int(db['data'][im].attrs['charBB'][xId][2][boxId]), int(db['data'][im].attrs['charBB'][yId][2][boxId])], 
#         [int(db['data'][im].attrs['charBB'][xId][3][boxId]), int(db['data'][im].attrs['charBB'][yId][3][boxId] )]
#     ]


def make_mask(image, points):
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", type=str, default="Image.jpg",
    #     help="path to the input image")
    # args = vars(ap.parse_args())
    # load the original input image and display it to our screen
    # image = cv2.imread(args["image"])
    # cv2.imshow("Original", image)
    # a mask is the same size as our image, but has only two pixel
    # values, 0 and 255 -- pixels with a value of 0 (background) are
    # ignored in the original image while mask pixels with a value of
    # 255 (foreground) are allowed to be kept
    mask = np.zeros(image.shape[:2], dtype="uint8")
    
    # cv2.rectangle(mask, leftTop, rightBottom, 1, -1)

    # new points for polygon
    # points = [[383, 313], [380, 382], [538, 528], [623, 545], [845, 475], [872, 401], [668, 352], [578, 333]]
    # create and reshape array
    points = np.array(points)
    points = points.reshape((-1, 1, 2))

    # Attributes
    # isClosed = True
    # color = (255, 0, 0)
    # thickness = 1

    # draw closed polyline
    # cv2.polylines(mask, [points], isClosed, color, thickness)
    # cv2.fillPoly(mask, [points], isClosed, color, thickness)
    cv2.fillPoly(mask, pts=[points], color=(1, 0, 0))

    
    # cv2.imshow("Rectangular Mask", mask)
    # apply our mask -- notice how only the person in the image is
    # cropped out
    # masked = cv2.bitwise_and(image, image, mask=mask)

    # print(mask)

    # cv2.imshow("Mask Applied to Image", masked)
    # cv2.waitKey(0)
    return mask



# def make_mask(image, leftTop, rightBottom):
#     # construct the argument parser and parse the arguments
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-i", "--image", type=str, default="Image.jpg",
#         help="path to the input image")
#     args = vars(ap.parse_args())
#     # load the original input image and display it to our screen
#     image = cv2.imread(args["image"])
#     cv2.imshow("Original", image)
#     # a mask is the same size as our image, but has only two pixel
#     # values, 0 and 255 -- pixels with a value of 0 (background) are
#     # ignored in the original image while mask pixels with a value of
#     # 255 (foreground) are allowed to be kept
#     mask = np.zeros(image.shape[:2], dtype="uint8")
#     # cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
#     cv2.rectangle(mask, (0, 0), (20, 20), 1, -1)
#     # cv2.rectangle(mask, (0, 0), (20, 20), 1, -1)
#     cv2.imshow("Rectangular Mask", mask)
#     # apply our mask -- notice how only the person in the image is
#     # cropped out
#     masked = cv2.bitwise_and(image, image, mask=mask)

#     # np.set_printoptions(threshold=np.inf, linewidth=140)
#     print(mask)

#     cv2.imshow("Mask Applied to Image", masked)
#     cv2.waitKey(0)
# return mask

# # now, let's make a circular mask with a radius of 100 pixels and
# # apply the mask again
# mask = np.zeros(image.shape[:2], dtype="uint8")
# cv2.circle(mask, (145, 200), 100, 255, -1)
# masked = cv2.bitwise_and(image, image, mask=mask)
# # show the output images
# cv2.imshow("Circular Mask", mask)
# cv2.imshow("Mask Applied to Image", masked)
# cv2.waitKey(0)


# maskk = make_mask(img, points)
# print(maskk)


