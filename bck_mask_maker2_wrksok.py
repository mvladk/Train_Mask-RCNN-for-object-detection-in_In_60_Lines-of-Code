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
from PIL import Image, ImageDraw, ImageTransform, ImageOps, ImageFilter


file_name = "../fonts/SynthText_train.h5"

db = h5py.File(file_name, 'r')
im_names = list(db['data'].keys())
# im = im_names[0]
# im = "maze_25.png_0"
# img = cv2.imread("../fonts/images/maze_25.jpg")
im = "ant+hill_84.jpg_0"
img = cv2.imread("../fonts/images/ant+hill_84.jpg")

# img = cv2.imread("../fonts/images/"+im)
# img = cv2.imread("image_test_font.jpg")
# print(im)
img = db['data'][im][:]
font = db['data'][im].attrs['font']
charBB = db['data'][im].attrs['charBB']

xId = 0
yId = 1
boxId = 1

points = [
    [int(db['data'][im].attrs['charBB'][xId][0][boxId]), int(db['data'][im].attrs['charBB'][yId][0][boxId])],
        [int(db['data'][im].attrs['charBB'][xId][1][boxId]), int(db['data'][im].attrs['charBB'][yId][1][boxId])], 
    [int(db['data'][im].attrs['charBB'][xId][2][boxId]), int(db['data'][im].attrs['charBB'][yId][2][boxId])], 
        [int(db['data'][im].attrs['charBB'][xId][3][boxId]), int(db['data'][im].attrs['charBB'][yId][3][boxId] )]
    ]

points_abla = [
        (int(db['data'][im].attrs['charBB'][xId][0][boxId]), int(db['data'][im].attrs['charBB'][yId][0][boxId])),
        (int(db['data'][im].attrs['charBB'][xId][1][boxId]), int(db['data'][im].attrs['charBB'][yId][1][boxId])), 
        (int(db['data'][im].attrs['charBB'][xId][2][boxId]), int(db['data'][im].attrs['charBB'][yId][2][boxId])), 
        (int(db['data'][im].attrs['charBB'][xId][3][boxId]), int(db['data'][im].attrs['charBB'][yId][3][boxId]))
]


# Open starting image and ensure RGB
# im_pil = Image.open('../fonts/images/ant+hill_102_bb.jpg').convert('RGB')
# im_pil = Image.fromarray(img)

# Define 8-tuple with x,y coordinates of top-left, bottom-left, bottom-right and top-right corners and apply

# transform=[31,146,88,226,252,112,195,31]
transform=[
    int(db['data'][im].attrs['charBB'][xId][0][boxId]), int(db['data'][im].attrs['charBB'][yId][0][boxId]),
    int(db['data'][im].attrs['charBB'][xId][3][boxId]), int(db['data'][im].attrs['charBB'][yId][3][boxId]),
    int(db['data'][im].attrs['charBB'][xId][2][boxId]), int(db['data'][im].attrs['charBB'][yId][2][boxId]),
    int(db['data'][im].attrs['charBB'][xId][1][boxId]), int(db['data'][im].attrs['charBB'][yId][1][boxId] )
    ]

x_max = int(max([points[i][0] for i in range(4)]))
x_min = int(min([points[i][0] for i in range(4)]))
y_max = int(max([points[i][1] for i in range(4)]))
y_min = int(min([points[i][1] for i in range(4)]))

# x_len = x_max - x_min
# y_len = y_max - y_min
# # result = im_pil.transform((x_len,y_len), ImageTransform.QuadTransform(transform))
# result = im_pil.transform((y_len, x_len), ImageTransform.QuadTransform(transform))
# result.show()
# cv2.waitKey(0)
# Save the result
# result.save('result.png')



def make_mask(image, points, im_name):
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


    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # x, y, w, h = cv2.boundingRect(contours[0])
    # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # cv2.rectangle(mask, leftTop, rightBottom, 1, -1)

    # new points for polygon
    # points = [[383, 313], [380, 382], [538, 528], [623, 545], [845, 475], [872, 401], [668, 352], [578, 333]]
    # create and reshape array
    # points = np.array(points)
    # points = points.reshape((-1, 1, 2))


    # cropped_img = image[y_min:y_max, x_min:x_max]
    # cropped_img = image[x_min:x_max, y_min:y_max]
    # cropped_img = image[x_min:x_max, y_min:y_max]
    # cropped_img = image[points[0][1]:points[2][1], points[0][0]:points[2][0]]
    # finalImage = cropedImage[minY:maxY,minX:maxX]
    # print(points)
    # print(points[0])
    # print(points[2])

    # cv2.imshow("cropped_img", cropped_img)
    # cv2.waitKey(0)

    # if cropped_img.any():
    #     gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
# ----------------------------------
        # cv2.imshow("gray", gray) 
        # cv2.waitKey(0)
        # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # x, y, w, h = cv2.boundingRect(contours[0])
        # cv2.rectangle(gray, (x, y), (x + w, y + h), (1, 0, 0), 3)

# ----------------------------------

    #     gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    #     # img_ = cv2.threshold(gray,100,225,cv2.THRESH_BINARY)
        # edges = cv2.Canny(gray, 250, 250)
        # print(edges)
        # cv2.imshow("edges", edges)
        # cv2.waitKey(0)
        # binedge = (edges > 0).astype(np.uint8)
        # ker = np.ones((3, 3))
        # fatedge = cv2.dilate(binedge, ker)
        # print(fatedge)
        # cv2.imshow("fatedge", fatedge)
        # cv2.waitKey(0)
        # n, comp = cv2.connectedComponents((fatedge == 0).astype(np.uint8))
        # print(comp)
        # cv2.waitKey(0)
        # filled = (comp != comp[0, 0]).astype(np.uint8)
        # output = cv2.erode(filled, ker) * 255
        # cv2.imshow("output", output)
        # cv2.waitKey(0)
    #     # cv2.imwrite('output.png', output)
    #     # cv2.fillPoly(mask, pts=[points], color=(1, 0, 0))
    #     backtorgb = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    #
        # mask[points[0][1]:points[2][1], points[0][0]:points[2][0]] = gray
        # mask[y_min:y_max, x_min:x_max] = gray

# ----------------------------------

        # Open the image
        # img = Image.open("image.jpg")

        # Define the polygon vertices
        # points = [(10, 10), (100, 10), (100, 100), (10, 100)]

        # Create a mask image
    # try:
    #     im_pil = Image.fromarray(image)
    # except Exception as e:
    #     print(e)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize https://stackoverflow.com/questions/46260601/convert-image-from-cv-64f-to-cv-8u
    # gray = np.uint16(gray)
    # Define the polygon vertices
    points_np = np.array(points, np.int32)
    # Create a mask image
    mask = np.zeros(gray.shape, np.uint8)
    # filler= 255
    filler= 1
    cv2.fillPoly(mask, [points_np], filler)
    # Apply the mask to the grayscale image
    # cv2.imshow("gray: ", gray)
    # cv2.waitKey(0)
    # cv2.imshow("mask: ", mask)
    # cv2.waitKey(0)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    # cv2.imshow("masked_grayaaa: ", masked_gray)
    masked_gray = masked_gray.astype("uint8")
    blur = cv2.GaussianBlur(masked_gray,(5,5),0)
    # cv2.imshow("masked_gray: ", masked_gray)
    # cv2.imshow("blur: ", blur)
    # cv2.waitKey(0)
    cropped_im = np.zeros(gray.shape, np.uint8)
    # Threshold the masked grayscale image
    try:
        # cv2.imshow("masked_gray: ", masked_gray)
        # cv2.waitKey(0)
        _, thresholded = cv2.threshold(blur, 0, filler, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # cv2.imshow("thresholded: ", thresholded)
        # cv2.waitKey(0)
        # if thresholded.any():
        # Perform morphological operations to remove noise
        kernel = np.ones((3,3), np.uint8)
        # cv2.imshow("kernel: ", kernel)
        # cv2.waitKey(0)

        erosion = cv2.erode(thresholded, kernel, iterations=1)
        # cv2.imshow("erosion: ", erosion)
        # cv2.waitKey(0)
        # Copy the thresholded text to the empty image
        cropped_im[mask == filler] = erosion[mask == filler]
    except Exception as e:
        print("----small badaboom ----")
        print(e)
        print(im_name)
        exit()
        # try:
        #     _, thresholded = cv2.threshold(np.uint16(masked_gray), 0, filler, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #     if thresholded.any():
        #         # Perform morphological operations to remove noise
        #         kernel = np.ones((3,3), np.uint8)
        #         erosion = cv2.erode(thresholded, kernel, iterations=1)
        #         # Copy the thresholded text to the empty image
        #         cropped_im[mask == filler] = erosion[mask == filler]
        # except Exception as e2:
        #     print("----big badaboom ----")
        #     print(e2)

    # Create an empty image with the same size as the grayscale image

    
    # im_pil = Image.open("image_test_font.jpg")
    
    # mask = Image.new("1", im_pil.size)
    # draw = ImageDraw.Draw(mask)
    # # points_gpt = [(10, 10), (100, 10), (100, 100), (10, 100)]
    # draw.polygon(points_abla, fill=255)
    # # Dilation
    # mask = mask.filter(ImageFilter.MaxFilter(5))
    # Apply the mask to the image
    # mask = ImageOps.invert(mask)
    # im_pil = im_pil.crop(im_pil.getbounds())
    # im_pil.putalpha(mask)
    # bbox = mask.getbbox()

    # Crop the image to the bounds of the polygon
    # cropped_im = im_pil.crop(bbox)
    # Create an empty image with the same size as the mask
    # cropped_im = Image.new("RGB", mask.size, (0, 0, 0, 0))

    # # Paste the original image onto the empty image using the mask
    # cropped_im.paste(im_pil, (0, 0), mask)
    # cv2.imshow("cropped_im", cropped_im)
    # # cropped_im.show()
    # cv2.waitKey(0)

# ----------------------------------

        # mask[y_min:y_max, x_min:x_max] = im_pil
        # mask[x_min:x_max, y_min:y_max] = gray
    
    # cv2.imshow("Rectangular Mask", mask)
    # apply our mask -- notice how only the person in the image is
    # cropped out

# masked = cv2.bitwise_and(image, output, mask=None)

    # print(mask)

    # cv2.imshow("Mask Applied to Image", mask)
    # cv2.waitKey(0)
    # print()
    # cv2.imshow("origin", image)
    # cv2.waitKey(0)
    # cv2.imshow("Mask Applied to Image", cropped_img)
    # cv2.waitKey(0)
    return cropped_im



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


# maskk = make_mask(img, points, im)
