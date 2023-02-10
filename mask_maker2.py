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
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image, ImageDraw, ImageTransform, ImageOps, ImageFilter


def set_zero_except_max(image):
    max_value = np.max(image)
    max_value_less = max_value - 45
    image[image < max_value_less] = 0
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def maximize_intensity(image):
    # Normalize the image intensity to the range [0, 255]
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return image.astype(np.uint8)

#  check if the background of an image is lighter than the text, you can compute the average intensity of the image and compare it to a threshold value. If the average intensity is greater than the threshold, you can invert the image to make the text lighter.
# In this code, the input image is first converted to grayscale using the np.dot function, which computes the dot product of the image and a weighting matrix to produce a single-channel grayscale image. The average intensity is then computed using the np.mean function, and compared to a threshold of 128. If the average intensity is greater than the threshold, the image is inverted by subtracting all pixel values from 255. The inverted image is then returned.
# def invert_if_background_lighter(gray_image):
#     # Convert the image to grayscale
#     # gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
#     # Compute the average intensity
#     gray_image = maximize_intensity(gray_image)
#     avg_intensity = np.mean(gray_image)
#     # Set a threshold for the average intensity
#     # threshold = 128
#     threshold = 98
#     # Invert the image if the average intensity is greater than the threshold
#     if avg_intensity > threshold:
#         gray_image = 255 - gray_image
#     return gray_image
# def invert_if_background_lighter(gray_image):
#     # Convert the image to grayscale
#     # gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
#     gray_image = maximize_intensity(gray_image)
#     # avg_intensity = np.mean(gray_image)
#     # Compute the histogram of the image intensity
#     hist, _ = np.histogram(gray_image, bins=256, range=(0, 255))
#     # Find the threshold that separates the background from the text
#     threshold = np.argmax(hist[:128])
#     # Invert the image if the threshold is above 128
#     print(hist)
#     print(hist[:128])
#     print(threshold)
#     if threshold < 0:
#         gray_image = 255 - gray_image
#     return gray_image

# def invert_if_background_lighter(gray_image, mask):
#     # Convert the image to grayscale
#     # gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
#     # Compute the histogram of the image intensity
#     mean_intensity = np.mean(gray_image[mask > 0])
#     print(mean_intensity)
#     # Invert the image if the mean intensity is above 128
    
#     hist, _ = np.histogram(gray_image[mask > 0], bins=256, range=(0, 255))
#     # Compute the cumulative distribution function (CDF) of the histogram
#     cdf = np.cumsum(hist) / np.sum(hist)
#     # Find the threshold that separates the background from the text
#     threshold = np.argmax(cdf < 0.5)
#     print(cdf)
#     print(np.argmax(cdf < 0.5))
#     print(threshold)
#     # Invert the image if the threshold is above 128
#     # if threshold > 128:
#     if threshold > 128:
#         gray_image = 255 - gray_image

#     return gray_image
# def invert_if_background_lighter(gray_image, polygon_points):
#     # Convert the image to grayscale
#     # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Create a binary mask that is 1 inside the polygon and 0 outside
#     mask = np.zeros_like(gray_image)
#     polygon = np.array(polygon_points, np.int32)
#     cv2.fillPoly(mask, [polygon], 1)
#     mean_intensity = np.mean(gray_image[mask > 0])
#     # Compute the histogram of the region defined by the mask
#     histogram, _ = np.histogram(gray_image[mask > 0], bins=256, range=(0, 255))
#     mean_intensity_hist = np.mean(histogram)
#     # Compute the cumulative distribution function
#     cdf = histogram.cumsum()
#     mean_cdf = np.mean(cdf)
#     # Normalize the CDF
#     cdf = cdf / cdf[-1]
#     # Invert the image if the background (0th percentile) is lighter than the text (100th percentile)
#     if cdf[0] < cdf[-1]:
#         gray_image = 255 - gray_image

#     print(cdf)
#     print(histogram)
#     print(cdf[0] < cdf[-1])
#     print(cdf[0])
#     print(cdf[-1])
#     print(mean_intensity)
#     print(mean_intensity_hist)
#     print(mean_cdf)
#     return gray_image

def invert_if_background_lighter(gray_image, polygon_points):
    # Convert the image to grayscale
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create a binary mask that is 1 inside the polygon and 0 outside
    mask = np.zeros_like(gray_image)
    polygon = np.array(polygon_points, np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    # Compute the mean intensity of the region defined by the mask
    mean_intensity = np.mean(gray_image[mask > 0])
    # Compute the mean intensity of the background (0 intensity)
    mean_background = np.mean(gray_image[mask == 0])
    # Invert the image if the background mean intensity is lighter than the text mean intensity
    # if mean_background < mean_intensity:
    trash_background_diff = 24
    if mean_background > mean_intensity or \
        ( mean_intensity - mean_background < trash_background_diff ):
        gray_image = 255 - gray_image

    # print(cdf)
    # print(histogram)
    # print(cdf[0] < cdf[-1])
    # print(cdf[0])
    # print(cdf[-1])
    # print(mean_background > mean_intensity or \
    #     ( mean_intensity - mean_background < trash_background_diff ))
    # print(mean_intensity)
    # print(mean_background)
    # print(mean_intensity_hist)
    # print(mean_cdf)
    return gray_image

# def invert_if_background_lighter(gray_image, mask):
#     # Convert the image to grayscale
#     # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Create a binary mask that is 1 inside the polygon and 0 outside
#     # mask = np.zeros_like(gray_image)
#     # polygon = np.array(polygon_points, np.int32)
#     # cv2.fillPoly(mask, [polygon], 1)
#     # Compute the mean intensity within the polygon
#     mean_intensity = np.mean(gray_image[mask > 0])
#     print(mean_intensity)
#     # Invert the image if the mean intensity is above 128
#     if mean_intensity > 128:
#         gray_image = 255 - gray_image
#     return gray_image

# In this code, the input image is first converted to grayscale using the convert method with the argument 'L'. The grayscale image is then converted to a numpy array for easy computation. The average intensity is computed using the np.mean function and compared to a threshold of 128. If the average intensity is greater than the threshold, the image is inverted by subtracting all pixel values from 255. Finally, the inverted image is converted back to a PIL image and returned.
# The error message "AttributeError: 'numpy.ndarray' object has no attribute 'convert'" indicates that the input image is a numpy array, but it is being treated as a PIL image. The convert method is only available for PIL images, not for numpy arrays.
# def invert_if_background_lighter(image):
#     # Convert the image to grayscale
#     gray_image = image.convert('L')
#     # Convert the image to a numpy array
#     gray_image = np.array(gray_image)
#     # Compute the average intensity
#     avg_intensity = np.mean(gray_image)
#     # Set a threshold for the average intensity
#     threshold = 128
#     # Invert the image if the average intensity is greater than the threshold
#     if avg_intensity > threshold:
#         gray_image = 255 - gray_image
#     # Convert the inverted image back to a PIL image
#     inverted_image = Image.fromarray(gray_image)
#     return inverted_image


def make_mask(image, points, im_name):
    mask_canvas = np.zeros(image.shape[:2], dtype="uint8")
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize https://stackoverflow.com/questions/46260601/convert-image-from-cv-64f-to-cv-8u
    # Define the polygon vertices
    points_np = np.array(points, np.int32)
    # Create a mask image
    mask_canvas = np.zeros(gray.shape, np.uint8)
    # filler= 255
    filler = 255
    cv2.fillPoly(mask_canvas, [points_np], filler)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_canvas)
    # count avg of maskfilled pixels in array
    # if count of items bigger than avg is more than half than probably it is white background
    # grey_inverted = gray  # cv2.bitwise_not(gray)
    grey_inverted = invert_if_background_lighter(gray, points_np)
    # if is_bg_lighter(gray):
    #     grey_inverted = cv2.bitwise_not(gray)

    masked_gray_inverted = cv2.bitwise_and(grey_inverted,
                                           grey_inverted,
                                           mask=mask_canvas)
    # masked_gray = cv2.convertScaleAbs(masked_gray)
    blur = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    # cv2.imshow("masked_gray: ", masked_gray)
    # cv2.imshow("blur: ", blur)
    # cv2.imshow("mask: ", mask)

    # cv2.waitKey(0)
    cropped_im = np.zeros(gray.shape, np.uint8)

    # Set the adaptive thresholding (gasussian) parameters:
    windowSize = 41
    windowConstant = -1
    # Apply the threshold:
    masked_gray_inverted = cv2.convertScaleAbs(masked_gray_inverted)
    binaryImage = cv2.adaptiveThreshold(masked_gray_inverted, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, windowSize,
                                        windowConstant)
    # binaryImageInv = cv2.invert(binaryImage)
    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
    cv2.connectedComponentsWithStats(binaryImage, connectivity=4)

    # Set the minimum pixels for the area filter:
    minArea = 20

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [
        i for i in range(1, componentsNumber)
        if componentStats[i][4] >= minArea
    ]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(
        np.isin(labeledImage, remainingComponentLabels) == True, 255,
        0).astype('uint8')
    # cv2.imshow("filteredImage", filteredImage)
    filteredImageBitwise = cv2.bitwise_and(masked_gray_inverted, filteredImage)
    # cv2.imshow("filteredImageBitwise", filteredImageBitwise)
    # max_val = filteredImageBitwise.max()
    filteredImageBitwiseMaxMin = set_zero_except_max(filteredImageBitwise)
    # cv2.imshow("filteredImageBitwiseMaxMin", filteredImageBitwiseMaxMin)

    # # Set kernel (structuring element) size:
    # kernelSize = 3

    # # Set operation iterations:
    # opIterations = 1

    # # Get the structuring element:
    # maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

    # # Perform closing:
    # closingImage = cv2.morphologyEx(filteredImage, cv2.MORPH_CLOSE, maxKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    # cv2.imshow("closingImage", closingImage)
    # closingImageBitwise = cv2.bitwise_and(masked_gray_inverted, closingImage)
    # cv2.imshow("closingImageBitwise", closingImageBitwise)

    binaryImageInv = cv2.bitwise_and(masked_gray_inverted, binaryImage)
    # binaryImageInv = cv2.bitwise_and(binaryImageInv, binaryImage)
    # binaryImageInv = cv2.bitwise_not(binaryImageInv)
    # cv2.imshow("mask", mask)
    # cv2.imshow("masked_gray_inverted", masked_gray_inverted)

    # cv2.imshow("binaryImage", binaryImage)
    # cv2.imshow("binaryImageInv", binaryImageInv)
    # cv2.fillPoly(mask, [points_np], filler)
    masked_gray_filtered = cv2.bitwise_and(binaryImageInv,
                                           binaryImageInv,
                                           mask=mask_canvas)
    # cv2.imshow("masked_gray_filtered", masked_gray_filtered)
    # cv2.waitKey(0)

    # exit()
    # edges = cv2.Canny(masked_gray, 250, 250)
    # # print(masked_gray)
    # cv2.imshow("mask", masked_gray)
    # cv2.imshow("edges", edges)
    # # cv2.waitKey(0)
    # binedge = (edges > 0).astype(np.uint8)
    # ker = np.ones((3, 3))
    # fatedge = cv2.dilate(binedge, ker)
    # # print(fatedge)
    # cv2.imshow("fatedge", fatedge)
    # # cv2.waitKey(0)
    # n, comp = cv2.connectedComponents((fatedge == 0).astype(np.uint8))
    # # print(comp)
    # filled = (comp != comp[0, 0]).astype(np.uint8)
    # cv2.imshow("filled", filled)
    # output = cv2.erode(filled, ker) * 255
    # cv2.imshow("output", output)
    # cv2.waitKey(0)

    # gray = cv2.cvtColor(masked_gray,cv2.COLOR_RGB2GRAY)
    #img_ = cv2.threshold(gray,100,225,cv2.THRESH_BINARY)
    # edges = cv2.Canny(masked_gray, 20, 250, True)
    # cv2.imshow("edges", edges)

    # # Binarize edges
    # binedge=(edges>0).astype(np.uint8)
    # # Removing edges too close from left and right borders
    # binedge[:,:20]=0
    # binedge[:,-20:]=0
    # # Fatten them so that there is no hole
    # ker=np.ones((3,3))
    # fatedge=cv2.dilate(binedge, ker)
    # cv2.imshow("fatedge", fatedge)
    # # Find connected black areas
    # n,comp=cv2.connectedComponents((fatedge==0).astype(np.uint8))
    # # comp is an image whose each value is the index of the connected component
    # # Assuming that point (0,0) is in the border, not inside the character, border is where comp is == comp[0,0]
    # # So character is where it is not
    # # Or, variant from new image: considering "outside" any part that touches one of the left, right, or top border
    # # Note: that is redundant with previous 0ing of left and right borders
    # # Set of all components touching left, right or top border
    # listOutside=set(comp[:,0]).union(comp[:,-1]).union(comp[0,:])
    # if 0 in listOutside: listOutside.remove(0) # 0 are the lines, that is what is False in fatedge==0
    # filled=(~np.isin(comp, list(listOutside))).astype(np.uint8) # isin need array or list, not set

    # # Just to be extra accurate, since we had dilated edges, with can now erode result
    # output=cv2.erode(filled, ker)
    # cv2.imshow("output", output)
    # cv2.waitKey(0)

    # exit()

    # Threshold the masked grayscale image
    # try:
        # cv2.imshow("masked_gray: ", masked_gray)
        # cv2.waitKey(0)
        # _, thresholded = cv2.threshold(blur, 0, filler, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # masked_gray = cv2.convertScaleAbs(filteredImageBitwiseMaxMin)
        # cv2.imshow("filteredImageBitwiseMaxMin: ", filteredImageBitwiseMaxMin)
        # thresholded = cv2.adaptiveThreshold(filteredImageBitwiseMaxMin, filler,
        #                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                     cv2.THRESH_BINARY, 73, 2)
        # # cv2.imshow("thresholded: ", thresholded)
        # # cv2.waitKey(0)
        # # Perform morphological operations to remove noise
        # kernel = np.ones((3, 3), np.uint8)
        # # cv2.imshow("kernel: ", kernel)
        # # cv2.waitKey(0)
        # erosion = cv2.erode(thresholded, kernel, iterations=1)
        # cv2.imshow("erosion: ", erosion)

        # # cv2.imshow("erosion: ", erosion)
        # # cv2.waitKey(0)
        # # Copy the thresholded text to the empty image
        # # cropped_im[mask == filler] = blur[mask == filler]
        # cropped_im[filteredImageBitwiseMaxMin == filler] = erosion[filteredImageBitwiseMaxMin == filler]
    # except Exception as e:
    #     print("----small badaboom ----")
    #     print(e)
    #     print(im_name)
    #     raise Exception(e)
        # try:
        #     masked_gray = cv2.convertScaleAbs(masked_gray)
        #     thresholded = cv2.adaptiveThreshold(masked_gray, filler,
        #                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                         cv2.THRESH_BINARY, 73, 2)
        #     # _, thresholded = cv2.threshold(masked_gray, 0, filler, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #     # if thresholded.any():
        #     # Perform morphological operations to remove noise
        #     kernel = np.ones((3, 3), np.uint8)
        #     erosion = cv2.erode(thresholded, kernel, iterations=1)
        #     # Copy the thresholded text to the empty image
        #     cropped_im[masked_gray == filler] = erosion[masked_gray == filler]
        # except Exception as e2:
        #     print("----big badaboom ----")
        #     print(e2)
    mask_canvas_result = np.zeros(image.shape[:2], dtype="uint8")
    # Normalize https://stackoverflow.com/questions/46260601/convert-image-from-cv-64f-to-cv-8u
    # Define the polygon vertices
    # points_np = np.array(points, np.int32)
    # Create a mask image
    mask_canvas_result = np.zeros(filteredImageBitwiseMaxMin.shape, np.uint8)
    filler = 1
    cv2.fillPoly(mask_canvas_result, [points_np], filler)
    cropped_im = cv2.bitwise_and(filteredImageBitwiseMaxMin, filteredImageBitwiseMaxMin, mask=mask_canvas_result)
    cropped_im[cropped_im < 60] = 0
    cropped_im[cropped_im >= 60] = filler
    # cv2.imshow("cropped_im: ", cropped_im)
    # cv2.waitKey(0)
    return cropped_im


# file_name = "../fonts/SynthText_train.h5"

# db = h5py.File(file_name, 'r')
# im_names = list(db['data'].keys())
# # im = im_names[0]
# # im = "maze_25.png_0"
# # img = cv2.imread("../fonts/images/maze_25.jpg")
# # im = "city_99.jpg_0"
# # im = "jaipur_58.jpg_0"
# im = "ant+hill_84.jpg_0"
# # im = "delhi_55.jpg_0"
# # img = cv2.imread("../fonts/images/ant+hill_84.jpg")

# # img = cv2.imread("../fonts/images/"+im)
# # img = cv2.imread("image_test_font.jpg")
# # print(im)
# img = db['data'][im][:]
# font = db['data'][im].attrs['font']
# charBB = db['data'][im].attrs['charBB']

# xId = 0
# yId = 1
# boxId = 1

# points = [[
#     int(db['data'][im].attrs['charBB'][xId][0][boxId]),
#     int(db['data'][im].attrs['charBB'][yId][0][boxId])
# ],
#           [
#               int(db['data'][im].attrs['charBB'][xId][1][boxId]),
#               int(db['data'][im].attrs['charBB'][yId][1][boxId])
#           ],
#           [
#               int(db['data'][im].attrs['charBB'][xId][2][boxId]),
#               int(db['data'][im].attrs['charBB'][yId][2][boxId])
#           ],
#           [
#               int(db['data'][im].attrs['charBB'][xId][3][boxId]),
#               int(db['data'][im].attrs['charBB'][yId][3][boxId])
#           ]]

# maskk = make_mask(img, points, im)
