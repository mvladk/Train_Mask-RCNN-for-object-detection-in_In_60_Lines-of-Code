# importing cv2 & numpy:
import numpy as np
import cv2
# Set the OCR libraries:
from PIL import Image
import pyocr
import pyocr.builders

# Set image path
path = "./"
fileName = "jaipur_58.jpg"

# Read input image:
inputImage = cv2.imread(path + fileName)
inputCopy = inputImage.copy()

# Convert BGR to grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Set the adaptive thresholding (gasussian) parameters:
windowSize = 31
windowConstant = -1
# Apply the threshold:
binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, windowSize,
                                    windowConstant)

# Perform an area filter on the binary blobs:
componentsNumber, labeledImage, componentStats, componentCentroids = \
cv2.connectedComponentsWithStats(binaryImage, connectivity=4)

# Set the minimum pixels for the area filter:
minArea = 20

# Get the indices/labels of the remaining components based on the area stat
# (skip the background component at index 0)
remainingComponentLabels = [
    i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea
]

# Filter the labeled pixels based on the remaining labels,
# assign pixel intensity to 255 (uint8) for the remaining pixels
filteredImage = np.where(
    np.isin(labeledImage, remainingComponentLabels) == True, 255,
    0).astype('uint8')

# Set kernel (structuring element) size:
kernelSize = 3

# Set operation iterations:
opIterations = 1

# Get the structuring element:
maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

# Perform closing:
closingImage = cv2.morphologyEx(filteredImage, cv2.MORPH_CLOSE, maxKernel,
                                None, None, opIterations,
                                cv2.BORDER_REFLECT101)

# Get each bounding box
# Find the big contours/blobs on the filtered image:
contours, hierarchy = cv2.findContours(closingImage, cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_SIMPLE)

contours_poly = [None] * len(contours)
# The Bounding Rectangles will be stored here:
boundRect = []

# Alright, just look for the outer bounding boxes:
for i, c in enumerate(contours):

    if hierarchy[0][i][3] == -1:
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect.append(cv2.boundingRect(contours_poly[i]))

# Draw the bounding boxes on the (copied) input image:
for i in range(len(boundRect)):
    color = (0, 255, 0)
    cv2.rectangle(inputCopy, (int(boundRect[i][0]), int(boundRect[i][1])), \
              (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)

    # Crop the characters:
for i in range(len(boundRect)):
    # Get the roi for each bounding rectangle:
    x, y, w, h = boundRect[i]

    # Crop the roi:
    croppedImg = closingImage[y:y + h, x:x + w]
    cv2.imshow("Cropped Character: " + str(i), croppedImg)
    cv2.waitKey(0)

# cv2.imshow("Cropped Character: " , croppedImg)
# cv2.waitKey(0)

# Set pyocr tools:
# tools = pyocr.get_available_tools()
# # The tools are returned in the recommended order of usage
# tool = tools[0]

# # Set OCR language:
# langs = tool.get_available_languages()
# lang = langs[0]

# # Get string from image:
# txt = tool.image_to_string(Image.open(path + "closingImage.png"),
#                            lang=lang,
#                            builder=pyocr.builders.TextBuilder())

# print("Text is:" + txt)
