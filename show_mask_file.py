import cv2
import numpy as np

# Read the mask image from file
# file_path = "../LabPics Medical/Train/386Train_Blood/Vessels/3.png"
file_path = "../LabPics Medical/Train/539Train_Blood/SemanticMaps/PerVessel/1/Foam.png"

mask = cv2.imread(file_path, 0)

mask[mask == 1] = 255

# Show the mask
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()