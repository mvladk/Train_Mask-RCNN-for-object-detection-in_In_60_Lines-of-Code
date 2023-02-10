import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torch_directml

imageSize=[600,600]
# imgPath="../fonts/images/baroque_8.jpg"
# imgPath="../fonts/images/maze_25.png"
imgPath="./image_test_font.jpg"
# imgPath="./clothes_137_56.jpg"
# imgPath="./city_99.jpg"
# imgPath="./aquarium_7_9.jpg"



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
# device = torch_directml.device()
print(device)
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
# weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1
# weights='MaskRCNN_ResNet50_FPN_Weights.DEFAULT'
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1')  # load an instance segmentation model pre-trained pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.COCO_V1')  # load an instance segmentation model pre-trained pre-trained on COCO
# model = torchvision.models.detection.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=5)  # replace the pre-trained head with a new one

checkpointDir="./checkpoints/"
# lastStateFilePath = checkpointDir+"last_model.torch"
lastStateFilePath = checkpointDir+"last.torch"
lastFilePath = checkpointDir+"4000.torch"

print("loading lastStateFilePath: " + lastStateFilePath)
checkpoint = torch.load(lastStateFilePath, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
# model.load_state_dict(torch.load(lastFilePath, map_location=device), strict=False)
# model.load_state_dict(checkpoint)
# model.load_state_dict(checkpoint['model_state_dict'], map_location=device)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = 1 + int(checkpoint['epoch'])
# lossLast = checkpoint['loss']
# print('load epoch: ' + str(epoch))
# model.load_state_dict(torch.load("10000.torch"))
# model.load_state_dict(torch.load("10000.torch"))


model.to(device)# move model to the right devic
model.eval()

images=cv2.imread(imgPath)
images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
images=images.swapaxes(1, 3).swapaxes(2, 3)
images = list(image.to(device) for image in images)

with torch.no_grad():
    pred = model(images)

# print(pred)

im= images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
im2 = im.copy()
accuracy_score = 0.17
for i in range(len(pred[0]['masks'])):
    msk=pred[0]['masks'][i,0].detach().cpu().numpy()
    box=pred[0]['boxes'][i,0].detach().cpu().numpy()
    scr=pred[0]['scores'][i].detach().cpu().numpy()
    
    lbls=pred[0]['labels'][i].detach().cpu().numpy()
    color = (lbls + 1) * 50
    # print(f"lbls: {lbls}")
    # print(f"color: {color}")
    # color = 255
    if scr > accuracy_score :
        im2[:,:,0][msk>0.5] = color # random.randint(0,255)
        im2[:, :, 1][msk > 0.5] = color # random.randint(0,255)
        im2[:, :, 2][msk > 0.5] = color # random.randint(0,255)
        # im2[:,:,0][box>0.5] = random.randint(0,255)
        # im2[:, :, 1][box > 0.5] = random.randint(0,255)
        # im2[:, :, 2][box > 0.5] = random.randint(0, 255)
        
# print(f"pred[0]: {pred[0]}")
cv2.imshow(str(scr), np.hstack([im,im2]))
cv2.waitKey()
