import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os
import torch
import torch_directml
import h5py
# from sklearn import preprocessing

from mask_maker2 import make_mask
# dml = torch_directml.device()

batchSize=2
imageSize=[600,600]
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
device = torch_directml.device()
print("----------------------------------")
# print(torch.backends.context.backend)
# print(torch.backends.backend)
print(device)
# print(torch.device('cuda'));
# print(torch.device('opencl'));
# print(torch.cuda.is_available());
# print(torch.opencl.is_available());
print("-----------------======================")
# exit()

# trainDir="./LabPics Medical/Train"
trainDir=".././fonts/images"
checkpointDir="./checkpoints/"
lastStateFilePath = checkpointDir+"last.torch"
lastModelState = checkpointDir+"last_model.torch"
epoch = 1
# LOSS = 0.4
loadLast = False

h5_file_name = "../fonts/SynthText_train.h5"
db = h5py.File(h5_file_name, 'r')
im_names = list(db['data'].keys())
im_names = im_names[:600]

maped_font = {}
maped_font[b'Alex Brush'] = 0
maped_font[b'Sansation'] = 1
maped_font[b'Titillium Web'] = 2
maped_font[b'Open Sans'] = 3
maped_font[b'Ubuntu Mono'] = 4

# imgs=[]
# for pth in os.listdir(trainDir):
#     imgs .append(trainDir+"/"+pth +"//")
    
def loadData():

    # im = im_names[0]
    # img = db['data'][im][:]
    # font = db['data'][im].attrs['font']
    # charBB = db['data'][im].attrs['charBB']
    xId = 0
    yId = 1
    # boxId = 1
    batch_Imgs=[]
    batch_Data=[]# load images and masks

    # fonts_idxs_by_name = {}

    for i in range(batchSize):
        idx=random.randint(0,len(im_names)-1)
        # img = cv2.imread(os.path.join(imgs[idx], "Image.jpg"))
        im = im_names[idx]
        img = db['data'][im][:]
        fonts = db['data'][im].attrs['font']
        # print(fonts)
        # exit()
        # for f,k in range(fonts):
        #     fonts_idxs_by_name[f] = k
        fonts_num = len(db['data'][im].attrs['font'])
        # Create a one-hot encoding of the byte strings
        # vocab = np.unique(fonts)
        # vocab_to_int = {word: i for i, word in enumerate(vocab)}
        # int_to_vocab = {i: word for i, word in enumerate(vocab)}
        # encoded_fonts = np.zeros((fonts.shape[0],), dtype=np.float)
        # for i, word in enumerate(fonts):
        #     encoded_fonts[i] = vocab_to_int[word]


        charBB = db['data'][im].attrs['charBB']
        # img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)
        # maskDir=os.path.join(imgs[idx], "Vessels")
        masks=[]
        for boxId in range(fonts_num):
            # vesMask = (cv2.imread(maskDir+'/'+mskName, 0) > 0).astype(np.uint8)  # Read vesse instance mask
            points = [
                [int(charBB[xId][0][boxId]), int(charBB[yId][0][boxId])],
                    [int(charBB[xId][1][boxId]), int(charBB[yId][1][boxId])], 
                [int(charBB[xId][2][boxId]), int(charBB[yId][2][boxId])], 
                    [int(charBB[xId][3][boxId]), int(charBB[yId][3][boxId] )]
                ]
            vesMask = make_mask(img, points).astype(np.uint8)
            img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)
            vesMask=cv2.resize(vesMask,imageSize,cv2.INTER_NEAREST)
            # print(im)
            # np.set_printoptions(threshold=np.inf, linewidth=140)
            # print(vesMask)
            # exit()
            masks.append(vesMask)# get bounding box coordinates for each mask

            # np.set_printoptions(threshold=np.inf, linewidth=140)
            # # mask = np.zeros(vesMask.shape[:2], dtype="uint8")
            # masked = cv2.bitwise_and(img, img, mask=vesMask)
            # cv2.imshow("Rectangular vesMask", vesMask)
            # cv2.imshow("Rectangular masked", masked)
            # cv2.imshow("img ", img)
            # print(vesMask)
            # cv2.waitKey(0)
            # exit()

        num_objs = len(masks)
        masks_filtered = []
        fonts_filtered = []
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            if h == 0 or w == 0:
                continue
            masks_filtered.append(masks[i])
            fonts_filtered.append(fonts[i])
            
        # print(f"masks {len(masks)}")
        # print(f"masks_filtered {len(masks_filtered)}")
        # print(f"fonts_filtered {len(fonts_filtered)}")
        # print(f"fonts {len(fonts)}")
        # print(f"fonts {fonts}")
        # print(f"fonts_filtered {fonts_filtered}")


        

        num_objs_filtered = len(masks_filtered)
        if num_objs_filtered==0: return loadData() # if image have no objects just load another image
        boxes = torch.zeros([num_objs_filtered,4], dtype=torch.float32)
        # masks_filtered = []
        for i in range(num_objs_filtered):
            x,y,w,h = cv2.boundingRect(masks_filtered[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
            # if h == 0 or w == 0:
            #     print("i: " + str(i))
            #     print("im: " + str(im))
            #     cv2.imshow("img ", img)
            #     cv2.waitKey(0)
            #     # print(masks[i])
            #     print(x,y,w,h)
        # print(boxes)
        # exit()
        # numpy.array( LIST )
        masks_tensor = torch.as_tensor(masks_filtered, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        # num_objs_lll = len(boxes)
        # data["labels"] =  torch.ones((num_objs_lll,), dtype=torch.int64)   # there is only one class
        
        maped_labels = [maped_font[f] for f in fonts_filtered ]

        # print(f"maped_labels {len(maped_labels)}")
        # print(f"boxes {len(boxes)}")
        # print(f"maped_labels {maped_labels}")
        # exit()

        data["labels"] = torch.as_tensor(maped_labels, dtype=torch.int64)

        # le = preprocessing.LabelEncoder()
        # lbls = le.fit_transform(fonts)
        # lbls = torch.frombuffer(fonts, dtype=torch.int32)
        # lbls = torch.as_tensor(fonts)
        # lbls = torch.from_numpy(encoded_fonts)
        # lbls = torch.as_tensor(encoded_fonts, dtype=torch.int64)
        # data["labels"] = lbls
        data["masks"] = masks_tensor

        # print("--------------")
        # print(im)
        # print("boxes count: " + str(len(boxes)))
        # print("label count: " + str(len(data["labels"])))
        # print("masks count: " + str(len(masks)))
        # print("==============")


        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=5)  # replace the pre-trained head with a new one
model.to(device)# move model to the right devic

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)

print("-----------Continue previous?----------------")
print("Should load? " + str(loadLast))
if(loadLast):
    # print("loading lastModelState: " + lastModelState)
    # checkpoint = torch.load(lastModelState)
    # model.load_state_dict(checkpoint)
    # epoch = 9001

    print("loading lastStateFilePath: " + lastStateFilePath)
    checkpoint = torch.load(lastStateFilePath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = 1 + int(checkpoint['epoch'])
    lossLast = checkpoint['loss']
    print('load epoch: ' + str(epoch))
print("-----------TRAIN----------------")

model.train()

saveEveryIterations = 500
maxIterations = 10001
for i in range(epoch, maxIterations):
            images, targets = loadData()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            print(i,'loss:', losses.item())
            if i%saveEveryIterations==0:
                torch.save(model.state_dict(), checkpointDir+str(i)+".torch")
                # torch.save(model.state_dict(), lastModelState)
                print("save epoch: "+ str(i))
                torch.save({
                    'epoch': str(i),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses.item(),
                    }, lastStateFilePath)


