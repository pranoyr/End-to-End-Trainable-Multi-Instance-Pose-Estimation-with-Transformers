# import torch



# z = torch.Tensor(2, 34) # 20, 17, 2 --> 20, 17 *2  
# v = torch.tensor([[1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0],
#                   [1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0]])

# print(torch.repeat_interleave(v , 2, dim=1).shape)


# import torch
# import torch.nn as nn

# loss = nn.BCELoss(reduction='mean')
# input = torch.ones(3, 5)
# print(input)
# target = torch.ones(3, 5)
# print(target)
# output = loss(input, target)
# print(output)
# print(output.shape)


# a = torch.tensor([1,2]*17)
# print(a)

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab

prefix = 'person_keypoints'

#initialize COCO ground truth api
dataDir='../'
dataType='train2017'
annFile = "/home/pranoy/code/detr/data/annotations/person_keypoints_train2017.json"
cocoGt=COCO(annFile)

imgIds=sorted(cocoGt.getImgIds())
print(len(imgIds))

# ann_ids = cocoGt.getAnnIds(imgIds=5)
# print(ann_ids)
# target = cocoGt.loadAnns(0)



from PIL import Image
c = 0
for i in imgIds:
    if cocoGt.getAnnIds(imgIds=i) == []:
        continue
    else:
        img = Image.open("/home/pranoy/code/detr/data/train2017/"+ cocoGt.loadImgs(i)[0]['file_name'])
        ann_ids = cocoGt.getAnnIds(imgIds=i)
        target = cocoGt.loadAnns(ann_ids)
        classes = [[obj["category_id"], obj["num_keypoints"]] for obj in target]
        print(classes)
        # print(cocoGt.loadImgs(i))
        # print(target)
        
    c +=1

print(c)
