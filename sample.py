import numpy as np
import torch 

keypoints = torch.tensor([[[2,2,1],
                          [1,4,2],     # 2 person, 4 keypoints, 3 coordinates
                          [3,2,1],
                          [1,3,0]],

                          [[10,2,1],
                          [10,11,2],
                          [4,2,1],
                          [10,11,0]]], dtype=torch.float32)


cxcy = keypoints.mean(dim=1)[:,:2] # center of the keypoints   torch.Size([2, 2])
cxcy_expand = cxcy.clone()

cxcy_expand = torch.repeat_interleave(cxcy_expand.unsqueeze(1) , 4, dim=1)

offsets = keypoints[:,:,:2] - cxcy_expand

C = cxcy
Z = offsets
V = keypoints[:,:,2]

# print(V.unsqueeze(1).repeat_interleave(2, dim=1).shape)

print("C: ", C.shape) #  num_people, 2
print("Z: ", Z.shape)  # num_people, 17, 2
print("V: ", V.shape) # num_people, 17

print(V.unsqueeze(2).shape)
a = Z * V.unsqueeze(2)
print(a.shape)

# print(cxxy)

# print(offsets)

