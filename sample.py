import numpy as np
import torch 

w , h = (100, 100)

keypoints = torch.tensor([[[2,2,1],
						  [1,4,2],     # 2 person, 4 keypoints, 3 coordinates
						  [3,2,1],
						  [1,3,0]],

						  [[10,2,1],
						  [10,11,2],
						  [4,2,1],
						  [10,11,0]]], dtype=torch.float32)


v = torch.tensor([[1,0,1,0],
				[0,1,0,1]], dtype=torch.float32)


keypoints = keypoints[:,:,:2]

cxcy = (keypoints * v.unsqueeze(2)).sum(dim=1) / v.unsqueeze(2).repeat_interleave(2, dim=2).sum(dim=1)
print(cxcy)




cxcy = keypoints.mean(dim=1)# center of the keypoints   torch.Size([2, 2])
print(cxcy)
cxcy_expand = cxcy.clone()

cxcy_expand = torch.repeat_interleave(cxcy_expand.unsqueeze(1) , 4, dim=1)

offsets = keypoints - cxcy_expand

C = cxcy
Z = offsets.view(-1, 2*4)
V = v




# print(V.unsqueeze(1).repeat_interleave(2, dim=1).shape)

print("C: ", C.shape) #  num_people, 2
print("Z: ", Z.shape)  # num_people, 17, 2
print("V: ", V.shape) # num_people, 17

C = C / torch.tensor([w, h], dtype=torch.float32)
Z = Z / torch.tensor([w, h] * 4, dtype = torch.float32)

all_keypoints = torch.cat([C, Z, V], dim=1)
all_keypoints = torch.cat([all_keypoints, all_keypoints], dim=0)
print(all_keypoints.shape)


C = all_keypoints[:, :2]
Z = all_keypoints[:, 2:10]
V = all_keypoints[:, 10:]


C_gt_expand = torch.repeat_interleave(C.unsqueeze(1), 4, dim=1).view(-1,8)

A_gt = C_gt_expand + Z
A_gt = A_gt * torch.tensor([w, h] * 4, dtype = torch.float32)

print(A_gt.view(-1,4,2))







# print(V.unsqueeze(2).shape)
# a = Z * V.unsqueeze(2)
# print(a.shape)

# print(cxxy)

# print(offsets)

# v = torch.tensor([[1,0,0,2,1,1,0,1],
# 				[1,0,0,2,1,1,0,1]])

# print(torch.tensor([2,1]*4))

# centre = torch.tensor([[2,3], [1,5]])
# print(centre)

# centre = torch.repeat_interleave(centre.unsqueeze(1), 17, dim=1).view(-1,34)
# print(centre)

# a = torch.ones(1,34)
# b = torch.zeros(1,34)
# c = torch.cat([a,b], dim=0)
# print(centre+ c)