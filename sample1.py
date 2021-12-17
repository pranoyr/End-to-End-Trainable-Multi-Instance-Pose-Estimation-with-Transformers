import torch



# num_people, 17, 2 --> num people , 17 *2
gt = torch.Tensor(20, 17, 2)  # batch [15,5]  , 2 images , batch size = 2


print(gt)

# num_people, 17, 2 --> num people , 17 *2  
keypoints = torch.Tensor(24, 17, 2)  # batch [12,12]  , 2 images , batch size = 2


gt = gt.view(-1, 17*2)

print("**")
print(gt)

keypoints = keypoints.view(-1, 17*2)





print(keypoints.shape)
print(gt.shape)


C = torch.cdist(keypoints, gt, p=2)

sizes = [15,5]



print(C.shape)

# C = C.view(2, 2, -1)

indices = [c[i] for i, c in enumerate(C.split(sizes, -1))]

print(indices)




# import torch
# keypoints = torch.tensor([[[2,2,1],       # batch size --> 2
#                           [1,4,2],   
#                           [3,2,1],
#                           [1,3,0]],

#                           [[1,3,0],
#                           [3, 4, 5],
#                           [24,5,1],
#                           [3,1,2]]], dtype=torch.float32)


# gt = torch.tensor(
#                           [[10,2,1],
#                           [10,11,2],
#                           [4,2,1],
#                           [2,2,1],

#                           [3,5,23],
#                           [4,12,1],
#                           [2,2,1]], dtype=torch.float32)                          

# print(keypoints.shape)
# print(gt.shape)

# sizes = [4,3]


# C  = torch.cdist(keypoints, gt, p=2)

# print(C)

# # C = C.view(2, 4, -1)

# indices = [c[i] for i, c in enumerate(C.split(sizes, -1))]

# print(indices)

# # [tensor([[ 8.0000, 12.0830,  2.0000,  0.0000],
# #         [ 9.2736, 11.4018,  3.7417,  2.4495],
# #         [ 7.0000, 11.4455,  1.0000,  1.0000],
# #         [ 9.1104, 12.2066,  3.3166,  1.7321]]), tensor([[23.1733,  9.5394,  1.7321],
# #         [18.0278,  9.0000,  4.5826],
# #         [30.4138, 21.1896, 22.2036],
# #         [21.3776, 11.0905,  1.7321]])]
