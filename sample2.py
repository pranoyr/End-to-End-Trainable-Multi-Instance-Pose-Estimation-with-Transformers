import torch



z = torch.Tensor(2, 34) # 20, 17, 2 --> 20, 17 *2  
v = torch.tensor([[1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0],
                  [1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0]])

print(torch.repeat_interleave(v , 2, dim=1).shape)
