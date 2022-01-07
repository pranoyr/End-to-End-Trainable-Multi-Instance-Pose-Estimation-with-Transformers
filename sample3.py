import torch



A_gt = torch.Tensor(2, 34)
A_pred = torch.Tensor(100, 34)

V_gt =  torch.Tensor(2, 17)
Vgt_ = torch.repeat_interleave(V_gt , 2, dim=1) # torch.Size([2, 34])

# abs_loss = [torch.cdist(A_pred * v_gt_single.unsqueeze(0), a_gt_single.unsqueeze(0) * v_gt_single.unsqueeze(0), p=1) for v_gt_single, a_gt_single in zip(Vgt_, A_gt)] 
# print(abs_loss)
# print(torch.cat(abs_loss, dim=1).shape)
# abs_loss = torch.cdist(A_pred, A_gt, p=1)
# print(abs_loss.shape)

# vgt_ = torch.Tensor()



# a = torch.tensor([[1,2,3],
#              [4,5,6],
#              [7,8,9]])

# b = torch.tensor([[2,2,2]])

# print(a*b)
