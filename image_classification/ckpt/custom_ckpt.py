"""
Convert .pth.ar to .pth

"""

import torch 

ckpt = torch.load('./ckpt/model_best.pth.tar')


print(ckpt.keys())

# save model
torch.save(ckpt['state_dict'], f'./ckpt/spanetv2_s36_pure_res-scale_Full-ExSPAM.pth')
