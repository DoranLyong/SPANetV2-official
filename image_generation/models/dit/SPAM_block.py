import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers.helpers import to_2tuple

def resize_weight(origin_weight, new_h, new_w):
    """ original_weight shape: (dim, h, w)
        new_weight shape: (dim, new_h, new_w)
    """
    num_heads, h, w  = origin_weight.shape
    origin_weight = origin_weight.reshape(1, num_heads, h, w)
    new_weight = F.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).reshape(num_heads, new_h, new_w).contiguous()
    return new_weight


class ModifiedSigmoid(nn.Module):
    def __init__(self, alpha=1.0):
        super(ModifiedSigmoid, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))  # alpha 매개변수

    def forward(self, x):
        return torch.sigmoid(self.alpha * x) 


class ExSPAM(nn.Module):
    """ Expanded SPAM
    """
    def __init__(self, dim, num_heads=4, expand_ratio=2, act_layer=nn.GELU, bias=False,
                 proj_drop=0., proj_bias=False, size=14, **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0] # H
        self.filter_size = size[1] // 2 + 1 # W//2 + 1

        self.dim = dim
        self.num_heads = num_heads

        self.scale = ModifiedSigmoid(alpha=1.0)  

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.query = nn.Sequential(nn.Linear(dim, dim, bias=bias), act_layer())
        self.ctx = nn.Linear(dim, dim, bias=bias)

        self.split_groups=self.dim// num_heads
        self.split_indexes = [self.split_groups for i in range(num_heads)]

        self.local_convs = nn.ModuleList()
        for i in range(self.num_heads):
            local_conv = nn.Conv2d(dim//self.num_heads, dim//self.num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.num_heads)                                 
            self.local_convs.append(local_conv)

        self.filter = nn.Parameter(torch.randn(dim, self.size, self.filter_size, dtype=torch.float32) * 0.02)  # (dim, h, w)
        
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups, bias=bias),
            nn.GroupNorm(1, dim*expand_ratio), # layer norm for (B,C,H,W)
            act_layer(),
            nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1, bias=bias))

        # -- Update -- #
        self.proj_out = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        q = self.query(x) # (B, H, W, C)
        heads = torch.split(self.ctx(x).permute(0,-1,1,2), self.split_indexes, dim=1)
        conv_feat = torch.cat([self.local_convs[i](heads[i]) 
                            for i in range(self.num_heads)], dim=1)
        
        if (H, W//2+1) != (self.size, self.filter_size):
            filter = resize_weight(self.filter,  H, W//2+1)     
        else:
            filter = self.filter

        SRF = torch.sigmoid(filter)  #-- the submission version (Mar, 2025)      
        #SRF = self.scale(filter)      #-- the revision version (Nov, 2025)
        s_i = torch.fft.rfft2(conv_feat.to(torch.float32), dim=(2, 3), norm='ortho')
        s_i.mul_(SRF) # inplace for s_i = s_i * SRF
        s_i = torch.fft.irfft2(s_i, s=(H, W), dim=(2, 3), norm='ortho').to(x.dtype)
        s_ctx = self.proj(s_i).permute(0, 2, 3, 1) # (B, H, W, C)

        x = q * s_ctx # modulation

        # == Update == #
        x = self.proj_out(x)
        x = self.proj_drop(x)
        return x