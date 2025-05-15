import copy

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple

from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner.checkpoint import _load_checkpoint
from mmengine.logging import MMLogger

from mmdet.registry import MODELS




class Downsampling(BaseModule):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class Scale(BaseModule):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale
        

class SquaredReLU(BaseModule):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(BaseModule):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class Attention(BaseModule):
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # -- Update
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)

        # -- Update -- #
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixAttention(BaseModule):
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # -- Local Conv
        k_size = 7 
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=k_size, padding=k_size//2, stride=1, groups=dim, bias=qkv_bias)        
        #self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=qkv_bias)

        # -- Update
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim) + \
            self.local_conv(v.transpose(1, 2).reshape(B, H, W, C).permute(0,-1, 1,2)).permute(0,2,3,1)

        # -- Update -- #
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



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


class ExSPAM(BaseModule):
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

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.query = nn.Sequential(nn.Linear(dim, dim, bias=bias), act_layer())
        self.ctx = nn.Linear(dim, dim, bias=bias)

        self.split_groups=self.dim// num_heads

        for i in range(self.num_heads):
            local_conv = nn.Conv2d(dim//self.num_heads, dim//self.num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.num_heads)
            setattr(self, f"local_conv_{i + 1}", local_conv)
            filter = nn.Parameter(torch.randn(dim//self.num_heads, self.size, self.filter_size, dtype=torch.float32) * 0.02)  # (dim, h, w)
            setattr(self, f"filter_{i + 1}", filter)

        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups, bias=bias),
            nn.GroupNorm(1, dim*expand_ratio),
            act_layer(),
            nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1, bias=bias))

        # -- Update -- #
        self.proj_out = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        q = self.query(x) # (B, H, W, C)
        ctx = self.ctx(x).reshape(B, H, W, self.num_heads, C//self.num_heads).permute(3, 0, 4, 1, 2) # (num_heads, B,  head_size, H, W)

        # -- Initialize s_ctx for accumulation
        s_ctx = torch.zeros((B, self.split_groups, self.num_heads, H, W), device=x.device, dtype=x.dtype)

        for i in range(self.num_heads):
            Conv = getattr(self, f"local_conv_{i + 1}")
            filter = getattr(self, f"filter_{i + 1}")

            if (H, W//2+1) != (self.size, self.filter_size):
                filter = resize_weight(filter,  H, W//2+1)                
            SPF = torch.sigmoid(filter)

            s_i = torch.fft.rfft2(Conv(ctx[i]).to(torch.float32), dim=(2, 3), norm='ortho')
            s_i = s_i * SPF
            s_i = torch.fft.irfft2(s_i, s=(H, W), dim=(2, 3), norm='ortho').reshape(B, self.split_groups, H, W).to(x.dtype)

            # -- Accumulate each head in s_ctx
            s_ctx[:, :, i, :, :] = s_i  # <- alternative to concat

        s_ctx = s_ctx.contiguous().view(B, C, H, W)
        s_ctx = self.proj(s_ctx).permute(0, 2, 3, 1) # (B, H, W, C)

        x = q * s_ctx # modulation

        # == Update == #
        x = self.proj_out(x)
        x = self.proj_drop(x)
        return x



class LayerNormGeneral(BaseModule):
    r""" General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default. 
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance. 
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(BaseModule):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster, 
    because it directly utilizes otpimized F.layer_norm
    """
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class Mlp(BaseModule):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(BaseModule):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
        norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(BaseModule):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None,
                 size=14,
                 ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop, size=size)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


r"""
downsampling (stem) for the first stage is a layer of conv with k7, s4 and p2
downsamplings for the last 3 stages is a layer of conv with k3, s2 and p1
DOWNSAMPLE_LAYERS_FOUR_STAGES format: [Downsampling, Downsampling, Downsampling, Downsampling]
use `partial` to specify some arguments
"""
DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling,
            kernel_size=7, stride=4, padding=2,
            post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6)
            )] + \
            [partial(Downsampling,
                kernel_size=3, stride=2, padding=1, 
                pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True
            )]*3



class SPANetV2(BaseModule):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [2, 2, 6, 2].
        dims (int): Feature dimension at each stage. Default: [64, 128, 320, 512].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage. Default: nn.Identity.
        mlps (list, tuple or mlp_fcn): Mlp for each stage. Default: Mlp.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage. Default: partial(LayerNormGeneral, eps=1e-6, bias=False).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_dropout (float): dropout for MLP classifier. Default: 0.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: None.
            None means not use the layer scale. Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: [None, None, 1.0, 1.0].
            None means not use the layer scale. From: https://arxiv.org/abs/2110.09456.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=nn.Identity,
                 mlps=Mlp,
                 norm_layers=partial(LayerNormWithoutBias, eps=1e-6), # partial(LayerNormGeneral, eps=1e-6, bias=False),
                 drop_path_rate=0.,
                 head_dropout=0.0, 
                 layer_scale_init_values=[None, None, None, None],
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 output_norm=partial(nn.LayerNorm, eps=1e-6), 
                 head_fn=nn.Linear,
                 input_size=(3, 224, 224),
                 init_cfg=None, 
                 pretrained=None, 
                 **kwargs,
                 ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)]
        )
        
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = ModuleList() # each stage consists of multiple metaformer blocks

        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[MetaFormerBlock(dim=dims[i],
                token_mixer=token_mixers[i],
                mlp=mlps[i],
                norm_layer=norm_layers[i],
                drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_values[i],
                res_scale_init_value=res_scale_init_values[i],
                size=(input_size[1] // (2 ** (i + 2)),
                      input_size[2] // (2 ** (i + 2))),
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

            # -- for dense prediction features 
            layer = norm_layers[i](dims[i])
            setattr(self, f"norm{i}", layer)
                    

        # -- Classifier head 
        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        
        
        self.init_cfg = copy.deepcopy(init_cfg)        
        self.apply(self._init_weights)

        # --load pre-trained model 
        if (self.init_cfg is not None or pretrained is not None):
            self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading 
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            
            print("****** Pretrained weights are loaded for SPANetV2 ******")
            # -- show for debug
            #print('missing_keys: ', missing_keys)
            #print('unexpected_keys: ', unexpected_keys)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}


    def forward_features(self, x):
        outs = []
        for i in range(self.num_stage):            
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            norm_layer = getattr(self, f'norm{i}')
            out = norm_layer(x).permute(0, 3, 1, 2).contiguous() # (B,H,W,C) -> (B,C,H,W)
            outs.append(out)

        # -- output the features of four stages for dense prediction
        return outs 

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)  # classification head 
        return x


""" The following models are for dense prediction based on 
    mmdetection and mmsegmentation.
"""
@ MODELS.register_module()
class spanetv2_s18_pure(SPANetV2):
    def __init__(self, **kwargs):
        super().__init__(
            depths=[3, 3, 9, 3],
            dims=[64, 128, 320, 512],
            token_mixers=ExSPAM,
            head_fn=MlpHead,
            input_size=(3, 224, 224),
            **kwargs
        )

@ MODELS.register_module()
class spanetv2_s18_hybrid(SPANetV2):
    def __init__(self, **kwargs):
        super().__init__(
            depths=[3, 3, 9, 3],
            dims=[64, 128, 320, 512],
            token_mixers=[ExSPAM, ExSPAM, MixAttention, Attention],
            head_fn=MlpHead,
            input_size=(3, 224, 224),
            **kwargs
        )

@ MODELS.register_module()
class spanetv2_s36_pure(SPANetV2):
    def __init__(self, **kwargs):
        super().__init__(
            depths=[3, 12, 18, 3],
            dims=[64, 128, 320, 512],
            token_mixers=ExSPAM,
            head_fn=MlpHead,
            input_size=(3, 224, 224),
            **kwargs
        )

@ MODELS.register_module()
class spanetv2_s36_hybrid(SPANetV2):
    def __init__(self, **kwargs):
        super().__init__(
            depths=[3, 12, 18, 3],
            dims=[64, 128, 320, 512],
            token_mixers=[ExSPAM, ExSPAM, MixAttention, Attention],
            head_fn=MlpHead,
            input_size=(3, 224, 224),
            **kwargs
        )



if __name__ == "__main__": 
    import torch 

    ckpt_path = "/home/kist-cvipl/Workspace/Projects/mmsegmentation/ckpt/spanetv2_s18_pure_res-scale_Full-ExSPAM.pth"

    model = SPANetV2(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=ExSPAM,
        head_fn=MlpHead,
        init_cfg={'checkpoint':ckpt_path},
        pretrained=ckpt_path,
        )
        
    model.eval()

    image_size = [800, 1280] # (H,W)
    input = torch.rand(1, 3, *image_size)

    out = model(input)

    for i in range(len(out)):
        print(f"stage:{i}: {out[i].shape}")


    print(model)


    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)


