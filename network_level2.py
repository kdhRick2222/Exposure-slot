import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import numbers
from einops import rearrange
from torchvision.utils import save_image
from scipy.optimize import linear_sum_assignment
from torch import tensor as to_tensor
from torchvision.transforms.functional import pil_to_tensor
from config.basic import ConfigBasic
from torch.nn import init


            
##################################################################################################
##################################################################################################

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),)

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


def to_3d(x): return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w): return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##################################################################################################
##################################################################################################

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.relu = nn.ReLU(inplace=True)
        
        self.slots_mu = nn.Parameter(torch.randn(1, self.num_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)
        self.prompts = nn.Parameter(torch.rand(1, self.num_slots, dim))

        self.to_q = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_input  = nn.LayerNorm(dim) #
        self.norm_k_inp  = nn.LayerNorm(dim)
        self.norm_v_inp  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, q_inp, k_inp, v_inp): # [B, HW, C]
        # print(k_inp.shape, v_inp.shape)
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = self.num_slots

        slots = self.slots_mu.repeat(b, 1, 1)
        prompts = self.prompts.repeat(b, 1, 1)

        k, v = self.norm_k_inp(k_inp), self.norm_v_inp(v_inp)

        for _ in range(self.iters):
            slots_prev = slots[:, :, :self.dim]

            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            q_slot = q[:, :, :self.dim]
            q_prompt = q[:, :, self.dim:]

            dots = torch.einsum('bid,bjd->bij', q_slot, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps

            attn_ = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn_)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        
        prompt_feature = torch.einsum('bnd,bnp->bdp', prompts, attn) # [B, dim, HW]]
        slot_feature = torch.einsum('bnd,bnp->bdp', slots, attn) # [B, dim, HW]]

        slot_cossim_total = slot_feature.permute(0, 2, 1)

        return slots, attn, prompt_feature.permute(0, 2, 1), slot_feature.permute(0, 2, 1), slot_cossim_total


class sub_SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.relu = nn.ReLU(inplace=True) 

        self.slots_mu = nn.Parameter(torch.randn(1, self.num_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)
        self.prompts = nn.Parameter(torch.rand(1, self.num_slots, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_input  = nn.LayerNorm(dim)
        self.norm_k_inp  = nn.LayerNorm(dim)
        self.norm_v_inp  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, q_inp, k_inp, v_inp): # [B, HW, C]
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = self.num_slots

        slots = self.slots_mu.repeat(b, 1, 1)
        prompts = self.prompts.repeat(b, 1, 1)

        k, v = self.norm_k_inp(k_inp), self.norm_v_inp(v_inp)
        sub_attn_list = []

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            sub_attn_list.append(attn)

            attn_ = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn_)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        prompt_feature = torch.einsum('bnd,bnp->bdp', prompts, attn) # [B, dim, HW]]
        slot_feature = torch.einsum('bnd,bnp->bdp', slots, attn) # [B, dim, HW]]

        return slots, prompts, prompt_feature.permute(0, 2, 1), slot_feature.permute(0, 2, 1), sub_attn_list


class Slot_in_slot_Attention(nn.Module):
    def __init__(self, num_slots, num_subslots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_subslots = num_subslots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.relu = nn.ReLU(inplace=True)
        
        self.slots_mu = nn.Parameter(torch.randn(1, self.num_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)
        self.prompts = nn.Parameter(torch.rand(1, self.num_slots, dim))

        self.to_q = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_k_inp  = nn.LayerNorm(dim)
        self.norm_v_inp  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.sub_slot_attention = sub_SlotAttention(num_slots=self.num_subslots, dim=self.dim)
    
    def forward(self, inputs, q_inp, k_inp, v_inp): # [B, HW, C]
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = self.num_slots

        slots = self.slots_mu.repeat(b, 1, 1)
        prompts = self.prompts.repeat(b, 1, 1)
        k, v = self.norm_k_inp(k_inp), self.norm_v_inp(v_inp)

        main_attn_list = []
        sub_promptfeature_rebuttal = []
        v_rebuttal, k_rebuttal = [], []

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            
            attn = dots.softmax(dim=1) + self.eps
            main_attn_list.append(attn)
            attn_ = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn_)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            prompt_feature_ = torch.einsum('bnd,bnp->bdp', self.prompts, attn) # [B, dim, HW]
            k, v = k * prompt_feature_.permute(0, 2, 1), v * prompt_feature_.permute(0, 2, 1)
            sub_slots, sub_prompts, sub_slotfeature, sub_promptfeature, sub_attn = self.sub_slot_attention(inputs * prompt_feature_.permute(0, 2, 1), q, k, v)
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

            k = k * sub_promptfeature
            v = v * sub_promptfeature
            sub_promptfeature_rebuttal.append(sub_promptfeature)
            v_rebuttal.append(v)
            k_rebuttal.append(k)

        prompt_feature = torch.einsum('bnd,bnp->bdp', prompts, attn) # [B, dim, HW]
        slot_feature = torch.einsum('bnd,bnp->bdp', slots, attn) # [B, dim, HW]]
        slot_cossim_total = [v * prompt_feature.permute(0, 2, 1) * sub_promptfeature, prompt_feature.permute(0, 2, 1) * sub_promptfeature_rebuttal[1], prompt_feature.permute(0, 2, 1)]
        attn_total = torch.cat(main_attn_list+sub_attn, dim=1)

        return slots, attn_total, prompt_feature.permute(0, 2, 1) * sub_promptfeature, slot_feature.permute(0, 2, 1) * sub_slotfeature, slot_cossim_total


##################################################################################################
##################################################################################################


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),)

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            num_slots=3,
            num_subslots=7,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.num_slots = num_slots
        self.num_subslots = num_subslots

        slot_list = []
        for i in range(heads):
            # slot_list.append(SlotAttention(num_slots=self.num_slots, dim=self.dim_head)) 
            slot_list.append(Slot_in_slot_Attention(num_slots=self.num_slots, num_subslots = self.num_subslots, dim=self.dim_head))
        self.slot_list = nn.Sequential(*slot_list)

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        illu_fea: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        attn_list, prompt_list, slot_list, slots_cossim_list = [], [], [], []
        for i in range(len(self.slot_list)):
            slot_attention = self.slot_list[i]
            slots_, attn_maps_, prompt_map_, slot_map_, slot_cossim_total = slot_attention(x, q_inp[:, :, i*c:(i+1)*c], k_inp[:, :, i*c:(i+1)*c], v_inp[:, :, i*c:(i+1)*c]) # slot_map.shape == [8, 4096, 128]
            attn_list.append(attn_maps_)
            prompt_list.append(prompt_map_)
            slot_list.append(slot_map_)
            slots_cossim_list = slot_cossim_total


        attn_maps = torch.cat(attn_list, dim=1)
        prompt_map = torch.cat(prompt_list, dim=2)
        slot_map = torch.cat(slot_list, dim=2)
        
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inp, k_inp, v_inp, prompt_map))
        v = v * illu_attn

        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q  # shape: [8, 2, 128, 128]
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out = out_c 
        return out, attn_maps, slot_map.view(b, h, w, -1).permute(0, 3, 1, 2), slots_cossim_list


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            num_slots=3,
            num_subslots=7,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, num_slots=num_slots, num_subslots=num_subslots, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        
        x = x.permute(0, 2, 3, 1)
        attn_list = []
        for (attn, ff) in self.blocks:
            x_attn, attn_maps, slot_features, slots_cossim_list = attn(x)
            attn_list.append(attn_maps)
            x = x_attn + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out, torch.cat(attn_list, dim=1), slot_features, slots_cossim_list


##########################################################################
## Resizing modules ##
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
    
    def build_grid(self, resolution, device):
        ranges = [np.linspace(0., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        grid = self.build_grid((H, W), inputs.device)
        grid = self.embedding(grid)
        # print(grid.shape, inputs.shape)
        return inputs + grid.permute(0, 3, 1, 2)

##########################################################################

class Slot_Decoder(nn.Module):
    def __init__(self, hid_dim=128):
        super().__init__()
        self.dim = hid_dim//2
        self.upsample1 = Upsample(self.dim*2)
        self.upsample2 = Upsample(self.dim)
        self.activation = nn.GELU()
        self.conv1 = nn.Conv2d(self.dim, self.dim, 3, padding=1)
        self.conv2 = nn.Conv2d(self.dim, self.dim, 3, padding=1)
        self.conv3 = nn.Conv2d(self.dim//2, self.dim//2, 3, padding=1)
        self.conv4 = nn.Conv2d(self.dim//2, self.dim//4, 3, padding=1)
        self.conv5 = nn.Conv2d(self.dim//4, 3, 3, padding=1)
        self.activation = nn.GELU()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        
        x = self.upsample2(x)
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = self.conv5(x)
        return x

##########################################################################

class Slot_model(nn.Module):
    
    def __init__(self, cfg, use_slot=True):
        super(Slot_model, self).__init__()

        self.device = cfg.device
        self.mse_loss = nn.MSELoss()

        self.use_slot = use_slot

        self.dim = 32
        # self.soft_positionembed = SoftPositionEmbed(self.dim*4)
        self.downsample1 = Downsample(self.dim)
        self.downsample2 = Downsample(self.dim*2)
        self.upsample1 = Upsample(self.dim*4)
        self.upsample2 = Upsample(self.dim*2)
        self.batchnorm1 = nn.BatchNorm2d(self.dim)
        self.batchnorm2 = nn.BatchNorm2d(self.dim*2)
        self.batchnorm3 = nn.BatchNorm2d(self.dim*4)

        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.conv1_1 = nn.Conv2d(3, self.dim, 3, padding=1)
        self.conv1_2 = nn.Conv2d(self.dim, self.dim, 3, padding=1)
        self.conv2_2 = nn.Conv2d(self.dim*2, self.dim*2, 3, padding=1)
        self.conv3_2 = nn.Conv2d(self.dim*4, self.dim*4, 3, padding=1)
        self.conv4_1 = nn.Conv2d(self.dim*4, self.dim*2, 3, padding=1)
        self.conv4_2 = nn.Conv2d(self.dim*2, self.dim*2, 3, padding=1)
        self.conv5_1 = nn.Conv2d(self.dim*2, self.dim, 3, padding=1)
        self.conv5_2 = nn.Conv2d(self.dim, self.dim, 3, padding=1)
        self.conv6 = nn.Conv2d(self.dim, 3, 1)

        self.slot_num = 3
        self.subslot_num = 7
        ## SSAB part
        self.TransformerBlock = IGAB(dim=self.dim*4, num_blocks=1, dim_head=self.dim*4, num_slots=self.slot_num, num_subslots=self.subslot_num, heads=1)
        self.slot_decoder = Slot_Decoder(hid_dim=self.dim*4)
        self.conv_fusion = nn.Conv2d(self.dim*8, self.dim*4, 3, padding=1)

    def forward(self, x, gt, inference=False):
        B, C, H, W = x.shape
        dH = H%4
        dW = W%4
        if dH!=0 or dW!=0:
            x = F.interpolate(x, (H - dH, W - dW),mode="bilinear")
            gt = F.interpolate(gt, (H - dH, W - dW),mode="bilinear")

        x_input = x
        x_gt = gt

        # Encoder-input
        x = self.activation(self.conv1_1(x))
        conv1 = self.batchnorm1(self.conv1_2(x))
        conv2 = self.activation(self.downsample1(conv1))
        conv2 = self.batchnorm2(self.conv2_2(conv2))
        # bottleneck
        conv3 = self.activation(self.downsample2(conv2))
        conv3 = self.batchnorm3(self.conv3_2(conv3))

        B3, C3, H3, W3 = conv3.shape
        feature_i, attn_maps, slot_features, slots_cossim_list = self.TransformerBlock(conv3)

        if inference == 1:
            recon_slot = x_gt
        else:
            recon_slot = self.slot_decoder(slot_features)
        
        # Decoder-input
        conv4 = self.upsample1(feature_i)
        up4 = torch.cat([conv4, conv2], 1)
        up4 = self.activation(self.conv4_1(up4))
        up4 = self.activation(self.conv4_2(up4))
        conv5 = self.upsample2(up4)
        up5 = torch.cat([conv5, conv1], 1)
        up5 = self.activation(self.conv5_1(up5))
        up5 = self.activation(self.conv5_2(up5))
            
        # output trans.
        output = self.conv6(up5) + x_input

        if dH!=0 or dW!=0:
            output = F.interpolate(output, (H, W),mode="bilinear")
            recon_slot = F.interpolate(recon_slot, (H, W),mode="bilinear")
            x_gt = F.interpolate(x_gt, (H, W),mode="bilinear")

        feature_loss = self.mse_loss(recon_slot, x_gt)

        return output, recon_slot, feature_loss


if __name__ == '__main__':
    
    cfg = ConfigBasic()
    cfg.device = "cuda:0"
    model = Slot_model(cfg, use_slot=True, inference=True)

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb), get_n_params(model))

    x = torch.randn(8, 3, 256, 256)
    label = torch.where(torch.randn(8)>0, 1, 0)
    x, _, slot_loss_total = model(x, x)
