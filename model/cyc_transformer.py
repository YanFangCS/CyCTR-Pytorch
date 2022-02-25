import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import math 
import random
from model.ops.modules import MSDeformAttn
from model.positional_encoding import SinePositionalEncoding

class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int): The number of fully-connected layers in FFNs.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.dropout = dropout
        self.activate = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels, bias=False), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims, bias=False))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

class MyCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim, bias=False)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.ass_drop   = nn.Dropout(0.1)

        self.drop_prob = 0.1


    def forward(self, q, k, v, supp_valid_mask=None, supp_mask=None, cyc=True):
        B, N, C = q.shape
        N_s = k.size(1)

        q = self.q_fc(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    
        k = self.k_fc(k).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_fc(v).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if supp_valid_mask is not None:
            supp_valid_mask = supp_valid_mask.unsqueeze(1).repeat(1, self.num_heads, 1) # [bs, nH, n]

        attn = (q @ k.transpose(-2, -1)) * self.scale # [bs, nH, n, n]

        if supp_mask is not None and cyc==True:
            association = []
            for hd_id in range(self.num_heads):
                attn_single_hd = attn[:, hd_id, ...]
                k2q_sim_idx = attn_single_hd.max(1)[1] # [bs, n]

                q2k_sim_idx = attn_single_hd.max(2)[1] # [bs, n]

                re_map_idx = torch.gather(q2k_sim_idx, 1, k2q_sim_idx)
                re_map_mask = torch.gather(supp_mask, 1, re_map_idx)

                asso_single_head = (supp_mask == re_map_mask).to(attn.device) # [bs, n], True means matched position in supp
                association.append(asso_single_head.unsqueeze(1))
            association = torch.cat(association, dim=1) # [bs, nH, ns]

        if cyc:
            inconsistent = ~association
            inconsistent = inconsistent.float()
            inconsistent = self.ass_drop(inconsistent)
            supp_valid_mask[inconsistent>0] = 1.
            

        if supp_valid_mask is not None:
            supp_valid_mask = supp_valid_mask.unsqueeze(-2).float() # [bs, nH, 1, ns]
            supp_valid_mask = supp_valid_mask * -10000.0
            attn = attn + supp_valid_mask       

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class CyCTransformer(nn.Module):
    def __init__(self,
                 embed_dims=384, 
                 num_heads=8, 
                 num_layers=2,
                 num_levels=1,
                 num_points=9,
                 use_ffn=True,
                 dropout=0.1,
                 shot=1,
                 rand_fg_num=300, 
                 rand_bg_num=300, 
                 ):
        super(CyCTransformer, self).__init__()
        self.embed_dims             = embed_dims
        self.num_heads              = num_heads
        self.num_layers             = num_layers
        self.num_levels             = num_levels
        self.num_points             = num_points
        self.use_ffn                = use_ffn
        self.feedforward_channels   = embed_dims*3
        self.dropout                = dropout
        self.shot                   = shot
        self.use_cross              = True
        self.use_self               = True
        self.use_cyc                = True
        
        self.rand_fg_num = rand_fg_num * shot
        self.rand_bg_num = rand_bg_num * shot

        if self.use_cross:
            self.cross_layers = []
        self.qry_self_layers  = []
        self.layer_norms = []
        self.ffns = []
        for l_id in range(self.num_layers):
            if self.use_cross:
                self.cross_layers.append(
                    MyCrossAttention(embed_dims, num_heads=12 if embed_dims%12==0 else self.num_heads, attn_drop=self.dropout, proj_drop=self.dropout),
                )
                self.layer_norms.append(nn.LayerNorm(embed_dims))
                if self.use_ffn:
                    self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                    self.layer_norms.append(nn.LayerNorm(embed_dims))
            
            if self.use_self:
                self.qry_self_layers.append(
                    MSDeformAttn(embed_dims, num_levels, 12 if embed_dims%12==0 else self.num_heads, num_points)
                )
                self.layer_norms.append(nn.LayerNorm(embed_dims))

                if self.use_ffn:
                    self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                    self.layer_norms.append(nn.LayerNorm(embed_dims))

        if self.use_cross: 
            self.cross_layers = nn.ModuleList(self.cross_layers)
        if self.use_self:
            self.qry_self_layers  = nn.ModuleList(self.qry_self_layers)
        if self.use_ffn:
            self.ffns         = nn.ModuleList(self.ffns)
        self.layer_norms  = nn.ModuleList(self.layer_norms)

        self.positional_encoding = SinePositionalEncoding(embed_dims//2, normalize=True) 
        self.level_embed = nn.Parameter(torch.rand(num_levels, embed_dims))
        nn.init.xavier_uniform_(self.level_embed)

        self.proj_drop  = nn.Dropout(dropout)
            

    def init_weights(self, distribution='uniform'):
        """Initialize the transformer weights."""
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)
        return reference_points

    def get_qry_flatten_input(self, x, qry_masks):
        src_flatten = [] 
        qry_valid_masks_flatten = []
        pos_embed_flatten = []
        spatial_shapes = []        
        for lvl in range(self.num_levels):   
            src = x[lvl]
            bs, c, h, w = src.shape
            
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).permute(0, 2, 1) # [bs, c, h*w] -> [bs, h*w, c]
            src_flatten.append(src)

            if qry_masks is not None:
                qry_mask = qry_masks[lvl]
                qry_valid_mask = []
                qry_mask = F.interpolate(
                    qry_mask.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
                for img_id in range(bs):
                    qry_valid_mask.append(qry_mask[img_id]==255)
                qry_valid_mask = torch.stack(qry_valid_mask, dim=0)
            else:
                qry_valid_mask = torch.zeros((bs, h, w))

            pos_embed = self.positional_encoding(qry_valid_mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            pos_embed_flatten.append(pos_embed)

            qry_valid_masks_flatten.append(qry_valid_mask.flatten(1))

        src_flatten = torch.cat(src_flatten, 1) # [bs, num_elem, c]
        qry_valid_masks_flatten = torch.cat(qry_valid_masks_flatten, dim=1) # [bs, num_elem]
        pos_embed_flatten = torch.cat(pos_embed_flatten, dim=1) # [bs, num_elem, c]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) # [num_lvl, 2]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # [num_lvl]
        
        return src_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index

    def get_supp_flatten_input(self, s_x, supp_mask, s_padding_mask):
        s_x_flatten = []
        supp_valid_mask = []
        supp_obj_mask = []
        supp_mask = F.interpolate(supp_mask, size=s_x.shape[-2:], mode='nearest').squeeze(1) # [bs*shot, h, w]
        supp_mask = supp_mask.view(-1, self.shot, s_x.size(2), s_x.size(3))

        s_padding_mask = F.interpolate(s_padding_mask, size=s_x.shape[-2:], mode='nearest').squeeze(1) # [bs*shot, h, w]
        s_padding_mask = s_padding_mask.view(-1, self.shot, s_x.size(2), s_x.size(3))
        s_x = s_x.view(-1, self.shot, s_x.size(1), s_x.size(2), s_x.size(3))

        for st_id in range(s_x.size(1)):
            supp_valid_mask_s = []
            supp_obj_mask_s = []
            for img_id in range(s_x.size(0)):
                supp_valid_mask_s.append(s_padding_mask[img_id, st_id, ...]==255)
                obj_mask = supp_mask[img_id, st_id, ...]==1
                if obj_mask.sum() == 0: # To avoid NaN
                    obj_mask[obj_mask.size(0)//2-1:obj_mask.size(0)//2+1, obj_mask.size(1)//2-1:obj_mask.size(1)//2+1] = True
                if (obj_mask==False).sum() == 0: # To avoid NaN
                    obj_mask[0, 0]   = False
                    obj_mask[-1, -1] = False 
                    obj_mask[0, -1]  = False
                    obj_mask[-1, 0]  = False
                supp_obj_mask_s.append(obj_mask)
            supp_valid_mask_s = torch.stack(supp_valid_mask_s, dim=0) # [bs, h, w]
            supp_valid_mask_s = supp_valid_mask_s.flatten(1) # [bs, h*w]
            supp_valid_mask.append(supp_valid_mask_s)

            supp_obj_mask_s = torch.stack(supp_obj_mask_s, dim=0)
            supp_obj_mask_s = (supp_obj_mask_s==1).flatten(1) # [bs, n]
            supp_obj_mask.append(supp_obj_mask_s)

            s_x_s = s_x[:, st_id, ...]
            s_x_s = s_x_s.flatten(2).permute(0, 2, 1)  # [bs, c, h*w] -> [bs, h*w, c]
            s_x_flatten.append(s_x_s)

        s_x_flatten = torch.cat(s_x_flatten, 1) # [bs, h*w*shot, c]
        supp_valid_mask = torch.cat(supp_valid_mask, 1)
        supp_mask_flatten = torch.cat(supp_obj_mask, 1)

        return s_x_flatten, supp_valid_mask, supp_mask_flatten

    def sparse_sampling(self, s_x, supp_mask, supp_valid_mask):
        if self.training:
            scale_min = 0.6
            scale_max = 4.0 if self.shot==1 else 1.4
            sampling_scale = random.uniform(scale_min, scale_max)
            rand_fg_num = int(self.rand_fg_num*sampling_scale)
            rand_bg_num = int(self.rand_bg_num*sampling_scale)
        else:
            rand_fg_num = self.rand_fg_num
            rand_bg_num = self.rand_bg_num
        assert supp_mask is not None
        re_arrange_k = []
        re_arrange_mask = []
        re_arrange_valid_mask = []
        for b_id in range(s_x.size(0)):
            k_b = s_x[b_id] # [num_elem, c]
            supp_mask_b = supp_mask[b_id] # [num_elem]
            num_fg = supp_mask_b.sum()
            num_bg = (supp_mask_b==False).sum()
            fg_k = k_b[supp_mask_b] # [num_fg, c]
            bg_k = k_b[supp_mask_b==False] # [num_bg, c]

            if num_fg<rand_fg_num:
                rest_num = rand_fg_num+rand_bg_num-num_fg
                bg_select_idx = torch.randperm(num_bg)[:rest_num]                
                re_k = torch.cat([fg_k, bg_k[bg_select_idx]], dim=0)
                re_mask = torch.cat([supp_mask_b[supp_mask_b==True], supp_mask_b[bg_select_idx]], dim=0)
                re_valid_mask = torch.cat([supp_valid_mask[b_id][supp_mask_b==True], supp_valid_mask[b_id][bg_select_idx]], dim=0)

            elif num_bg<rand_bg_num:
                rest_num = rand_fg_num+rand_bg_num-num_bg
                fg_select_idx = torch.randperm(num_fg)[:rest_num]
                re_k = torch.cat([fg_k[fg_select_idx], bg_k], dim=0)
                re_mask = torch.cat([supp_mask_b[fg_select_idx], supp_mask_b[supp_mask_b==False]], dim=0)
                re_valid_mask = torch.cat([supp_valid_mask[b_id][fg_select_idx], supp_valid_mask[b_id][supp_mask_b==False]], dim=0)
                
            else:
                fg_select_idx = torch.randperm(num_fg)[:rand_fg_num]
                bg_select_idx = torch.randperm(num_bg)[:rand_bg_num]
                re_k = torch.cat([fg_k[fg_select_idx], bg_k[bg_select_idx]], dim=0)
                re_mask = torch.cat([supp_mask_b[fg_select_idx], supp_mask_b[bg_select_idx]], dim=0)
                re_valid_mask = torch.cat([supp_valid_mask[b_id][fg_select_idx], supp_valid_mask[b_id][bg_select_idx]], dim=0)
            
            re_arrange_k.append(re_k)
            re_arrange_mask.append(re_mask) 
            re_arrange_valid_mask.append(re_valid_mask)

        k = torch.stack(re_arrange_k, dim=0)
        supp_mask = torch.stack(re_arrange_mask, dim=0) #[bs, num_select]
        supp_valid_mask = torch.stack(re_arrange_valid_mask, dim=0) #[bs, num_select]

        return k, supp_mask, supp_valid_mask

    def forward(self, x, qry_masks, s_x, supp_mask, s_padding_mask):
        if not isinstance(x, list):
            x = [x]
        if not isinstance(qry_masks, list):
            qry_masks = [qry_masks.clone() for _ in range(self.num_levels)]

        assert len(x) == len(qry_masks) == self.num_levels
        bs, c = x[0].size()[:2]

        x_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index = self.get_qry_flatten_input(x, qry_masks)

        s_x, supp_valid_mask, supp_mask_flatten = self.get_supp_flatten_input(s_x, supp_mask.clone(), s_padding_mask.clone())

        reference_points = self.get_reference_points(spatial_shapes, device=x_flatten.device)

        q = x_flatten
        pos = pos_embed_flatten
        
        ln_id = 0
        ffn_id = 0
        for l_id in range(self.num_layers):
            if self.use_self:
                q =  q + self.proj_drop(self.qry_self_layers[l_id](q + pos, reference_points, q, spatial_shapes, level_start_index, qry_valid_masks_flatten))
                q = self.layer_norms[ln_id](q)
                ln_id += 1
       
                if self.use_ffn:
                    q = self.ffns[ffn_id](q)
                    ffn_id += 1
                    q = self.layer_norms[ln_id](q)
                    ln_id += 1

            if self.use_cross:
                k, sampled_mask, sampled_valid_mask = self.sparse_sampling(s_x, supp_mask_flatten, supp_valid_mask) if self.training or l_id==0 else (k, sampled_mask, sampled_valid_mask)
                v = k.clone()
                cross_out = self.cross_layers[l_id](q, k, v, sampled_valid_mask, sampled_mask, cyc=self.use_cyc)

                q = cross_out + q
                q = self.layer_norms[ln_id](q)
                ln_id += 1

                if self.use_ffn:
                    q = self.ffns[ffn_id](q)
                    ffn_id += 1
                    q = self.layer_norms[ln_id](q)
                    ln_id += 1

        qry_feat = q.permute(0, 2, 1) # [bs, c, num_ele]
        qry_feat_decouple = []
        for lvl in range(self.num_levels):
            start_idx = level_start_index[lvl].long()
            h, w = spatial_shapes[lvl]
            qry_feat_decouple.append(qry_feat[:, :, start_idx:start_idx+h*w].view(bs, c, h, w))

        return qry_feat_decouple


