import torch
from torch import nn
import torch.nn.functional as F

from model.resnet import *
from model.loss import WeightedDiceLoss
from model.cyc_transformer import CyCTransformer
from model.ops.modules import MSDeformAttn
from model.backbone_utils import Backbone

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

class CyCTR(nn.Module):
    def __init__(self, layers=50, classes=2, shot=1, reduce_dim=384, \
        criterion=WeightedDiceLoss(), with_transformer=True, trans_multi_lvl=1):
        super(CyCTR, self).__init__()
        assert layers in [50, 101]
        assert classes > 1 
        self.layers = layers
        self.criterion = criterion
        self.shot = shot
        self.with_transformer = with_transformer
        if self.with_transformer:
            self.trans_multi_lvl = trans_multi_lvl
        self.reduce_dim = reduce_dim

        self.print_params()

        in_fea_dim = 1024 + 512      

        drop_out = 0.5

        self.adjust_feature_supp = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        ) 
        self.adjust_feature_qry = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        )

        self.high_avg_pool = nn.Identity()

        prior_channel = 1 
        self.qry_merge_feat = nn.Sequential(
                    nn.Conv2d(reduce_dim*2+prior_channel, reduce_dim, kernel_size=1, padding=0, bias=False),
                    nn.ReLU(inplace=True),
                )

        if self.with_transformer:
            self.supp_merge_feat = nn.Sequential(
                nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
            self.addtional_proj = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, bias=False)
            )
            self.transformer = CyCTransformer(embed_dims=reduce_dim, shot=self.shot, num_points=9)
            self.merge_multi_lvl_reduce = nn.Sequential(
                    nn.Conv2d(reduce_dim*self.trans_multi_lvl, reduce_dim, kernel_size=1, padding=0, bias=False),
                    nn.ReLU(inplace=True),
                )
            self.merge_multi_lvl_sum = nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )
        else:
            self.merge_res = nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )

        self.cls = nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.1),
                    nn.Conv2d(reduce_dim, classes, kernel_size=1)
                )

        qry_dim_scalar = 1
        self.pred_supp_qry_proj = nn.Sequential(
                nn.Conv2d(reduce_dim*qry_dim_scalar, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
        scalar = 2
        self.supp_init_merge = nn.Sequential(
                nn.Conv2d(reduce_dim*scalar, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
        self.supp_beta_conv = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
        self.supp_cls = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            )

        self.init_weights()
        self.backbone = Backbone('resnet{}'.format(layers), train_backbone=False, return_interm_layers=True, dilation=[False, True, True])


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()


    def print_params(self):
        repr_str = self.__class__.__name__
        repr_str += f'(backbone layers={self.layers}, '
        repr_str += f'reduce_dim={self.reduce_dim}, '
        repr_str += f'shot={self.shot}, '
        repr_str += f'with_transformer={self.with_transformer})'
        print(repr_str)
        return repr_str


    def forward(self, x, s_x=torch.FloatTensor(1,1,3,473,473).cuda(), s_y=torch.FloatTensor(1,1,473,473).cuda(), y=None, padding_mask=None, s_padding_mask=None):
        batch_size, _, h, w = x.size()
        assert (h-1) % 8 == 0 and (w-1) % 8 == 0
        img_size = x.size()[-2:]

        # backbone feature extraction
        qry_bcb_fts = self.backbone(x)
        supp_bcb_fts = self.backbone(s_x.view(-1, 3, *img_size)) 
        query_feat = torch.cat([qry_bcb_fts['1'], qry_bcb_fts['2']], dim=1)
        supp_feat = torch.cat([supp_bcb_fts['1'], supp_bcb_fts['2']], dim=1)
        query_feat = self.adjust_feature_qry(query_feat)
        mid_query_feat = query_feat.clone()
        supp_feat = self.adjust_feature_supp(supp_feat) 

        fts_size = query_feat.shape[-2:]
        supp_mask = F.interpolate((s_y==1).view(-1, *img_size).float().unsqueeze(1), size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)         
 
        # global feature extraction
        supp_feat_list = []
        r_supp_feat = supp_feat.view(batch_size, self.shot, -1, fts_size[0], fts_size[1])
        for st in range(self.shot):
            mask = (s_y[:,st,:,:]==1).float().unsqueeze(1)
            mask = F.interpolate(mask, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            tmp_supp_feat = r_supp_feat[:,st,...]
            tmp_supp_feat = Weighted_GAP(tmp_supp_feat, mask)
            supp_feat_list.append(tmp_supp_feat)    
        global_supp_pp = supp_feat_list[0]
        if self.shot > 1:
            for i in range(1, len(supp_feat_list)):
                global_supp_pp += supp_feat_list[i]
            global_supp_pp /= len(supp_feat_list)
            multi_supp_pp = Weighted_GAP(supp_feat, supp_mask) # [bs*shot, c, 1, 1]
        else:
            multi_supp_pp = global_supp_pp

        # prior generation
        query_feat_high = qry_bcb_fts['3']
        supp_feat_high = supp_bcb_fts['3'].view(batch_size, -1, 2048, fts_size[0], fts_size[1])
        corr_query_mask = self.generate_prior(query_feat_high, supp_feat_high, s_y, fts_size)
        
        # feature mixing
        query_cat_feat = [query_feat, global_supp_pp.expand(-1, -1, fts_size[0], fts_size[1]), corr_query_mask]
        query_feat = self.qry_merge_feat(torch.cat(query_cat_feat, dim=1))

        if self.with_transformer:
            to_merge_fts = [supp_feat, multi_supp_pp.expand(-1, -1, fts_size[0], fts_size[1])]
            aug_supp_feat = torch.cat(to_merge_fts, dim=1)
            aug_supp_feat = self.supp_merge_feat(aug_supp_feat)

            query_feat_list = self.transformer(query_feat, padding_mask.float(), aug_supp_feat, s_y.clone().float(), s_padding_mask.float())
            fused_query_feat = []
            for lvl, qry_feat in enumerate(query_feat_list):
                if lvl == 0:
                    fused_query_feat.append(qry_feat)
                else:
                    fused_query_feat.append(F.interpolate(qry_feat, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True))
            fused_query_feat = torch.cat(fused_query_feat, dim=1)
            fused_query_feat = self.merge_multi_lvl_reduce(fused_query_feat)
            fused_query_feat = self.merge_multi_lvl_sum(fused_query_feat)+fused_query_feat

        else:
            query_feat = self.merge_res(query_feat) + query_feat
            query_feat_list = [query_feat]
            fused_query_feat = query_feat.clone()


        # Output Part
        out = self.cls(fused_query_feat)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            
        if self.training:
            # prepare inputs for aux loss
            qry_mask = F.interpolate((y==1).float().unsqueeze(1), size=(fused_query_feat.size(2), fused_query_feat.size(3)), mode='bilinear', align_corners=True) # 'nearest')
            qry_proj_feat = self.pred_supp_qry_proj(fused_query_feat)+mid_query_feat
            qry_pp = Weighted_GAP(qry_proj_feat, qry_mask)
            qry_pp = qry_pp
            qry_pp = qry_pp.expand(-1, -1, supp_feat.size(2), supp_feat.size(3)) # default
            temp_supp_feat = supp_feat.view(batch_size, self.shot, -1, supp_feat.size(2), supp_feat.size(3))
            supp_out_list = []
            for st_id in range(self.shot):
                supp_merge_bin = torch.cat([temp_supp_feat[:, st_id, ...], qry_pp], dim=1)
                merge_supp_feat = self.supp_init_merge(supp_merge_bin)
                merge_supp_feat = self.supp_beta_conv(merge_supp_feat) + merge_supp_feat
                supp_out = self.supp_cls(merge_supp_feat)
                supp_out_list.append(supp_out)

            # calculate loss
            main_loss = self.criterion(out, y.long()) 
            out_list = []
            for lvl, query_feat in enumerate(query_feat_list):
                inter_out = self.cls[lvl](query_feat)
                out_list.append(F.interpolate(inter_out, size=(h, w), mode='bilinear', align_corners=True))

            aux_loss = torch.zeros_like(main_loss)
            for st_id, supp_out in enumerate(supp_out_list):
                supp_out = F.interpolate(supp_out, size=(h, w), mode='bilinear', align_corners=True)
                supp_loss = self.criterion(supp_out, s_y[:, st_id, ...].long())
                aux_loss += supp_loss/self.shot

            return out.max(1)[1], main_loss, aux_loss
        else:
            return out


    def generate_prior(self, query_feat_high, supp_feat_high, s_y, fts_size):
        bsize, _, sp_sz, _ = query_feat_high.size()[:]
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for st in range(self.shot):
            tmp_mask = (s_y[:,st,:,:] == 1).float().unsqueeze(1)
            tmp_mask = F.interpolate(tmp_mask, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)

            tmp_supp_feat = supp_feat_high[:,st,...] * tmp_mask               
            q = self.high_avg_pool(query_feat_high.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
            s = self.high_avg_pool(tmp_supp_feat.flatten(2).transpose(-2, -1))  # [bs, h*w, c]

            tmp_query = q
            tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous() 
            tmp_supp = tmp_supp.contiguous() 
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)     
        return corr_query_mask
