import torch
import torch.nn as nn
import torchvision
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.models._utils import IntermediateLayerGetter
import timm
import numpy as np

# sys.path.append("/mnt2/wangyuntao/geoLoc/model")
from .common import scTransformerLayer, scTransformerEncoder, PositionEncodingSine, PSP

# sys.path.append("..")
# from visual import gem

"""
ConvNext model name
{
    'convnext_tiny_in22ft1k': 'convnext_tiny.fb_in22k_ft_in1k',
    'convnext_small_in22ft1k': 'convnext_small.fb_in22k_ft_in1k',
    'convnext_base_in22ft1k': 'convnext_base.fb_in22k_ft_in1k',
    'convnext_large_in22ft1k': 'convnext_large.fb_in22k_ft_in1k',
    'convnext_xlarge_in22ft1k': 'convnext_xlarge.fb_in22k_ft_in1k',
    'convnext_tiny_384_in22ft1k': 'convnext_tiny.fb_in22k_ft_in1k_384',
    'convnext_small_384_in22ft1k': 'convnext_small.fb_in22k_ft_in1k_384',
    'convnext_base_384_in22ft1k': 'convnext_base.fb_in22k_ft_in1k_384',
    'convnext_large_384_in22ft1k': 'convnext_large.fb_in22k_ft_in1k_384',
    'convnext_xlarge_384_in22ft1k': 'convnext_xlarge.fb_in22k_ft_in1k_384',
    'convnext_tiny_in22k': 'convnext_tiny.fb_in22k',
    'convnext_small_in22k': 'convnext_small.fb_in22k',
    'convnext_base_in22k': 'convnext_base.fb_in22k',
    'convnext_large_in22k': 'convnext_large.fb_in22k',
    'convnext_xlarge_in22k': 'convnext_xlarge.fb_in22k',

    'convnextv2_tiny_22k_224_ema': 'convnextv2_tiny.fcmae_ft_in22k_in1k'
    'convnextv2_tiny_22k_384_ema': 'convnextv2_tiny.fcmae_ft_in22k_in1k_384'
}
)

"""


class Backbone(nn.Module):
    def __init__(self, model_name, bk_checkpoint, return_interm_layers: bool, img_size=[122, 671]):
        super().__init__()
        self.name = model_name 
        # print('\nname\n',  name)
        if 'resnet' in self.name.lower():
            backbone = getattr(torchvision.models, self.name.lower())(weights='{}_Weights.IMAGENET1K_V1'.format(self.name))
            assert self.name in ('ResNet18', 'ResNet34', 'ResNet50'), "number of channels are hard coded"

            if return_interm_layers:
                # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [8, 16, 32]
                if self.name == 'ResNet50':
                    self.num_channels = [512, 1024, 2048]
                else: # resnet18 / resnet34
                    self.num_channels = [128, 256, 512]
            else:
                return_layers = {'layer4': "0"}
                self.strides = [32]
                if self.name == 'ResNet50':
                    self.num_channels = [2048]
                else: # resnet18 / resnet34
                    self.num_channels = [512]
            self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
            self.data_config = None
        elif 'convnext' in self.name.lower():
            self.backbone = timm.create_model(self.name, pretrained=True, num_classes = 0, pretrained_cfg_overlay=dict(file=bk_checkpoint))
            self.data_config = timm.data.resolve_model_data_config(self.backbone)
            if return_interm_layers:
                self.strides = [8, 16, 32]
                if 'base' in self.name.lower():
                    self.num_channels = [256, 512, 1024]
                elif 'tiny' in self.name.lower():
                    self.num_channels = [192, 384, 768]
            else:
                self.strides = [32]
                self.num_channels = [1024]

        else:
            raise RuntimeError(f'error model_name [resnet* or convnext]')

    def forward(self, x):
        if 'resnet' in self.name.lower():
            xs = self.backbone(x)
            out = []
            for _, x in xs.items():
                out.append(x)
        if 'convnext' in self.name.lower():
            x = self.backbone.stem(x)
            x0 = self.backbone.stages[0](x)

            x1 = self.backbone.stages[1](x0)
            x2 = self.backbone.stages[2](x1)
            x3 = self.backbone.stages[3](x2)
            # print(x.shape, x0.shape, x1.shape, x2.shape, x3.shape)

            out = [x1, x2, x3]
        return out


class BackboneEmbed(nn.Module):
    def __init__(self, d_model, backbone_strides, backbone_num_channels, return_interm_layers: bool):
        super().__init__()
        self.return_interm_layers = return_interm_layers

        self.d_model = 128
        self.pos_embed = PositionEncodingSine(d_model=self.d_model)


        if self.return_interm_layers:
            num_backbone_outs = len(backbone_strides) + 1
            input_proj_list = []
            for n in range(num_backbone_outs):
                if n == num_backbone_outs - 1:
                    in_channels = backbone_num_channels[n - 1]
                    input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.d_model)))
                else:
                    in_channels = backbone_num_channels[n]
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model)))
                    
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )])
        

    def forward(self, features):
        feats_embed = []
        srcs = []
        for l, feat in enumerate(features):
            src = self.input_proj[l](feat)
            srcs.append(src)
            p = self.pos_embed(src)
            feats_embed.append(p)
            # feats_embed.append(p.flatten(2).transpose(1, 2))
        if self.return_interm_layers:
            src = self.input_proj[-1](features[-1])
            srcs.append(src)
            p = self.pos_embed(src)
            feats_embed.append(p)
            # feats_embed.append(p.flatten(2).transpose(1, 2))
        return feats_embed, srcs

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class TimmModel(nn.Module):
    def __init__(self, model_name,
                       sat_size,
                       grd_size,
                       psm=True,
                       is_polar=False):
                 
        super(TimmModel, self).__init__()
        
        self.is_polar = is_polar
        self.backbone_name = model_name

        self.d_model = 128
        self.nheads = 4
        self.nlayers = 2
        self.ffn_dim = 1024
        self.dropout = 0.3
        self.em_dim = 2048

        self.activation = nn.GELU()
        self.single_features = False

        self.sat_size = sat_size
        self.grd_size = grd_size
        
        self.sample = psm

        if 'tiny' in self.backbone_name:
            if 'v2' in self.backbone_name:
                self.bk_checkpoint = 'pretrained/convnextv2_tiny_22k_224_ema.pt' #'/mnt2/wangyuntao/pretrained/convnextv2_tiny_22k_384_ema.pt'
            else:
               self.bk_checkpoint = 'pretrained/convnext_tiny_22k_1k_224.pth' 
        elif 'base' in self.backbone_name:
            if 'v2' in self.backbone_name:
                self.bk_checkpoint = 'pretrained/convnextv2_base_22k_224_ema.pt'
            else:
                self.bk_checkpoint = 'pretrained/convnext_base_22k_1k_224.pth'
        else:
            self.bk_checkpoint = None

        if '384' in self.backbone_name:
            self.bk_checkpoint = self.bk_checkpoint.replace('224', '384')


        if self.sample:
            self.norm1 = nn.LayerNorm(self.d_model)
            self.norm2 = nn.LayerNorm(self.d_model)
            self.sample_L_sat = PSP(sizes=[(1, 1), (6, 6), (12, 12), (21, 21)], dimension=2)
            self.sample_L_grd = PSP(sizes=[(1, 1), (3, 12), (6, 24), (7, 63)], dimension=2)
            self.in_dim_L = 622

        #----------------------- global -----------------------# 
        # Backbone
        self.backbone = Backbone(self.backbone_name, self.bk_checkpoint, return_interm_layers=not self.single_features)

        # Position and embed
        self.embed = BackboneEmbed(self.d_model, self.backbone.strides, self.backbone.num_channels, return_interm_layers=not self.single_features)

        #----------------------- global -----------------------# 
        # multi-scale self-cross attention for sat
        layer_sat_H = scTransformerLayer(self.d_model, self.nheads, self.ffn_dim, self.dropout, activation=self.activation, is_ffn=True)
        self.transformer_sat_H = scTransformerEncoder(layer_sat_H, num_layers=2)
        layer_sat_L = scTransformerLayer(self.d_model, self.nheads, self.ffn_dim, self.dropout, activation=self.activation, is_ffn=True, q_low=True)
        self.transformer_sat_L = scTransformerEncoder(layer_sat_L, num_layers=1)

        # multi-scale self-cross attention for grd
        layer_grd_H = scTransformerLayer(self.d_model, self.nheads, self.ffn_dim, self.dropout, activation=self.activation, is_ffn=True)
        self.transformer_grd_H = scTransformerEncoder(layer_grd_H, num_layers=2)
        layer_grd_L = scTransformerLayer(self.d_model, self.nheads, self.ffn_dim, self.dropout, activation=self.activation, is_ffn=True, q_low=True)
        self.transformer_grd_L = scTransformerEncoder(layer_grd_L, num_layers=1)

        
        out_dim_g = 14
        self.feat_dim_sat, self.H_sat, self.W_sat = self._dim(self.backbone_name, self.backbone.strides, img_size=self.sat_size)
        in_dim_sat = sum(self.feat_dim_sat[1:]) +  self.in_dim_L if self.sample else sum(self.feat_dim_sat)
        self.proj_sat = nn.Linear(in_dim_sat, out_dim_g)


        self.feat_dim_grd, self.H_grd, self.W_grd = self._dim(self.backbone_name, self.backbone.strides, img_size=self.grd_size)
        in_dim_grd = sum(self.feat_dim_grd[1:]) +  self.in_dim_L if self.sample else sum(self.feat_dim_grd)
        self.proj_grd = nn.Linear(in_dim_grd, out_dim_g)


        #----------------------- local -----------------------# 
        self.num_channles = self.backbone.num_channels
        self.num_channles.append(self.d_model)

        ratio = 1
        proj_gl_sat = nn.ModuleList(nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model*ratio, kernel_size=self.k_size(self.d_model), padding=(self.k_size(self.d_model) - 1) // 2),
            nn.BatchNorm1d(self.d_model*ratio),
            nn.Conv1d(self.d_model*ratio, self.num_channles[i], kernel_size=self.k_size(self.d_model*ratio), padding=(self.k_size(self.d_model*ratio) - 1) // 2),
            nn.GELU(),
            nn.BatchNorm1d(self.num_channles[i])
        ) for i in range(len(self.num_channles)))

        proj_gl_grd = nn.ModuleList(nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model * ratio, kernel_size=self.k_size(self.d_model), padding=(self.k_size(self.d_model) - 1) // 2),
            nn.BatchNorm1d(self.d_model*ratio),
            # nn.Dropout(p=0.5)
            nn.Conv1d(self.d_model*ratio, self.num_channles[i], kernel_size=self.k_size(self.d_model*ratio), padding=(self.k_size(self.d_model*ratio) - 1) // 2),
            nn.GELU(),
            nn.BatchNorm1d(self.num_channles[i])
        ) for i in range(len(self.num_channles)))

        proj_gl_sat.apply(weights_init_kaiming)
        proj_gl_grd.apply(weights_init_kaiming)

        self.proj_gl_sat = proj_gl_sat
        self.proj_gl_grd = proj_gl_grd
        ch_sat = [nn.Conv2d(self.num_channles[i], self.num_channles[i], kernel_size=1) for i in range(len(self.H_sat))]
        ch_grd = [nn.Conv2d(self.num_channles[i], self.num_channles[i], kernel_size=1) for i in range(len(self.H_grd))]
        self.ch_sat = nn.Sequential(*ch_sat)
        self.ch_grd = nn.Sequential(*ch_grd)

        sat_k = [9, 7, 5, 3]
        grd_k = [(7, 13), (5, 11), (3, 9), (1, 7)]
        pad = [(3, 6), (2, 5), (1, 4), (0, 3)]
        sp_sat = [nn.Conv2d(1, 1, kernel_size=sat_k[i], padding=(sat_k[i] - 1) // 2)  for i in range(len(self.num_channles))]
        sp_grd = [nn.Conv2d(1, 1, kernel_size=(7, 13), padding=(3, 6)) for i in range(len(self.num_channles))]
        
        if self.sample:
            sp_sat[0] = nn.Conv2d(1, 1, kernel_size=(sat_k[0]*sat_k[0], 1), padding=((sat_k[0]*sat_k[0] - 1) // 2, 0))
            sp_grd[0] = nn.Conv2d(1, 1, kernel_size=(7*13, 1), padding=((7*13 - 1) // 2, 0))

        self.sp_sat = nn.Sequential(*sp_sat)
        self.sp_grd = nn.Sequential(*sp_grd)
        # print(self.sp_sat, self.sp_grd)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        out_dim_l = 256
        self.proj_local_sat = nn.Linear(sum(self.num_channles), out_dim_l)
        self.proj_local_grd = nn.Linear(sum(self.num_channles), out_dim_l)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def get_config(self,):
        data_config = self.backbone.data_config
        return data_config


    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)
    
    def k_size(self, in_dim):
        t = int(abs((math.log(in_dim, 2) + 1) / 2))
        k_size = t if t % 2 else t + 1

        return k_size


    
    # @autocast()    
    def forward(self, img1, img2=None, input_id=0):
        if img2 is not None:
            grd_b = img1.shape[0]
            sat_b = img2.shape[0]
            sat_x = self.backbone(img2)
            grd_x = self.backbone(img1)

            sat_e, sat_src = self.embed(sat_x)
            grd_e, grd_src = self.embed(grd_x)

            # global
            sat_embed = [x.flatten(2).transpose(1, 2) for x in sat_e]
            grd_embed = [x.flatten(2).transpose(1, 2) for x in grd_e]

            # get low / high-level feature
            if self.sample:
                L_sat_embed = self.sample_L_sat(sat_e[0])
                L_sat_embed = self.norm1(L_sat_embed.flatten(2).transpose(1, 2))
                L_grd_embed = self.sample_L_grd(grd_e[0])
                L_grd_embed = self.norm2(L_grd_embed.flatten(2).transpose(1, 2))
                sat_x[0] = self.sample_L_sat(sat_x[0])
                grd_x[0] = self.sample_L_grd(grd_x[0])
            else:
                L_sat_embed = sat_embed[0]
                L_grd_embed = grd_embed[0]

            H_sat_embed = torch.cat(sat_embed[1:], 1)
            H_grd_embed = torch.cat(grd_embed[1:], 1)
            
            # grd <-> sat multi-scale cross attention
            sat_H, sat_L = self.transformer_sat_H(H_sat_embed, L_sat_embed)
            sat_L, sat_H = self.transformer_sat_L(sat_L, sat_H)

            grd_H, grd_L = self.transformer_grd_H(H_grd_embed, L_grd_embed)
            grd_L, grd_H = self.transformer_grd_L(grd_L, grd_H)

            sat = torch.cat([sat_L, sat_H], dim=1) # (B, L_sat, d_model)
            grd = torch.cat([grd_L, grd_H], dim=1) # (B, L_grd, d_model)

            sat_global = self.proj_sat(sat.transpose(1, 2)).contiguous().view(sat_b,-1)
            grd_global = self.proj_grd(grd.transpose(1, 2)).contiguous().view(grd_b,-1)


            # local
            sat_h1, sat_h2, sat_h3 = self._reshape_feat(sat_H, self.H_sat[1:], self.W_sat[1:])
            grd_h1, grd_h2, grd_h3 = self._reshape_feat(grd_H, self.H_grd[1:], self.W_grd[1:])

            sat_x.append(sat_src[-1])
            grd_x.append(grd_src[-1])

            sat_local = self._gpab(sat_x, [sat_L, sat_h1, sat_h2, sat_h3], proj=self.proj_gl_sat, ch_att=self.ch_sat, sp_att=self.sp_sat, h=self.H_sat, w=self.W_sat)
            grd_local = self._gpab(grd_x, [grd_L, grd_h1, grd_h2, grd_h3], proj=self.proj_gl_grd, ch_att=self.ch_grd, sp_att=self.sp_grd, h=self.H_grd, w=self.W_grd)

            sat_local = self.proj_local_sat(sat_local)
            grd_local = self.proj_local_grd(grd_local)

            desc_sat = torch.cat([sat_global, sat_local], dim=-1)
            desc_grd = torch.cat([grd_global, grd_local], dim=-1)
            
            desc_sat = F.normalize(desc_sat.contiguous(), p=2, dim=1)
            desc_grd = F.normalize(desc_grd.contiguous(), p=2, dim=1)
            
            return desc_sat.contiguous(), desc_grd.contiguous()      
              
        else:
            if input_id == 2:
                b = img1.shape[0]
                sat_x = self.backbone(img1)

                sat_e, sat_src = self.embed(sat_x)
                # global
                sat_embed = [x.flatten(2).transpose(1, 2) for x in sat_e]


                # get low / high-level feature
                if self.sample:
                    L_sat_embed = self.sample_L_sat(sat_e[0])
                    L_sat_embed = self.norm1(L_sat_embed.flatten(2).transpose(1, 2))
                    sat_x[0] = self.sample_L_sat(sat_x[0])
                else:
                    L_sat_embed = sat_embed[0]
                H_sat_embed = torch.cat(sat_embed[1:], 1)
                
                # grd <-> sat multi-scale cross attention
                sat_H, sat_L = self.transformer_sat_H(H_sat_embed, L_sat_embed)
                sat_L, sat_H = self.transformer_sat_L(sat_L, sat_H)
                sat = torch.cat([sat_L, sat_H], dim=1) # (B, L_sat, d_model)
                sat_global = self.proj_sat(sat.transpose(1, 2)).contiguous().view(b,-1)


                # local
                sat_h1, sat_h2, sat_h3 = self._reshape_feat(sat_H, self.H_sat[1:], self.W_sat[1:])
                sat_x.append(sat_src[-1])
                sat_local = self._gpab(sat_x, [sat_L, sat_h1, sat_h2, sat_h3], proj=self.proj_gl_sat, ch_att=self.ch_sat, sp_att=self.sp_sat, h=self.H_sat, w=self.W_sat)
                sat_local = self.proj_local_sat(sat_local)

                desc_sat = torch.cat([sat_global, sat_local], dim=-1)   
                desc_sat = F.normalize(desc_sat.contiguous(), p=2, dim=1)

                return desc_sat
                
            elif input_id == 1:
                b = img1.shape[0]
                grd_x = self.backbone(img1)
                grd_e, grd_src = self.embed(grd_x)

                # global
                grd_embed = [x.flatten(2).transpose(1, 2) for x in grd_e]

                # get low / high-level feature
                if self.sample:
                    L_grd_embed = self.sample_L_grd(grd_e[0])
                    L_grd_embed = self.norm2(L_grd_embed.flatten(2).transpose(1, 2))
                    grd_x[0] = self.sample_L_grd(grd_x[0])
                else:
                    L_grd_embed = grd_embed[0]

                H_grd_embed = torch.cat(grd_embed[1:], 1)
                
                # grd <-> sat multi-scale cross attention
                grd_H, grd_L = self.transformer_grd_H(H_grd_embed, L_grd_embed)
                grd_L, grd_H = self.transformer_grd_L(grd_L, grd_H)
                grd = torch.cat([grd_L, grd_H], dim=1) # (B, L_grd, d_model)
                grd_global = self.proj_grd(grd.transpose(1, 2)).contiguous().view(b,-1)


                # local
                grd_h1, grd_h2, grd_h3 = self._reshape_feat(grd_H, self.H_grd[1:], self.W_grd[1:])
                grd_x.append(grd_src[-1])
                grd_local = self._gpab(grd_x, [grd_L, grd_h1, grd_h2, grd_h3], proj=self.proj_gl_grd, ch_att=self.ch_grd, sp_att=self.sp_grd, h=self.H_grd, w=self.W_grd)
                grd_local = self.proj_local_grd(grd_local)

                desc_grd = torch.cat([grd_global, grd_local], dim=-1)
                desc_grd = F.normalize(desc_grd.contiguous(), p=2, dim=1)

                return desc_grd
            
            else:
                print('set correct "input_id" (1 or 2) when each branch is evaluated independently')
                raise ValueError


    def _reshape_feat(self, feat_H, H, W):
        p1 = H[0] * W[0]
        p2 = H[-1] * W[-1]
        feat_h1 = feat_H[:, :p1, :].contiguous()
        feat_h2 = feat_H[:, p1:-p2, :].contiguous()
        feat_h3 = feat_H[:, -p2:, :].contiguous()

        return [feat_h1, feat_h2, feat_h3]
    

    def _gpab(self, local_feats, global_feats, proj, ch_att, sp_att, h, w):
        geo_att = []
        for i, feat in enumerate(local_feats):
            global_feat = proj[i](global_feats[i].transpose(1, 2))
            b, c, _ = global_feat.shape
            if self.sample and i == 0:
                feat = feat.unsqueeze(-1)
                global_feat = global_feat.unsqueeze(-1)
            else:
                global_feat = global_feat.reshape(b, c, h[i], w[i])

            # channels attrion
            avg_out = self.avg_pool(global_feat)
            att_ch = ch_att[i](avg_out)

            # spatial attrion
            max_out, _ = torch.max(global_feat, dim=1, keepdim=True)
            att_sp = sp_att[i](max_out)
            m = feat * self.sigmoid(att_ch) * self.sigmoid(att_sp)
            m = feat + m
            m = self.avg_pool(m)

            geo_att.append(m.view(b, -1))
        
        results = torch.cat(geo_att, dim=-1).contiguous()
        return results


    def _dim(self, model_name, strides, img_size=[122, 671]):
        if 'convnext' in model_name.lower():
                H = [math.floor(img_size[0] / r) for r in strides]
                W = [math.floor(img_size[1] / r) for r in strides]
                feat_dim = [H[i] * W[i] for i in range(len(H))]
        elif 'resnet' in model_name.lower():
                H = [math.ceil(img_size[0] / r) for r in strides]
                W = [math.ceil(img_size[1] / r) for r in strides]
                feat_dim = [H[i] * W[i] for i in range(len(H))]
        H.append(math.ceil(H[-1] / 2))
        W.append(math.ceil(W[-1] / 2))
        feat_dim.append(H[-1] * W[-1])
        return feat_dim, H, W