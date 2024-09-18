import torch
import timm
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast
from timm.models import convnext


class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        if 'tiny' in model_name:
            bk_checkpoint = 'pretrained/convnext_tiny_22k_1k_224.pth'
        elif 'base' in model_name:
            bk_checkpoint = 'pretrained/convnext_base_22k_1k_224.pth'
        
        if '384' in bk_checkpoint:
            bk_checkpoint = bk_checkpoint.replace('224', '384')
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            # self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) # raise huggingface_hub.utils._errors
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size, pretrained_cfg_overlay=dict(file=bk_checkpoint))
        else:
            # self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, pretrained_cfg_overlay=dict(file=bk_checkpoint))
        
        # for name, module in self.model.named_children():
        #     print(name)
        #     if name == 'stages':
        #         print(module[0])
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # print(self.model)
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    # @autocast()    
    def forward(self, img1, img2=None, input_id=1):
        if img2 is not None:
       
            image_features1 = self.model(img1)     
            image_features2 = self.model(img2)
            
            return image_features1, image_features2            
              
        else:
            image_features = self.model(img1)
              
            return image_features


if __name__ == '__main__':
    from collections import OrderedDict
    from torchstat import stat
    from ptflops import get_model_complexity_info
    model_name = 'convnext_base.fb_in22k_ft_in1k_384'
    # model_name = "convnext_tiny"
    # model_name = 'resnet18'
    pretrained = True
    img_size = 384

    model = TimmModel(model_name, pretrained, img_size)

    # data_config = model.get_config()
    # print(data_config)
    # dict_list = model.ConvNext.group_matcher()
    # print(dict_list)
    # stat(model, (3, 384, 384))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params/1000000)

    x = torch.randn(2, 3, 384, 384)
    y1, y2 = model(x, x)
    # print(model)
    # print(y1.shape, y2.shape)

    macs, params = get_model_complexity_info(model, (3, 140, 768), as_strings=True,
                                       print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

