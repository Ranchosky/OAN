import torch
import torch.nn as nn 
import numpy as np

from mmdet.models.builder import OAN, build_loss
import torch.nn.functional as F

import logging
from mmcv.runner import load_checkpoint

@OAN.register_module()
class standard_OAN(nn.Module):
    def __init__(self,
                 oan_use_layer = 3,
                 grid_num = 16, 
                 oan_fc_hidden = 512,
                 act_func = 'GELU',
                 loss_oan_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=8.0),
                 generate_dy_thr = False,
                 dy_iterations = 2000,
                 filter_thr = 0.18,
                 pretrained=None):
        super(standard_OAN,self).__init__()
        self.oan_use_layer = oan_use_layer 

        self.loss_oan_cls = build_loss(loss_oan_cls)  
        self.grid_num = grid_num
        self.grid_size = int(1024//grid_num)
        oan_in_features = 256
        self.oan_hidden_features = oan_fc_hidden
        oan_out_features = 1
        if act_func=='GELU':
            act_func= nn.GELU
        if act_func=='ReLU':
            act_func = nn.ReLU
        oan_act_layer=act_func
        oan_drop=0.
        self.fc1 = nn.Linear(oan_in_features, self.oan_hidden_features)
        self.act = oan_act_layer()
        self.fc2 = nn.Linear(self.oan_hidden_features, oan_out_features)
        
        self.drop = nn.Dropout(oan_drop)

        self.oan_conv = nn.Conv2d(2048, 256, 3, stride=2, padding=1)
        self.filter_num = 0

        self.ada_list = []
        self.ada_sum = 0
        self.ada_mean_list = []
        self.ada_mean = 0
        self.generate_dy_thr = generate_dy_thr
        self.dy_iterations = dy_iterations
        self.filter_thr = filter_thr

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):

        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # use default initializer or customized initializer in subclasses
            pass
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')
    
    def grid_partition(self, x, grid_size):
        x = x.permute(0,2,3,1).contiguous()
        B, H, W, C= x.shape 
        x = x.view(B, H//grid_size, grid_size, W//grid_size, grid_size, C)
        grids = x.permute(0,1,3,2,4,5).contiguous().view(-1, grid_size, grid_size, C)
        return grids
    
    def get_gt(self, img,gt_bboxes,grid_size=64):

        B, _, H, W = img.shape 
        assert H==1024 and W==1024
        gt_list = img.new_ones(B, H//grid_size, W//grid_size) 

        for i, gt_bbox in enumerate(gt_bboxes):
            cx = (gt_bbox[:, 0] + gt_bbox[:, 2])/2
            cy = (gt_bbox[:, 1] + gt_bbox[:, 3])/2   
            gt_list[i, (cy//grid_size).long(), (cx//grid_size).long()] = 0 

        gt_list = gt_list.view(-1)

        return gt_list

    def Statistical_Threshold(self,x,last_iteration_num):
        x_ada = x.view(-1)
        x_ada = F.sigmoid(x_ada)
        x_ada_max = x_ada.max().item()
        x_ada_var = x_ada.var().item()
        max_var = x_ada_max+x_ada_var
        self.ada_list.append(max_var)
        self.ada_sum += max_var

        x_ada_mean = x_ada.mean().item()
        self.ada_mean_list.append(x_ada_mean)
        self.ada_mean += x_ada_mean

        if len(self.ada_list)>last_iteration_num:
            self.ada_sum-=self.ada_list[0]
            del self.ada_list[0]
            # print('var_max:', self.ada_sum/last_iteration_num)
            self.ada_mean-=self.ada_mean_list[0]
            del self.ada_mean_list[0]

    def forward_train(self,x,img,gt_bboxes):
        target_label = self.get_gt(img,gt_bboxes,grid_size=self.grid_size)
        target_label = target_label.long()
        use_layer = self.oan_use_layer
        x = x[use_layer]

        x = self.oan_conv(x)
        _, _, H, W = x.shape
        x = self.grid_partition(x, grid_size=W//self.grid_num)

        all_num_grid = x.size(0) 
        x = x.view(all_num_grid, -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        if self.generate_dy_thr:
            self.Statistical_Threshold(x,self.dy_iterations)

        oan_loss = self.loss_oan_cls(x, target_label)  

        return oan_loss

    def simple_test(self, x):
        use_layer = self.oan_use_layer
        x = x[use_layer]

        x = self.oan_conv(x)
        _, _, H, W = x.shape
        x = self.grid_partition(x, grid_size=W//self.grid_num)

        all_num_grid = x.size(0) 
        x = x.view(all_num_grid, -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        pred_label = F.sigmoid(x)
        value, _ = torch.max(pred_label,dim=0)
        if value <= self.filter_thr:   
            return True
        return False
            



