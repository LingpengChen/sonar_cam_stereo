from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from models.submodule import *
from inverse_warp import warp_differentiable

def convtext(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

class PSNet(nn.Module):
    def __init__(self, nlabel=32, mindepth=1, label_factor=1.06, alpha=60):
        super(PSNet, self).__init__()
        self.nlabel = nlabel
        self.label_factor = label_factor
        self.mindepth = mindepth
        self.depths = torch.tensor([mindepth * self.label_factor**i for i in range(nlabel)])
        # 将 alpha 转换为张量
        self.alpha = torch.tensor(alpha, dtype=torch.int)
        
        self.rgb_feature_extraction = feature_extraction()
        self.sonar_feature_extraction = soanr_feature_extraction()
        # self.feature_extraction = feature_extraction()

        self.convs = nn.Sequential(
            convtext(33, 128, 3, 1, 1), # dilation
            convtext(128, 128, 3, 1, 2),
            convtext(128, 128, 3, 1, 4),
            convtext(128, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1)
        )

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def compute_d(self, depth, K_var):
        """
        calculate depth of pixel (i.e. autually depth) from depth of pseudo plane
        
        参数:
        - depth: 深度值，可以是标量或形状为[B,H,W]的张量
        - K_var: 相机内参矩阵，形状为[B,3,3]
        - v_coords: 纵坐标值，形状为[B,H,W]或与depth相同
        - alpha_deg: 角度alpha（以度为单位）
        
        返回:
        - d: 计算的d值，形状与depth相同
        """
        device = depth.device
        dtype = depth.dtype
    
        batch_size, _, height, width = depth.shape

        
        # 从相机内参矩阵中提取focal_y和cy
        focal_y = K_var[:, 1, 1]  # 形状为[B]
        cy = K_var[:, 1, 2]  # 形状为[B]
        
        # 将focal_y和cy调整为合适的形状以进行广播
        focal_y = focal_y.view(batch_size, 1, 1, 1)  # 形状变为[B,1,1]
        cy = cy.view(batch_size, 1, 1, 1)  # 形状变为[B,1,1]
        
        # 计算tan(alpha)
        tan_alpha = torch.tan(torch.deg2rad(self.alpha))
        
        
        # 创建v坐标网格
        v_coords = torch.arange(0, height, device=device, dtype=dtype)
        v_coords = v_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        
        # 计算系数 1/(tan(alpha)*fy)
        coeff = 1.0 / (tan_alpha * focal_y)
        
        # 计算d
        d = depth * (1 - coeff * (v_coords - cy))
        
        return d
    
    def forward(self, rgb_img_var, sonar_rect_img_var, K_var, KT_inv_var, distance_range_var, theta_range_var):  # reference + j * targets

        K_var = K_var.clone()
        K_var[:,:2,:] = K_var[:,:2,:] / 4
        
        KT_inv_var = KT_inv_var.clone()
        KT_inv_var[:,:2,:2] = KT_inv_var[:,:2,:2] * 4

        # rgb_img_feature = self.feature_extraction(rgb_img_var)  # torch.Size([1, 3, 480, 640])  =>  torch.Size([1, 32, 120, 160])
        # sonar_feature   = rgb_img_feature
        rgb_img_feature = self.rgb_feature_extraction(rgb_img_var)  # torch.Size([1, 3, 480, 640])  =>  torch.Size([1, 32, 120, 160])
        sonar_feature   = self.sonar_feature_extraction(sonar_rect_img_var)
                
        # Cost shape
        # Batch   feature_channel   label/layers   width   height
        cost = Variable(torch.FloatTensor(rgb_img_feature.size()[0], rgb_img_feature.size()[1]*2, self.nlabel,  rgb_img_feature.size()[2],  rgb_img_feature.size()[3]).zero_()).cuda()
        for i, depth in enumerate(self.depths):
            warped_sonar_feature = warp_differentiable(K_var, KT_inv_var, rgb_img_feature.size(), sonar_feature, depth, 
                                                        distance_range_var, theta_range_var, self.alpha)
            cost[:, :warped_sonar_feature.size()[1], i, :,:] = rgb_img_feature
            cost[:, warped_sonar_feature.size()[1]:, i, :,:] = warped_sonar_feature

            # cost = torch.Size([1, 2*32, 64, 120, 160]) 
            cost = cost.contiguous()  # 确保张量在内存中是连续存储的
            cost0 = self.dres0(cost)
            cost0 = self.dres1(cost0) + cost0
            cost0 = self.dres2(cost0) + cost0 
            cost0 = self.dres3(cost0) + cost0 
            cost0 = self.dres4(cost0) + cost0
            costs = self.classify(cost0)  # torch.Size([B, 1, 64, 120, 160])

        
        # cost aggregation costss torch.Size([B, 1, 64, 120, 160])
        costss = Variable(torch.FloatTensor(rgb_img_feature.size()[0], 1, self.nlabel,  rgb_img_feature.size()[2],  rgb_img_feature.size()[3]).zero_()).cuda()
        for i in range(self.nlabel):
            costt = costs[:, :, i, :, :]
            costss[:, :, i, :, :] = self.convs(torch.cat([rgb_img_feature, costt],1)) + costt

        costs = F.upsample(costs, [self.nlabel,rgb_img_var.size()[2],rgb_img_var.size()[3]], mode='trilinear')
        costs = torch.squeeze(costs,1)
        pred0 = F.softmax(costs,dim=1)
        # Convert probability distribution to disparity values
        pred0 = disparityregression(self.nlabel)(pred0) # torch.Size([B, 64, 480, 640])
        # depth0 = self.mindepth*self.nlabel/(pred0.unsqueeze(1)+1e-16)
        depth0_pseudo_plane = self.label_factor**(pred0.unsqueeze(1))
        depth0 = self.compute_d(depth0_pseudo_plane, K_var)
        
        costss = F.upsample(costss, [self.nlabel,rgb_img_var.size()[2],rgb_img_var.size()[3]], mode='trilinear')
        costss = torch.squeeze(costss,1)
        pred = F.softmax(costss,dim=1)
        pred = disparityregression(self.nlabel)(pred)
        depth_pseudo_plane = self.label_factor**(pred.unsqueeze(1))
        # depth = self.mindepth*self.nlabel/(pred.unsqueeze(1)+1e-16)
        depth = self.compute_d(depth_pseudo_plane, K_var)

        if self.training:
            return depth0, depth
        else:
            return depth
