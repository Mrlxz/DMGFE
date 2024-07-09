# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 鍘嬬缉H,W涓?
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d姣擫inear鏂逛究鎿嶄綔
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True鐩存帴鏇挎崲锛岃妭鐪佸唴瀛?            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out:',max_out.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out:',avg_out.shape)
        a = torch.cat([max_out, avg_out], dim=1)
        # print('a:',a.shape)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # print('spatial:',spatial_out.shape)
        x = spatial_out * x
        # print('x:',x.shape)
        return x


class ASPP_CBAM(nn.Module):
    def __init__(self, in_channel=512, depth=256, rate=1, bn_mom=0.1):
        dim_in = in_channel
        dim_out = depth
        super(ASPP_CBAM, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # print('dim_in:',dim_in)
        # print('dim_out:',dim_out)
        self.cbam = CBAMLayer(channel=dim_out * 5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)

        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # print('feature:',feature_cat.shape)
        # 鍔犲叆cbam娉ㄦ剰鍔涙満鍒?        cbamaspp = self.cbam(feature_cat)
        result = self.conv_cat(cbamaspp)

        return result


# class ASPP(nn.Module):
#     def __init__(self, in_channel=512, depth=256):
#         super(ASPP, self).__init__()
#         self.mean = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv = nn.Conv2d(in_channel, depth, 1, 1)
#
#         self.atrous_block1 = nn.Conv2d(in_channel, depth, kernel_size=1, stride=1)
#         self.atrous_block6 = nn.Conv2d(in_channel, depth, kernel_size=3, stride=1, padding=6, dilation=6)
#         self.atrous_block12 = nn.Conv2d(in_channel, depth, kernel_size=3, stride=1, padding=12, dilation=12)
#         self.atrous_block18 = nn.Conv2d(in_channel, depth, kernel_size=3, stride=1, padding=18, dilation=18)
#
#         self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
#
#     def forward(self, x):
#         size = x.shape[2:]
#
#         image_features = self.mean(x)
#         image_features = self.conv(image_features)
#         image_features = F.upsample(image_features, size=size, mode='bilinear')
#
#         atrous_block1 = self.atrous_block1(x)
#
#         atrous_block6 = self.atrous_block6(x)
#
#         atrous_block12 = self.atrous_block12(x)
#
#         atrous_block18 = self.atrous_block18(x)
#
#         net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
#                                               atrous_block12, atrous_block18], dim=1))
#         return net
