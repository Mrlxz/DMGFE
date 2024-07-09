#!/usr/bin/env python3
# coding: utf-8
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import VanillaNet
import random
from RepVGGs2 import repvgg
from RepVGGml import repvgg
from RepVGGlxz import repvgg
#from RepVGGlxz_new import repvgg
from RepVGG import repvgg
#from RepVGGlxz_B3 import repvgg
# from repvgg_case import repvgg
# from pycinversnet_DDGL_se_norelu3 import ddgl
# from DAMDNet import DAMDNet_v1
from PIL import Image
from matplotlib import pyplot as plt



from Diffusion import GaussianDiffusionSampler

from Mode import UNet
from mobilenet_v1 import mobilenet_1
# from Repvgg_ca_at_shuffle3 import repvgg
# from Repvgg_b1g4_ca_at_shuffle3 import repvgg
# from Repvgg_ca_at_shuffle3 import repvgg
# from HFCAN.HCANet import repvgg
# from Repvgg_ca_noat_shuffleself3 import repvgg
from VanillaNet import vanillanet_10
import mobilenet_v1
import time
import numpy as np
import torch.nn.functional as F
from benchmark_aflw2000 import calc_nme as calc_nme_alfw2000
from benchmark_aflw2000 import ana as ana_alfw2000
from benchmark_aflw import calc_nme as calc_nme_alfw
from benchmark_aflw import ana as ana_aflw

from utils.ddfa import ToTensorGjz, NormalizeGjz, DDFATestDataset, reconstruct_vertex
import argparse

from AttentionTransformer import AttentionTransformer as AT
from CRIS import CrissCrossAttention
# arch_choices=['VanillaNet','vanillanet_13']

def jigsaw_generator(inputs, n):
    l = []
    patch_num_list = []
    # local_confidence_list = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 120 // n
    rounds = n ** 2
    jigsaws = inputs.clone()

    for i in range(inputs.size(0)):
        patch_num = random.randint(0, 3)
        patch_num_list.append(patch_num)
        # local_confidence_list.append(patch_num / rounds)

    for j in range(inputs.size(0)):
        random.shuffle(l)
        for i in range(patch_num_list[j]):
            x, y = l[i]
            temp = jigsaws[j, :, 0:block_size, 0:block_size].clone()
            jigsaws[j, :, 0:block_size, 0:block_size] = jigsaws[j, :, x * block_size:(x + 1) * block_size,
                                                        y * block_size:(y + 1) * block_size].clone()
            jigsaws[j, :, x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


# def jigsaw_generator(inputs, n):
#     l = []
#     for a in range(n):
#         for b in range(n):
#             l.append([a, b])
#     block_size = 120 // n
#     rounds = n ** 2
#     random.shuffle(l)
#     jigsaws = inputs.clone()
#     for i in range(rounds):
#         x, y = l[i]
#         temp = jigsaws[..., 0:block_size, 0:block_size].clone()
#         # print(temp.size())
#         jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
#                                                    y * block_size:(y + 1) * block_size].clone()
#         jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp
#
#     return jigsaws


def visualize(x):
    x = x.detach().cpu().numpy()
    x = np.transpose(x, (0, 2, 3, 1))
    # print(x.shape)
    n = x.shape[0]
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=(4 * n, 4))
    for k in range(n):
        axs[k].imshow(x[k], cmap=None)
        axs[k].axis('off')

    plt.show()


def visualize1(x):
    x = x.view(2, 31, 2).detach().cpu().numpy()
    # x = np.transpose(x, (0, 1, 3, 2))
    n = x.shape[0]
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=(4 * n, 4))
    for k in range(n):
        axs[k].imshow(x[k], cmap=None)
        axs[k].axis('off')

    plt.show()


def extract_param(checkpoint_fp, root='', filelists=None, arch='vanillanet', num_classes=62, device_ids=[0],
                  batch_size=128, num_workers=4):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
    torch.cuda.set_device(device_ids[0])
    # model = getattr(HCANet, arch)()
    model = repvgg()
    # model = getattr(VanillaNet, arch)(num_classes=num_classes)
    # model=vanillanet_10()
    # model = getattr(repvgg, arch)(num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint,False)
    dataset = DDFATestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    cudnn.benchmark = True
    Sampler = GaussianDiffusionSampler(UNet(
         T=2, ch=32, ch_mult=[1, 2], attn=[1],
         num_res_blocks=1, dropout=0.1), 0.0001, 0.02, 2)
    Sampler = nn.DataParallel(Sampler, device_ids=device_ids).cuda()
    model.eval()

    end = time.time()
    outputs = []
    with torch.no_grad():
        for _, inputs in enumerate(data_loader):
            #inputs = F.interpolate(inputs, size=(119, 119))
            inputs = inputs.cuda()
            #inputs=inputs*3
            #inputs=inputs*1.3
            # input1 = jigsaw_generator(inputs, 3)
            # inputs = torch.cat((inputs, input1), dim=0)
            # visualize(inputs)
            # inputs = at(inputs)
            # visualize(inputs)

            # inputs = F.interpolate(inputs, size=(32, 32))
            # # inputs = Sampler(inputs)
            # # # visualize(inputs)
            # inputs = F.interpolate(inputs, size=(120, 120))


            output = model(inputs)
            # output = torch.cat((out1, out2), dim=0)
            # print(output.shape)

            # t = model.avgpool(features)
            # t = t.reshape(1, -1)
            # output = model.classifier(t)[0]
            # pred = torch.argmax(output).item()
            # pred_class = output[pred]
            #
            # pred_class.backward()

            # features = output[0]
            #
            # features = features.reshape(features.shape[0], -1)
            #
            # # avg_grads = torch.mean(grads[0], dim=(1, 2))
            # # avg_grads = avg_grads.expand(features.shape[1], features.shape[2], features.shape[0]).permute(2, 0, 1)
            # # features *= avg_grads
            #
            # heatmap = features.detach().cpu().numpy()
            # heatmap = np.mean(heatmap, axis=0)
            #
            # heatmap = np.maximum(heatmap, 0)
            # heatmap /= (np.max(heatmap) + 1e-8)
            # img_path = '/home/maxu/Data/test.data/AFLW2000-3D_crop/image00023.jpg'
            # # img = cv2.imread(img_path)
            # # img = Image.open(img_path).convert('RGB')
            # # transforms2 = torchvision.transforms.Compose([
            # #     torchvision.transforms.ToTensor(),
            # #     torchvision.transforms.Normalize([0.5, ], [0.5, ])])
            # # # data = transforms2(img).unsqueeze(0)
            #
            # # img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
            # # width = 500
            # # height = (img.size[1] * width / img.size[0])
            # img = cv2.imread(img_path)
            # img = cv2.resize(img, (features.shape[1], features.shape[0]))
            # heatmap = cv2.resize(heatmap, (features.shape[1], features.shape[0]))
            # # heatmap = cv2.resize(heatmap, (width, height))
            # heatmap = np.uint8(255 * heatmap)
            # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
            # cv2.imshow('1', superimposed_img)
            # cv2.waitKey(0)

            # print(output.shape)
            # visualize1(output)

            for i in range(output.shape[0]):
                param_prediction = output[i].cpu().numpy().flatten()

                outputs.append(param_prediction)
        outputs = np.array(outputs, dtype=np.float32)

    print(f'Extracting params take {time.time() - end: .3f}s')
    return outputs


def _benchmark_aflw(outputs):
    return ana_aflw(calc_nme_alfw(outputs))


def _benchmark_aflw2000(outputs):
    return ana_alfw2000(calc_nme_alfw2000(outputs))


def benchmark_alfw_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw(outputs)


def benchmark_aflw2000_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw2000(outputs)


def benchmark_pipeline(arch, checkpoint_fp):
    device_ids = [0]

    def aflw():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='/home/datatom/Data/test.data/AFLW_GT_crop',
            filelists='/home/datatom/Data/test.data/AFLW_GT_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=256)

        benchmark_alfw_params(params)

    def aflw2000():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='/home/datatom/Data/test.data/AFLW2000-3D_crop',
            filelists='/home/datatom/Data/test.data/AFLW2000-3D_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=256)

        benchmark_aflw2000_params(params)

    aflw2000()
    #aflw()


def main():
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    parser.add_argument('--arch', default='vanillanet_13', type=str)
    # parser.add_argument('-c', '--checkpoint-fp', default='models/phase1_wpdc_vdc.pth.tar', type=str)/home/maxu/lxz/
    parser.add_argument('-c', '--checkpoint-fp',
                        default='/home/datatom/Data1/ml/HFCAN/training/snapshot/phase_batchsize_128_wpdc_checkpoint_epoch_84.pth.tar',
                        type=str)
    # parser.add_argument('-c', '--checkpoint-fp', default='/home/maxu/lxz/HFCAN/models/157_3.52_phase1_wpdc_checkpoint_epoch_49.pth.tar', type=str)
    args = parser.parse_args()

    benchmark_pipeline(args.arch, args.checkpoint_fp)


if __name__ == '__main__':
    main()

# def jigsaw_generator(inputs, n):
#     l = []
#     patch_num_list = []
#     # local_confidence_list = []
#     for a in range(n):
#         for b in range(n):
#             l.append([a, b])
#     block_size = 120 // n
#     rounds = n ** 2
#     jigsaws = inputs.clone()
#
#     for i in range(inputs.size(0)):
#         patch_num = random.randint(0, 3)
#         patch_num_list.append(patch_num)
#         # local_confidence_list.append(patch_num / rounds)
#
#     for j in range(inputs.size(0)):
#         random.shuffle(l)
#         for i in range(patch_num_list[j]):
#             x, y = l[i]
#             temp = jigsaws[j, :, 0:block_size, 0:block_size].clone()
#             jigsaws[j, :, 0:block_size, 0:block_size] = jigsaws[j, :, x * block_size:(x + 1) * block_size,
#                                                         y * block_size:(y + 1) * block_size].clone()
#             jigsaws[j, :, x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp
#
#     return jigsaws
#
# def extract_param(checkpoint_fp, root='', filelists=None, arch='repvgg', num_classes=62, device_ids=[0],
#                   batch_size=128, num_workers=4):
#     map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}  # 锟斤拷锟斤拷锟斤拷锟接筹拷锟芥储位锟斤拷
#     # map_location = torch.device('cpu')
#     checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
#     torch.cuda.set_device(device_ids[0])
#
#     model = repvgg()
#     model = nn.DataParallel(model, device_ids=device_ids).cuda()
#     # model = nn.DataParallel(model, device_ids=device_ids)
#     model.load_state_dict(checkpoint)
#
#     dataset = DDFATestDataset(filelists=filelists, root=root,
#                               transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
#     data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
#
#     cudnn.benchmark = True  # 锟斤拷锟斤拷锟斤拷锟?flag 锟斤拷锟斤拷锟斤拷锟斤拷锟矫碉拷 cuDNN 锟斤拷 auto-tuner 锟皆讹拷寻锟斤拷锟斤拷锟绞合碉拷前锟斤拷锟矫的革拷效锟姐法锟斤拷锟斤拷锟斤到锟脚伙拷锟斤拷锟斤拷效锟绞碉拷锟斤拷锟解。
#     model.eval()
#
#     end = time.time()
#     outputs = []
#     with torch.no_grad():
#         for _, (inputs, inputs2) in enumerate(data_loader):
#             # inputs = inputs.cuda()
#             # inputs2 = jigsaw_generator(inputs2, 4)
#             output = model(inputs)
#             # input1 = jigsaw_generator(inputs, 3)
#
#             # output,_,_= model(inputs,input1)
#
#             for i in range(output.shape[0]):
#                 param_prediction = output[i].cpu().numpy().flatten()
#
#                 outputs.append(param_prediction)
#         outputs = np.array(outputs, dtype=np.float32)
#
#     print(f'Extracting params take {time.time() - end: .3f}s')
#     return outputs
#
#
# def _benchmark_aflw(outputs):  # x锟斤拷y
#     return ana_aflw(calc_nme_alfw(outputs))
#
#
# def _benchmark_aflw2000(outputs):
#     return ana_alfw2000(calc_nme_alfw2000(outputs))
#
#
# def benchmark_alfw_params(params):
#     outputs = []
#     for i in range(params.shape[0]):
#         lm = reconstruct_vertex(params[i])
#         outputs.append(lm[:2, :])
#     return _benchmark_aflw(outputs)
#
#
# def benchmark_aflw2000_params(params):
#     outputs = []
#     for i in range(params.shape[0]):
#         lm = reconstruct_vertex(params[i])
#         outputs.append(lm[:2, :])
#     return _benchmark_aflw2000(outputs)
#
#
# def benchmark_pipeline(arch, checkpoint_fp):
#     device_ids = [0]
#
#     def aflw():
#         params = extract_param(
#             checkpoint_fp=checkpoint_fp,
#             root='/home/data1/spwu_data/wk/Data/test.data/AFLW_GT_crop',
#             filelists='/home/data1/spwu_data/wk/Data/test.data/AFLW_GT_crop.list',
#             arch=arch,
#             device_ids=device_ids,
#             batch_size=128)
#         benchmark_alfw_params(params)
#
#     def aflw2000():
#         params = extract_param(
#             checkpoint_fp=checkpoint_fp,
#             root='/home/data1/spwu_data/wk/Data/test.data/AFLW2000-3D_crop',
#             filelists='/home/data1/spwu_data/wk/Data/test.data/AFLW2000-3D_crop.list',
#             arch=arch,
#             device_ids=device_ids,
#             batch_size=128)
#
#         benchmark_aflw2000_params(params)
#
#     aflw2000()
#     aflw()
#
#
# #  E:/FaeDataSet/3DDFA/phase1_wpdc_checkpoint_epoch_46.pth.tar
# #  ./models/phase1_wpdc_vdc.pth.tar
# # 45
# def main():
#     parser = argparse.ArgumentParser(description='3DDFA Benchmark')
#     parser.add_argument('--arch', default='', type=str)
#     parser.add_argument('-c', '--checkpoint-fp',
#                         default='/home/maxu/lxz/HFCAN/training/snapshot/phase1_wpdc_checkpoint_epoch_1.pth.tar',
#                         type=str)
#     args = parser.parse_args()
#
#     benchmark_pipeline(args.arch, args.checkpoint_fp)
#
#
# if __name__ == '__main__':
#     main()

# !/usr/bin/env python3
# coding: utf-8

# import torch
# import torch.nn as nn
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import torch.backends.cudnn as cudnn
#
# import VanillaNet
# import mobilenet_v1
# import respace
# # import mymynobilenet
# import RepVGG
# import time
# import numpy as np
#
# from benchmark_aflw2000 import calc_nme as calc_nme_alfw2000
# from benchmark_aflw2000 import ana as ana_alfw2000
# from benchmark_aflw import calc_nme as calc_nme_alfw
# from benchmark_aflw import ana as ana_aflw
#
# from utils.ddfa import ToTensorGjz, NormalizeGjz, DDFATestDataset, reconstruct_vertex
# import argparse
#
#
# # def extract_param(checkpoint_fp, root='', filelists=None, arch='mobilenet_1', num_classes=62, device_ids=[0],
# #                   batch_size=128, num_workers=4):
# def extract_param(checkpoint_fp, root='', filelists=None, arch='repvgg', num_classes=62, device_ids=[0],
#                   batch_size=128, num_workers=4):
#     map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
#     checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
#     torch.cuda.set_device(device_ids[0])
#     model = getattr(RepVGG, arch)()
#     # model = getattr(mymobilenet, arch)(num_classes=num_classes)
#     model = nn.DataParallel(model, device_ids=device_ids).cuda()
#     # model.load_state_dict(checkpoint)
#     model.load_state_dict(checkpoint, False)
#     dataset = DDFATestDataset(filelists=filelists, root=root,
#                               transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
#     data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
#
#     cudnn.benchmark = True
#     model.eval()
#
#     end = time.time()
#     outputs = []
#     with torch.no_grad():
#         for _, inputs in enumerate(data_loader):
#             inputs = inputs.cuda()
#             output = model(inputs)
#
#             for i in range(output.shape[0]):
#                 param_prediction = output[i].cpu().numpy().flatten()
#
#                 outputs.append(param_prediction)
#         outputs = np.array(outputs, dtype=np.float32)
#
#     print(f'Extracting params take {time.time() - end: .3f}s')
#     return outputs
#
#
# def _benchmark_aflw(outputs):
#     return ana_aflw(calc_nme_alfw(outputs))
#
#
# def _benchmark_aflw2000(outputs):
#     return ana_alfw2000(calc_nme_alfw2000(outputs))
#
#
# def benchmark_alfw_params(params):
#     outputs = []
#     for i in range(params.shape[0]):
#         lm = reconstruct_vertex(params[i])
#         outputs.append(lm[:2, :])
#     return _benchmark_aflw(outputs)
#
#
# def benchmark_aflw2000_params(params):
#     outputs = []
#     for i in range(params.shape[0]):
#         lm = reconstruct_vertex(params[i])
#         outputs.append(lm[:2, :])
#     return _benchmark_aflw2000(outputs)
#
#
# def benchmark_pipeline(arch, checkpoint_fp):
#     device_ids = [0]
#
#     def aflw():
#         params = extract_param(
#             checkpoint_fp=checkpoint_fp,
#             root='/home/maxu/Data/test.data/AFLW_GT_crop',
#             filelists='/home/maxu/Data/test.data/AFLW_GT_crop.list',
#             arch=arch,
#             device_ids=device_ids,
#             batch_size=128)
#
#         benchmark_alfw_params(params)
#
#     def aflw2000():
#         params = extract_param(
#             checkpoint_fp=checkpoint_fp,
#             root='/home/maxu/Data/test.data/AFLW2000-3D_crop',
#             filelists='/home/maxu/Data/test.data/AFLW2000-3D_crop.list',
#             arch=arch,
#             device_ids=device_ids,
#             batch_size=128)
#
#         benchmark_aflw2000_params(params)
#
#     aflw2000()
#     aflw()
#
#
# def main():
#     parser = argparse.ArgumentParser(description='3DDFA Benchmark')
#     parser.add_argument('--arch', default='repvgg', type=str)
#     # parser.add_argument('-c', '--checkpoint-fp', default='models/phase1_wpdc_vdc.pth.tar', type=str)/home/maxu/lxz/
#     parser.add_argument('-c', '--checkpoint-fp',
#                         default='/home/maxu/lxz/HFCAN/training/snapshot/phase1_wpdc_checkpoint_epoch_1.pth.tar',
#                         type=str)
#     # parser.add_argument('-c', '--checkpoint-fp', default='models/phase1_wpdc_vdc.pth.tar', type=str)
#     args = parser.parse_args()
#
#     benchmark_pipeline(args.arch, args.checkpoint_fp)
#
#
# if __name__ == '__main__':
#     main()
