#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

import os
import sys

import imageio

from RepVGGlxz import repvgg

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""
from glob import glob
import torch
import torchvision.transforms as transforms
import mobilenet_v1
#import respace
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
import RepVGGlxz
STD_SIZE = 120
from FaceBoxes import FaceBoxes
from utils.lighting import RenderPipeline
import imageio
import matplotlib.pyplot as plt
cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr
face_boxes=FaceBoxes()
def process_image(img_fp,model,dlib_landmark_model,face_detector,tri,transform,args):
        img_ori = cv2.imread(img_fp)
        boxes = face_boxes(img_ori)
        n = len(boxes)
        if n == 0:
            print(f'No face detected, exit')
            sys.exit(-1)
        print(f'Detect {n} faces')


        # if args.dlib_bbox:
        #     rects = face_detector(img_ori, 1)
        # else:
        #     rects = []
        #
        # if len(rects) == 0:
        #     rects = dlib.rectangles()
        #     rect_fp = img_fp + '.bbox'
        #     lines = open(rect_fp).read().strip().split('\n')[1:]
        #     for l in lines:
        #         l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
        #         rect = dlib.rectangle(l, r, t, b)
        #         rects.append(rect)

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = get_suffix(img_fp)
        out_folder = args.output if args.output else os.path.dirname(img_fp)
        k = os.path.join(out_folder, os.path.basename(img_fp))
        for rect in boxes:

            print(rect)
            roi_box = parse_roi_box_from_bbox(rect[:4])

            img = crop_img(img_ori, roi_box)

            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)

            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            pts_res.append(pts68)
            P, pose = parse_pose(param)
            Ps.append(P)
            poses.append(pose)
            # out_folder = args.output if args.output else os.path.dirname(img_fp)
            # k = os.path.join(out_folder, os.path.basename(img_fp))
            # dense face 3d vertices
            if args.image_3d or args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or args.dump_obj:
                vertices = predict_dense(param, roi_box)
                vertices_lst.append(vertices)
            if args.dump_ply:
                dump_to_ply(vertices, tri, '{}_{}.ply'.format(k.replace(suffix, ''), ind))
            if args.dump_vertex:
                dump_vertex(vertices, '{}_{}.mat'.format(k.replace(suffix, ''), ind))
            if args.dump_pts:
                wfp = '{}_{}.txt'.format(k.replace(suffix, ''), ind)
                np.savetxt(wfp, pts68, fmt='%.3f')
                print('Save 68 3d landmarks to {}'.format(wfp))
            if args.dump_roi_box:
                wfp = '{}_{}.roibox'.format(k.replace(suffix, ''), ind)
                np.savetxt(wfp, roi_box, fmt='%.3f')
                print('Save roi box to {}'.format(wfp))
            if args.dump_paf:
                wfp_paf = '{}_{}_paf.jpg'.format(k.replace(suffix, ''), ind)
                wfp_crop = '{}_{}_crop.jpg'.format(k.replace(suffix, ''), ind)
                paf_feature = gen_img_paf(img_crop=img, param=param, kernel_size=args.paf_size)

                cv2.imwrite(wfp_paf, paf_feature)
                cv2.imwrite(wfp_crop, img)
                print('Dump to {} and {}'.format(wfp_crop, wfp_paf))
            if args.dump_obj:
                wfp = '{}_{}.obj'.format(k.replace(suffix, ''), ind)
                colors = get_colors(img_ori, vertices)
                write_obj_with_colors(wfp, vertices, tri, colors)
                print('Dump obj with sampled texture to {}'.format(wfp))
            ind += 1
        # k = os.path.join(out_folder, os.path.basename(img_fp))
        if args.image_3d:
                img_render = imageio.imread(k, pilmode="RGB").astype(np.float32) / 255.
                triangles = sio.loadmat('visualize/tri.mat')['tri'].T - 1  # mx3
                triangles = _to_ctype(triangles).astype(np.int32)  # for type compatible
                app = RenderPipeline(**cfg)
                img_render = app(np.ascontiguousarray(vertices_lst[0].T), triangles, img_render)
                imageio.imwrite(k.replace(suffix, '_3d.png'), img_render)
                if len(vertices_lst)==1:
                    pass
                else:
                    for i in range(len(vertices_lst)-1):
                        img_render = imageio.imread(k.replace(suffix, '_3d.png'), pilmode="RGB").astype(np.float32) / 255.
                        app = RenderPipeline(**cfg)
                        img_render = app(np.ascontiguousarray(vertices_lst[i+1].T), triangles, img_render)
                        imageio.imwrite(k.replace(suffix, '_3d.png'), img_render)
        if args.dump_pose:
            # P, pose = parse_pose(param)  # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
            img_pose = plot_pose_box(img_ori, Ps, pts_res)
            # wfp = img_fp.replace(suffix, '_pose.jpg')
            wfp=os.path.join(out_folder,os.path.basename(img_fp).replace(suffix,'_pose.jpg'))
            cv2.imwrite(wfp, img_pose)
            print('Dump to {}'.format(wfp))
        if args.dump_depth:
            wfp = k.replace(suffix, '_depth.png')
            # depths_img = get_depths_image(img_ori, vertices_lst, tri-1)  # python version
            depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1)  # cython version
            cv2.imwrite(wfp, depths_img)
            print('Dump to {}'.format(wfp))
        if args.dump_pncc:
            wfp = k.replace(suffix, '_pncc.png')
            pncc_feature = cpncc(img_ori, vertices_lst, tri - 1)  # cython version
            cv2.imwrite(wfp, pncc_feature[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
            print('Dump to {}'.format(wfp))
        if args.dump_res:
            draw_landmarks(img_ori, pts_res, wfp=k.replace(suffix, '_lxz.jpg'), show_flg=args.show_flg)


def main(args):
    # 1. load pre-tained model

    checkpoint_fp = '/home/datatom/Data1/ml/HFCAN/best model/Diffusionx0.7_63_3.442_4.697_5.497.pth.tar'
    # checkpoint_fp = '/home/datatom/Data1/ml/HFCAN/training/snapshot/phase_batchsize_128_wpdc_checkpoint_epoch_20.pth.tar'
    arch = 'repvgg'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(RepVGGlxz, arch)()  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    image_files = []
    if args.files and os.path.isdir(args.files[0]):
        image_folder = args.files[0]
        image_files = glob(os.path.join(image_folder, '*.jpg')) + glob(os.path.join(image_folder, '*png'))
    elif args.files:
        image_files = args.files
    for img_fp in image_files:
        process_image(img_fp, model, face_regressor, face_detector, tri, transform, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('--output', type=str,help='')

    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='two', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='true', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='true', type=str2bool)
    parser.add_argument('--image_3d', default='true', type=str2bool)
    parser.add_argument('--dump_pts', default='false', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='false', type=str2bool)
    parser.add_argument('--dump_depth', default='false', type=str2bool)
    parser.add_argument('--dump_pncc', default='false', type=str2bool)
    parser.add_argument('--dump_paf', default='false', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='true', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')

    args = parser.parse_args()
    main(args)

    #
    # else:
    #     for img_fp in args.files:
    #         process_image(img_fp, model, dlib_landmark_model, face_detector, tri, transform, args)
