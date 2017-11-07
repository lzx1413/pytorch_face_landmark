from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
import dlib
from models.basenet import BaseNet
parser = argparse.ArgumentParser(description='PyTorch face landmark')
# Datasets
parser.add_argument('-img', '--image', default='face76', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu_id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-c', '--checkpoint', default='checkpoint/0918/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

args = parser.parse_args()
mean = np.asarray([0.4465, 0.4822, 0.4914])
std = np.asarray([0.1994, 0.1994, 0.2023])



def load_model():
    model = BaseNet()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == '__main__':
    out_size = 256
    model = load_model()
    model = model.eval()
    img = cv2.imread('/home/foto1/Database/face_keypoints_76/cropped_images_2/sort031825.jpg')
    img = cv2.resize(img,(256,256))
    raw_img = img
    img = img/255.0
    img = (img-mean)/std
    img = img.transpose((2, 0, 1))
    img = img.reshape((1,) + img.shape)
    input = torch.from_numpy(img).float()
    print input.size()
    input= torch.autograd.Variable(input)
    out = model(input).cpu().data.numpy()
    out = out.reshape(-1,2)
    raw_img = cv2.resize(raw_img,(out_size,out_size))
    for i in xrange(76):
        cv2.circle(raw_img,(int(out[i][0]*out_size),int(out[i][1]*out_size)),1,(255,0,0),-1)
    cv2.imwrite('result.png',raw_img)
    print out

