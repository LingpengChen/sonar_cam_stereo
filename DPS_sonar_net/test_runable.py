from models import PSNet as PSNet

import argparse
import time
import csv

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import custom_transforms
from utils import tensor2array
from loss_functions import compute_errors_test
from sequence_folders import SequenceFolder

import os
from path import Path
# from scipy.misc import imsave
import imageio

from tensorboardX import SummaryWriter

# python3 test.py ./dataset/test/ --sequence-length 2 --output-print --pretrained-dps ./pretrained/dpsnet.pth.tar

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', metavar='DIR', default="./dataset/test/",
                    help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=2)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--pretrained-dps', dest='pretrained_dps', default="./pretrained/dpsnet.pth.tar", metavar='PATH',
                    help='path to pre-trained dpsnet model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--output-dir', default='result', type=str, help='Output directory for saving predictions in a big 3D numpy file')
parser.add_argument('--ttype', default='test.txt', type=str, help='Text file indicates input data')
parser.add_argument('--nlabel', type=int ,default=64, help='number of label')
parser.add_argument('--mindepth', type=float ,default=0.5, help='minimum depth')
parser.add_argument('--maxdepth', type=float ,default=10, help='maximum depth')
parser.add_argument('--output-print', action='store_true', help='print output depth')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency')

def main():
    args = parser.parse_args()
    output_dir = Path("result")

    sequence_length = 3
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # normalized_value = (x - mean) / std
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    val_set = SequenceFolder(
        "./dataset/test/",
        transform=valid_transform,
        seed=0,
        ttype='test.txt',
        sequence_length=sequence_length
    )

    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    nlabel = 64
    mindepth = 0.5
    
    dpsnet = PSNet(nlabel, mindepth).cuda()
    pretrained_model_path = "./pretrained/dpsnet_updated.pth.tar"
    weights = torch.load(pretrained_model_path)
    dpsnet.load_state_dict(weights['state_dict'])
    dpsnet.eval()

    output_dir= Path("result")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print_freq = 1
    errors = np.zeros((2, 8, int(np.ceil(len(val_loader)/print_freq))), np.float32)
    with torch.no_grad():
        for ii, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth) in enumerate(val_loader):
            if ii % print_freq == 0:
                i = int(ii / print_freq)
                tgt_img_var = Variable(tgt_img.cuda())
                ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
                ref_poses_var = [Variable(pose.cuda()) for pose in ref_poses] # the relavative pose of reference frame
                intrinsics_var = Variable(intrinsics.cuda())
                intrinsics_inv_var = Variable(intrinsics_inv.cuda())
                tgt_depth_var = Variable(tgt_depth.cuda())

                # compute output
                pose = torch.cat(ref_poses_var,1)
                start = time.time()
                output_depth = dpsnet(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
                elps = time.time() - start
                tgt_depth_inv = mindepth*nlabel/tgt_depth
                output_depth_inv = mindepth*nlabel/output_depth
 
                mask = (tgt_depth <= args.maxdepth) & (tgt_depth >= args.mindepth) & (tgt_depth == tgt_depth) # tgt_depth == tgt_depth: 排除NaN值(NaN不等于自身)

                output_depth_inv_ = torch.squeeze(output_depth_inv.data.cpu(),1)
                output_depth_ = torch.squeeze(output_depth.data.cpu(),1)

                errors[0,:,i] = compute_errors_test(tgt_depth[mask], output_depth_[mask])
                errors[1,:,i] = compute_errors_test(tgt_depth_inv[mask], output_depth_inv_[mask])

                print('Elapsed Time {} Abs Error {:.4f}'.format(elps, errors[0,0,i]))

                if True:
                    output_depth_inv_n = (output_depth_inv_).numpy()[0]
                    np.save(output_dir/'{:04d}{}'.format(i,'.npy'), output_depth_inv_n)
                    disp = (255*tensor2array(torch.from_numpy(output_depth_inv_n), max_value=args.nlabel, colormap='bone')).astype(np.uint8)
                    # imsave(output_dir/'{:04d}_disp{}'.format(i,'.png'), disp)
                    imageio.imwrite(output_dir/'{:04d}_inv{}'.format(i,'.png'), disp)


    mean_errors = errors.mean(2)
    error_names = ['abs_rel','abs_diff','sq_rel','rms','log_rms','a1','a2','a3']
    print("{}".format(args.output_dir))
    print("Depth Results : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

    print("Disparity Results : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[1]))

    np.savetxt(output_dir/'errors.csv', mean_errors, fmt='%1.4f', delimiter=',')


if __name__ == '__main__':
    main()
