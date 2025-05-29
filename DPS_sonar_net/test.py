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
from utils import set_seed
SEED_NUM = 0
set_seed(SEED_NUM)
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

# parser.add_argument('--data', metavar='DIR', default="./dataset/test/",help='path to dataset')
parser.add_argument('--data', metavar='DIR', default="/media/clp/9CB0E82FB0E81196/Stereo_dataset/FLsea_vi/sonar_cam_stereo_dataset/",help='path to dataset')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
# parser.add_argument('--pretrained-dps', dest='pretrained_dps', default="./pretrained/dpsnet.pth.tar", metavar='PATH', help='path to pre-trained dpsnet model')
# parser.add_argument('--pretrained-dps', dest='pretrained_dps', default="./checkpoints/sonar_cam_stereo_dataset/05-28-20:05/dpsnet_0_checkpoint.pth.tar", metavar='PATH', help='path to pre-trained dpsnet model')
parser.add_argument('--pretrained-dps', dest='pretrained_dps', default="./checkpoints/sonar_cam_stereo_dataset/05-29-01:29/dpsnet_1_checkpoint.pth.tar", metavar='PATH', help='path to pre-trained dpsnet model')
# parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--output-dir', default='test_result', type=str, help='Output directory for saving predictions in a big 3D numpy file')
parser.add_argument('--ttype', default='test.txt', type=str, help='Text file indicates input data')

parser.add_argument('--label_factor', type=int ,default=1.06, help='label factors, depth of i th pseudo plane = label_factor**i')
parser.add_argument('--nlabel', type=int ,default=48, help='number of label')
parser.add_argument('--mindepth', type=float ,default=1, help='minimum depth')
parser.add_argument('--maxdepth', type=float ,default=16, help='minimum depth')

# parser.add_argument('--output-print', action='store_true', help='print output depth')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency')

def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalize = custom_transforms.Normalize(mean=0.5, std=0.5, gamma=0.3) # normalized_value = (x - mean) / std
    test_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    test_set = SequenceFolder(
        "./dataset/test/",
        # args.data,
        seed=SEED_NUM,
        ttype=args.ttype,
        transform=test_transform,
    )

    print('{} samples found in {} valid scenes'.format(len(test_set), len(test_set.scenes)))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    nlabel = args.nlabel
    mindepth = args.mindepth
    
    dpsnet = PSNet(nlabel, mindepth).to(device)
    # pretrained_model_path = "./pretrained/dpsnet_updated.pth.tar"
    pretrained_model_path = args.pretrained_dps
    weights = torch.load(pretrained_model_path)
    dpsnet.load_state_dict(weights['state_dict'])
    dpsnet.eval()

    output_dir= Path("result")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print_freq = args.print_freq
    errors = np.zeros((8, int(np.ceil(len(test_loader)/print_freq))), np.float32)
    with torch.no_grad():
        for i, (rgb_img, sonar_rect_img, depth_gt, K, KT_inv, distance_range, theta_range) in enumerate(test_loader):
            # if ii % print_freq == 0:
            # i = int(ii / print_freq)
            rgb_img_var = rgb_img.to(device)
            sonar_rect_img_var = sonar_rect_img.to(device)
            depth_gt_var = depth_gt.to(device)
            K_var = K.to(device)
            KT_inv_var = KT_inv.to(device)
            distance_range_var = distance_range.to(device)
            theta_range_var = theta_range.to(device)
            
            start = time.time()
            output_depth = dpsnet(rgb_img_var, sonar_rect_img_var, K_var, KT_inv_var, distance_range_var, theta_range_var)
            elps = time.time() - start
            
            mask = (depth_gt <= args.maxdepth) & (depth_gt >= args.mindepth) & (depth_gt == depth_gt) # tgt_depth == tgt_depth: 排除NaN值(NaN不等于自身)

            # output_depth_inv_ = torch.squeeze(output_depth_inv.data.cpu(),1)
            output_depth_ = torch.squeeze(output_depth.data.cpu(),1)

            errors[:,i] = compute_errors_test(depth_gt[mask], output_depth_[mask])
            # errors[1,:,i] = compute_errors_test(tgt_depth_inv[mask], output_depth_inv_[mask])

            print('Elapsed Time {} Abs Error {:.4f}'.format(elps, errors[0,i]))

            output_depth_array = (output_depth_).numpy()[0]
            gt_depth_array = (depth_gt).numpy()[0]
            np.save(output_dir/'{:04d}{}'.format(i,'.npy'), output_depth_array)
            np.save(output_dir/'{:04d}_gt{}'.format(i,'.npy'), gt_depth_array)
            
            if i > 10: break


    mean_errors = errors.mean(-1)
    error_names = ['abs_rel','abs_diff','sq_rel','rms','log_rms','a1','a2','a3']
    print("{}".format(args.output_dir))
    print("Depth Results : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors))

    np.savetxt(output_dir/'errors.csv', mean_errors, fmt='%1.4f', delimiter=',')


if __name__ == '__main__':
    main()
