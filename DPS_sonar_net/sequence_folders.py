import torch.utils.data as data
import numpy as np
# from scipy.misc import imread
import imageio

from path import Path
import random

# def load_as_float(path):
#     return imread(path).astype(np.float32)

def load_as_float(path):
    return imageio.imread(path).astype(np.float32)

class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, ttype='train.txt', transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/ttype
        # scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        scenes = [self.root/folder.rstrip() for folder in open(scene_list_path)]
        self.ttype = ttype
        self.scenes = sorted(scenes)
        self.transform = transform
        self.crawl_folders()

    def crawl_folders(self):
        sequence_set = []

        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            distance_range, theta_range = np.genfromtxt(scene/'sonar.txt')

            # poses = np.genfromtxt(scene/'pose.txt').astype(np.float32)
            rgb_img = scene.files('rgb.png')[0]
            sonar_rect_img = scene.files('sonar_rect.png')[0]
            depth_gt = scene.files('depth.npy')[0]
            
            # pose_tgt = np.concatenate((poses[i,:].reshape((3,4)), np.array([[0,0,0,1]])), axis=0)
            sample = {'intrinsics': intrinsics, 'rgb_img': rgb_img, 
                      'sonar_rect_img': sonar_rect_img, 'depth_gt': depth_gt,
                      'distance_range': distance_range, 'theta_range': theta_range}
            sequence_set.append(sample)

        if self.ttype == 'train.txt':
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        rgb_img = load_as_float(sample['rgb_img'])
        sonar_rect_img = load_as_float(sample['sonar_rect_img'])
        depth_gt = np.load(sample['depth_gt'])
        K = np.copy(sample['intrinsics'])
        KT_inv = np.linalg.inv(K.T)
        # ref_poses = sample['ref_poses']
        distance_range = np.copy(sample['distance_range'])
        theta_range = np.copy(sample['theta_range'])
        
        if self.transform is not None:
            rgb_img, sonar_rect_img = self.transform(rgb_img, sonar_rect_img)
    
        return rgb_img, sonar_rect_img.unsqueeze(0), depth_gt, K, KT_inv, distance_range.reshape(1), theta_range.reshape(1)    # np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
