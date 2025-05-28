from __future__ import division
import torch
import random
import numpy as np
# from scipy.misc import imresize
# from scipy.ndimage.interpolation import zoom
from skimage.transform import resize as skimage_resize
from scipy.ndimage import zoom  # New import path

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb_img, sonar_rect_img):
        for t in self.transforms:
            rgb_img, sonar_rect_img = t(rgb_img, sonar_rect_img)
        return rgb_img, sonar_rect_img


class Normalize(object):
    def __init__(self, mean, std, gamma=0.3):
        self.mean = mean
        self.std = std
        self.gamma = gamma

    def __call__(self, rgb_tensor, sonar_rect_tensor):
        rgb_tensor_norm = rgb_tensor.sub_(self.mean).div_(self.std)
        sonar_rect_tensor_gamma = torch.pow(sonar_rect_tensor, self.gamma)
        return rgb_tensor_norm, sonar_rect_tensor_gamma


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, rgb_img, sonar_rect_img):
        # put it from HWC to CHW format
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        # handle numpy array
        rgb_tensor = torch.from_numpy(rgb_img).float()/255
        sonar_rect_tensor = torch.from_numpy(sonar_rect_img).float()/255
       
        return rgb_tensor, sonar_rect_tensor


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, depth, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        out_h = 240
        out_w = 320
        in_h, in_w, _ = images[0].shape
        x_scaling = np.random.uniform(out_w/in_w, 1)
        y_scaling = np.random.uniform(out_h/in_h, 1)
        scaled_h, scaled_w = round(in_h * y_scaling), round(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        # scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]
        scaled_images = [skimage_resize(im, (scaled_h, scaled_w), preserve_range=True).astype(im.dtype) for im in images]
        scaled_depth = zoom(depth, (y_scaling, x_scaling))

        offset_y = np.random.randint(scaled_h - out_h + 1)
        offset_x = np.random.randint(scaled_w - out_w + 1)
        cropped_images = [im[offset_y:offset_y + out_h, offset_x:offset_x + out_w, :] for im in scaled_images]
        cropped_depth = scaled_depth[offset_y:offset_y + out_h, offset_x:offset_x + out_w]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        return cropped_images, cropped_depth, output_intrinsics
