from __future__ import division
import torch
import torch.nn.functional as F

pixel_coords = None

def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def warp_differentiable(K, KT_inv, target_rgb_shape, sonar_rect, depth, 
                        distance_range, theta_range, alpha=60):
    """
    Inverse warp a source image to the target image plane.

    Args:
        sonar_rect: the source feature (where to sample pixels) -- [B, CH, H, W]
        depth: scalar, depth of pseudo plane 
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    # check_sizes(K, 'intrinsics', 'B33')
    # check_sizes(KT_inv, 'intrinsics', 'B33')
    # check_sizes(sonar_rect, 'sonar_rect', 'BCHW')
    # # 确保是张量
    # if not isinstance(alpha, torch.Tensor):
    #     alpha = torch.tensor(alpha, dtype=torch.float32, device=sonar_rect.device)
    # if not isinstance(depth, torch.Tensor):
    #     depth = torch.tensor(depth, dtype=torch.float32, device=sonar_rect.device)
    
    # 确保所有输入都是 float32 类型
    dtype = torch.float32
    device = sonar_rect.device
    
    # 确保标量参数是 float32
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth, dtype=dtype, device=device)
    elif depth.dtype != dtype:
        depth = depth.to(dtype)
    
    # 确保矩阵参数是 float32
    if distance_range.dtype != dtype:
        distance_range = distance_range.to(dtype)
    if theta_range.dtype != dtype:
        theta_range = theta_range.to(dtype)
    
    (_, _, height, width) = target_rgb_shape
    
    # 获取图像尺寸
    batch, _, sonar_rect_height, sonar_rect_width = sonar_rect.shape
    
    
    # 创建目标图像中的坐标网格
    y_coords = torch.arange(0, height, device=sonar_rect.device, dtype=torch.float32)
    x_coords = torch.arange(0, width, device=sonar_rect.device, dtype=torch.float32)
    
    y_coords, x_coords = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # 将坐标转换为齐次坐标
    ones = torch.ones_like(x_coords)
    pixel_coords = torch.stack([x_coords, y_coords, ones], dim=-1)  # [H,W,3]
    
    # 这里需要修改：将坐标扩展到批量维度并适当处理内参矩阵
    pixel_coords_flat = pixel_coords.reshape(-1, 3).unsqueeze(0).expand(batch, -1, -1)  # [B,H*W,3]
    
    # 批量矩阵乘法应用内参逆矩阵
    normalized_coords = torch.bmm(pixel_coords_flat, KT_inv)  # [B,H*W,3]

    
    # 提取归一化坐标分量
    a = normalized_coords[:, :, 0]  # [B,H*W]
    b = normalized_coords[:, :, 1]  # [B,H*W]
    
    # 计算theta和d
    theta = torch.atan(a)
    alpha = torch.deg2rad(alpha)
        
    d = depth * torch.tan(alpha) / (b + torch.tan(alpha))
    
    # 计算原图像中的坐标
    theta_prime = (sonar_rect_width/theta_range) * (torch.rad2deg(theta) + theta_range/2) # [B,H*W]
    d_prime = (sonar_rect_height/distance_range) * d
  
    # 重塑为原始网格形状
    map_x = theta_prime.view(batch, height, width)
    map_y = d_prime.view(batch, height, width)
    
    # 将像素坐标从[0, width/height]范围转换为[-1, 1]范围，适用于grid_sample
    map_x = 2.0 * map_x / (sonar_rect_width - 1) - 1.0
    map_y = 2.0 * map_y / (sonar_rect_height - 1) - 1.0
    
    # 合并成采样网格
    grid = torch.stack([map_x, map_y], dim=-1)  # [B,H,W,2]
    
    # 使用grid_sample进行可微分的重采样
    projected_feat = F.grid_sample(sonar_rect, grid, mode='nearest', padding_mode='zeros', align_corners=True)

    return projected_feat

    # projected_feat = torch.nn.functional.grid_sample(feat, src_pixel_coords, padding_mode=padding_mode)
    # feat              = (N=1, C=32, H=xx, W=xx)
    # src_pixel_coords  = (N=1, H'=120, W'=160, 2)
    # projected_feat    =（N=1, C=32, H'=120, W'=160)


def warp_differentiable_optimized(K, KT_inv, target_rgb_shape, sonar_rect, depth, 
                        alpha=60, distance_range=8, theta_range=60):
    """
    优化版本的可微分图像扭曲函数
    """
    (height, width, _) = target_rgb_shape
    batch, _, sonar_rect_height, sonar_rect_width = sonar_rect.shape
    device = sonar_rect.device
    
    # 预计算常量，避免在每次调用时重新计算
    theta_scale = sonar_rect_width / theta_range
    d_scale = sonar_rect_height / distance_range
    half_theta_range = theta_range / 2
    alpha_rad = torch.deg2rad(torch.tensor(alpha, device=device))
    tan_alpha = torch.tan(alpha_rad)
    
    # 使用预先计算的网格坐标（如果可能，将此移到函数外部）
    if not hasattr(warp_differentiable_optimized, 'pixel_coords') or \
       warp_differentiable_optimized.pixel_coords.shape[0] != height or \
       warp_differentiable_optimized.pixel_coords.shape[1] != width:
        y_coords = torch.arange(0, height, device=device, dtype=torch.float32)
        x_coords = torch.arange(0, width, device=device, dtype=torch.float32)
        y_coords, x_coords = torch.meshgrid(y_coords, x_coords, indexing='ij')
        ones = torch.ones_like(x_coords)
        warp_differentiable_optimized.pixel_coords = torch.stack([x_coords, y_coords, ones], dim=-1)  # [H,W,3]
    
    # 使用预先计算的网格
    pixel_coords = warp_differentiable_optimized.pixel_coords
    
    # 一次性矩阵乘法计算归一化坐标
    normalized_coords = torch.matmul(pixel_coords.reshape(-1, 3), KT_inv)  # [H*W,3]
    
    # 提取坐标分量并直接计算
    a = normalized_coords[:, 0]
    b = normalized_coords[:, 1]
    
    # 计算 theta 和 d
    theta = torch.atan(a)
    d = depth * tan_alpha / (b + tan_alpha)
    
    # 直接计算原图像中的坐标
    theta_prime = theta_scale * (torch.rad2deg(theta) + half_theta_range)
    d_prime = d_scale * d
    
    # 重塑并缩放坐标（使用就地操作减少内存使用）
    map_x = theta_prime.view(height, width)
    map_y = d_prime.view(height, width)
    
    # 坐标归一化
    map_x = map_x.mul_(2.0 / (sonar_rect_width - 1)).sub_(1.0)
    map_y = map_y.mul_(2.0 / (sonar_rect_height - 1)).sub_(1.0)
    
    # 创建采样网格并扩展到批处理大小
    grid = torch.stack([map_x, map_y], dim=-1).unsqueeze(0)
    if batch > 1:
        grid = grid.expand(batch, -1, -1, -1)
    
    # 使用 grid_sample 进行采样
    return F.grid_sample(sonar_rect, grid, mode='nearest', padding_mode='zeros', align_corners=True)
