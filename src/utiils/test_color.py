import cv2
import numpy as np

# <!-- 较浅的蓝绿色 -->
# <color>0.1 0.6 0.5 1.0</color>

#   <color>0.0 0.5 0.4 1.0</color>  <!-- 蓝绿色 -->

# <!-- 较深的蓝绿色 -->
# <color>0.0 0.4 0.3 1.0</color>

def apply_underwater_effect(image, depth=None):
    # 创建蓝绿色滤镜
    underwater_color = 255*np.array([0.4, 0.6, 0.0], dtype=np.float32)  # BGR格式
    
    # 转换图像为浮点型
    img_float = image.astype(np.float32) / 255.0
    
    # 如果提供深度信息，创建基于深度的雾效果
    if depth is not None:
        # 归一化深度图
        depth_normalized = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
        fog_factor = np.expand_dims(depth_normalized, axis=2)
    else:
        # 创建从上到下渐变的雾效果
        height = image.shape[0]
        fog_factor = np.linspace(0.3, 0.8, height)
        fog_factor = np.tile(fog_factor.reshape(height, 1), (1, image.shape[1]))
        fog_factor = np.expand_dims(fog_factor, axis=2)

    # 混合原始图像和水下颜色
    underwater_color = underwater_color.reshape(1, 1, 3) / 255.0
    result = img_float * (1 - fog_factor) + underwater_color * fog_factor
    
    # 添加高斯模糊模拟散射效果
    result = cv2.GaussianBlur(result, (5, 5), 0)
    
    # 转回uint8格式
    result = (result * 255).astype(np.uint8)
    
    return result

# 测试代码
def main():
    # 打开摄像头或视频文件
    cap = cv2.VideoCapture(0)  # 使用0表示默认摄像头
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 应用水下效果
        underwater_frame = apply_underwater_effect(frame)
        
        # 显示原始和效果图像
        combined = np.hstack((frame, underwater_frame))
        cv2.imshow('Original vs Underwater Effect', combined)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()