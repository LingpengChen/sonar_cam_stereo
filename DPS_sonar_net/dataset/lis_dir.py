import os

def list_dirs(path='.'):
    # 获取指定路径下的所有目录名
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    # 打印目录列表
    for dir_name in dirs:
        print(dir_name)

# 示例用法
# list_dirs()  # 列出当前目录下的文件夹
list_dirs('/home/clp/catkin_ws/src/sonar_cam_stereo/DPS_sonar_net/dataset/train')  # 列出指定目录下的文件夹