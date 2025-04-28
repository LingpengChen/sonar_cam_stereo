import trimesh
import numpy as np
from scipy.interpolate import griddata
import os
import pyvista as pv

def process_mesh(input_path, output_path, scale_factor=None, interpolation_factor=2, visualization=True):
    # 加载模型
    scene = trimesh.load(input_path)
    
    # 获取Scene中的第一个mesh
    mesh_name = list(scene.geometry.keys())[0]
    mesh = scene.geometry[mesh_name]
    
    # 获取mesh在场景中的变换
    # 使用scene.graph.transforms获取变换
    transform = scene.graph.transforms.edge_data[('world', 'Landscape-mesh')]['matrix']
    # transform = np.eye(4)  # 默认变换矩阵
    # if 'Landscape-mesh' in scene.graph.transforms.edge_data:
    #     # 从edge_data中获取matrix
    #     edge_data = scene.graph.edge_data['Landscape-mesh']
    #     if 'matrix' in edge_data:
    
    # 应用变换后的mesh
    transformed_mesh = mesh.copy()
    transformed_mesh.apply_transform(transform)
    
    # 打印边界信息
    print(transform)
    print(f"Original mesh bounds: {mesh.bounds}")
    print(f"Scene bounds: {scene.bounds}")
    print(f"Transformed mesh bounds: {transformed_mesh.bounds}")
    
    # 获取变换后的顶点和面
    vertices = transformed_mesh.vertices
    faces = transformed_mesh.faces
    
    # 获取顶点坐标
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    # 创建更密集的采样点
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_new = np.linspace(x_min, x_max, int(len(np.unique(x)) * interpolation_factor))
    y_new = np.linspace(y_min, y_max, int(len(np.unique(y)) * interpolation_factor))
    x_grid, y_grid = np.meshgrid(x_new, y_new)
    
    # 进行插值
    print("Performing interpolation...")
    z_grid = griddata((x, y), z, (x_grid, y_grid), method='cubic')
    # 'linear': 线性插值，速度快，但结果可能不够平滑
    # 'nearest': 最近邻插值，简单但可能产生阶梯状效果
    # 'cubic': 三次样条插值，结果最平滑，但计算量大
    # 'quintic': 五次样条插值，比cubic更平滑，但计算量更大
    
    # 创建新的顶点数组
    new_vertices = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))
    
    # 创建新的面
    print("Creating new faces...")
    nx, ny = x_grid.shape
    new_faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v0 = i * ny + j
            v1 = i * ny + (j + 1)
            v2 = (i + 1) * ny + j
            v3 = (i + 1) * ny + (j + 1)
            new_faces.append([v0, v1, v2])
            new_faces.append([v1, v3, v2])
    
    # 创建新的mesh
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    # 如果需要缩放
    if scale_factor is not None:
        print(f"Applying scaling: {scale_factor}")
        if isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 3:
            # 分别缩放xyz
            matrix = np.eye(4)
            matrix[0:3, 0:3] = np.diag(scale_factor)
            new_mesh.apply_transform(matrix)
        else:
            # 统一缩放
            new_mesh.apply_scale(scale_factor)
    
    # 获取新模型的尺寸
    new_bounds = new_mesh.bounds
    new_dimensions = new_bounds[1] - new_bounds[0]
    print(f"New model dimensions (xyz): {new_dimensions}")
    
    # # 可视化
    # if visualization:
    #     print("Generating visualization...")
    #     plotter = pv.Plotter(shape=(1, 2))
        
    #     # 显示原始模型
    #     plotter.subplot(0, 0)
    #     plotter.add_mesh(pv.wrap(transformed_mesh), show_edges=True, color='tan')
    #     plotter.add_title('Original Mesh')
        
    #     # 显示新模型
    #     plotter.subplot(0, 1)
    #     plotter.add_mesh(pv.wrap(new_mesh), show_edges=True, color='tan')
    #     plotter.add_title('Smoothed Mesh')
        
    #     # 显示两个模型
    #     plotter.show()
    
    # 创建新的场景并保存
    new_scene = trimesh.Scene()
    new_scene.add_geometry(new_mesh)
    new_scene.export(output_path)
    print(f"Saved new mesh to: {output_path}")

def main():
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'heightmap/meshes/heightmap.dae')
    output_path = os.path.join(current_dir, 'heightmap/meshes/heightmap_smooth.obj')
    
    # 设置参数
    # scale_factor = [1.5, 1.5, 1.0]  # 分别缩放xyz
    # scale_factor = 1.5  # 统一缩放
    # scale_factor = None  # 不缩放
    scale_factor = 0.03
    interpolation_factor = 2  # 插值倍数
    
    try:
        # 处理模型
        process_mesh(
            input_path=input_path,
            output_path=output_path,
            scale_factor=scale_factor,
            interpolation_factor=interpolation_factor,
            visualization=True
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()