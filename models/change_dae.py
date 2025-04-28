import trimesh
import numpy as np
from scipy.interpolate import griddata
import os
import pyvista as pv
from PIL import Image
import shutil

def generate_uv_coordinates(vertices, bounds):
    """根据顶点位置生成UV坐标"""
    x_min, y_min = bounds[0][0], bounds[0][1]
    x_max, y_max = bounds[1][0], bounds[1][1]
    
    u = (vertices[:, 0] - x_min) / (x_max - x_min)
    v = (vertices[:, 1] - y_min) / (y_max - y_min)
    
    return np.column_stack((u, v))

def create_material(texture_path):
    """创建包含纹理的材质"""
    material = trimesh.visual.material.SimpleMaterial()
    
    # 加载纹理图像
    image = Image.open(texture_path)
    material.image = image
    
    return material

def process_mesh(input_path, output_path, texture_path, scale_factor=None, interpolation_factor=2, visualization=True):
    # 加载模型
    scene = trimesh.load(input_path)
    
    mesh_name = list(scene.geometry.keys())[0]
    mesh = scene.geometry[mesh_name]
    
    transform = scene.graph.transforms.edge_data[('world', 'Landscape-mesh')]['matrix']
    transformed_mesh = mesh.copy()
    transformed_mesh.apply_transform(transform)
    
    vertices = transformed_mesh.vertices
    faces = transformed_mesh.faces
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_new = np.linspace(x_min, x_max, int(len(np.unique(x)) * interpolation_factor))
    y_new = np.linspace(y_min, y_max, int(len(np.unique(y)) * interpolation_factor))
    x_grid, y_grid = np.meshgrid(x_new, y_new)
    
    print("Performing interpolation...")
    z_grid = griddata((x, y), z, (x_grid, y_grid), method='cubic')
    
    new_vertices = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))
    
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
    
    new_faces = np.array(new_faces)
    
    # 生成UV坐标
    bounds = np.array([[x_min, y_min, z.min()], [x_max, y_max, z.max()]])
    uv_coords = generate_uv_coordinates(new_vertices, bounds)
    
    # 创建新的mesh
    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        visual=trimesh.visual.TextureVisuals(
            uv=uv_coords,
            material=create_material(texture_path)
        )
    )
    
    # 如果需要缩放
    if scale_factor is not None:
        print(f"Applying scaling: {scale_factor}")
        if isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 3:
            matrix = np.eye(4)
            matrix[0:3, 0:3] = np.diag(scale_factor)
            new_mesh.apply_transform(matrix)
        else:
            new_mesh.apply_scale(scale_factor)
    
    new_bounds = new_mesh.bounds
    new_dimensions = new_bounds[1] - new_bounds[0]
    print(f"New model dimensions (xyz): {new_dimensions}")
    
    # 创建新的场景
    new_scene = trimesh.Scene()
    new_scene.add_geometry(new_mesh)
    
    # 导出为OBJ格式
    new_scene.export(output_path)
    
    # 生成MTL文件
    mtl_path = output_path.replace('.obj', '.mtl')
    texture_filename = os.path.basename(texture_path)
    with open(mtl_path, 'w') as f:
        f.write("newmtl material0\n")
        f.write(f"map_Kd {texture_filename}\n")
    
    # 确保纹理图片在正确的位置
    output_dir = os.path.dirname(output_path)
    texture_dest = os.path.join(output_dir, texture_filename)
    
    # 如果纹理不在目标目录，复制过去
    if texture_path != texture_dest:
        shutil.copy2(texture_path, texture_dest)
    
    print(f"Saved new mesh to: {output_path}")
    print(f"Saved material file to: {mtl_path}")
    print(f"Texture file location: {texture_dest}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'heightmap/meshes/heightmap.dae')
    output_path = os.path.join(current_dir, 'heightmap/meshes/heightmap_textured.obj')
    texture_path = os.path.join(current_dir, 'heightmap/meshes/soil_sand_0045_01.jpg')
    
    scale_factor = 0.03
    interpolation_factor = 2
    
    try:
        process_mesh(
            input_path=input_path,
            output_path=output_path,
            texture_path=texture_path,
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