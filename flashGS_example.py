import lzma, pickle, numpy as np, torch

def _to_t(npa, dtype=torch.float32):
    return torch.from_numpy(np.asarray(npa)).to(dtype=dtype)

def _rotmat_from_wxyz(q: torch.Tensor) -> torch.Tensor:
    # q: (N,4) [w,x,y,z]
    q = q / torch.linalg.vector_norm(q, dim=1, keepdim=True).clamp_min(1e-12)
    w,x,y,z = q[:,0], q[:,1], q[:,2], q[:,3]
    R = torch.empty((q.size(0),3,3), device=q.device, dtype=q.dtype)
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-w*z); R[:,0,2]=2*(x*z+w*y)
    R[:,1,0]=2*(x*y+w*z);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-w*x)
    R[:,2,0]=2*(x*z-w*y);   R[:,2,1]=2*(y*z+w*x);   R[:,2,2]=1-2*(x*x+y*y)
    return R

def load(
    path: str,
):
    with open(path, "rb") as f:
        data = pickle.loads(lzma.decompress(f.read()))

    qa = bool(data.get("quantization", False))
    have_q_dc = all(k in data for k in ("features_dc","features_dc_scale","features_dc_zero_point"))
    have_q_fr = all(k in data for k in ("features_rest","features_rest_scale","features_rest_zero_point"))

    if qa and have_q_dc and have_q_fr:
        dc_int = _to_t(data["features_dc"], dtype=torch.int32)
        fr_int = _to_t(data["features_rest"], dtype=torch.int32)
        dc = (dc_int - int(np.asarray(data["features_dc_zero_point"]).item())).to(torch.float32) * float(np.asarray(data["features_dc_scale"]).item())
        fr = (fr_int - int(np.asarray(data["features_rest_zero_point"]).item())).to(torch.float32) * float(np.asarray(data["features_rest_scale"]).item())
    else:
        dc = _to_t(data["features_dc"], dtype=torch.float32)
        fr = _to_t(data["features_rest"], dtype=torch.float32)

    if dc.dim()==3 and fr.dim()==3 and dc.size(1)==1 and dc.size(2)==3 and fr.size(1)==15 and fr.size(2)==3:
        shs = torch.cat([dc, fr], dim=1).reshape(dc.size(0), 48).contiguous()
    elif dc.dim()==3 and fr.dim()==3 and dc.size(1)==3 and dc.size(2)==1 and fr.size(1)==3 and fr.size(2)==15:
        shs = torch.cat([dc, fr], dim=2).transpose(1,2).reshape(dc.size(0),48).contiguous()
    else:
        raise ValueError(f"features_* shape inattesa: dc={tuple(dc.shape)}, rest={tuple(fr.shape)}")

    xyz      = _to_t(data["xyz"], dtype=torch.float32).contiguous()
    opacity  = _to_t(data["opacity"], dtype=torch.float32)

    opacity = torch.sigmoid(opacity)
    opacity = opacity.squeeze(-1).clamp_(1e-6, 1-1e-6).contiguous()

    scaling  = _to_t(data["scaling"], dtype=torch.float32).contiguous()   # (N,3) log-space
    rotation = _to_t(data["rotation"], dtype=torch.float32).contiguous()   # (N,4) salvata xyzw

    rotation = rotation[:, [3,0,1,2]]
    rotation = rotation / torch.linalg.vector_norm(rotation, dim=1, keepdim=True).clamp_min(1e-12)

    s = torch.exp(scaling)[:, [2,1,0]]  

    R  = _rotmat_from_wxyz(rotation)
    S2 = (s*s).unsqueeze(1)             
    C  = (R.transpose(1,2) * S2) @ R
    C  = 0.5*(C + C.transpose(1,2))     

    P = torch.tensor([[0., 0., 1.],
                      [0.,-1., 0.],
                      [1., 0., 0.]], device=C.device, dtype=C.dtype)
    C = P @ C @ P.T

    cov3d = torch.stack([C[:,0,0], C[:,0,1], C[:,0,2],
                         C[:,1,1], C[:,1,2], C[:,2,2]], dim=1).contiguous()

    num_verts = int(xyz.size(0))

    xyz     = xyz.to(torch.float32).contiguous()
    shs     = shs.to(torch.float32).contiguous()
    opacity = opacity.to(torch.float32).contiguous()
    cov3d   = cov3d.to(torch.float32).contiguous()

    return num_verts, xyz, shs, opacity, cov3d


import torch
import flash_gaussian_splatting

import os
import json
import time


class Scene:
    def __init__(self, device):
        self.device = device
        self.num_vertex = 0
        self.position = None
        self.shs = None
        self.opacity = None
        self.cov3d = None

    def loadPly(self, scene_path):
        if cmp:
            self.num_vertex, self.position, self.shs, self.opacity, self.cov3d = load(
                scene_path) 
        else:
            self.num_vertex, self.position, self.shs, self.opacity, self.cov3d = flash_gaussian_splatting.ops.loadPly(
                scene_path)

        self.position = self.position.to(self.device)  # 3
        self.shs = self.shs.to(self.device)  # 48
        self.opacity = self.opacity.to(self.device)  # 1
        self.cov3d = self.cov3d.to(self.device)  # 6


class Camera:
    def __init__(self, camera_json):
        self.id = camera_json['id']
        self.img_name = camera_json['img_name']
        self.width = camera_json['width']
        self.height = camera_json['height']
        self.position = torch.tensor(camera_json['position'])
        self.rotation = torch.tensor(camera_json['rotation'])
        self.focal_x = camera_json['fx']
        self.focal_y = camera_json['fy']
        self.zFar = 100.0
        self.zNear = 0.01


# 静态分配内存光栅化器
class Rasterizer:
    # 构造函数中分配内存
    def __init__(self, scene, MAX_NUM_RENDERED, MAX_NUM_TILES):
        # 24 bytes
        self.gaussian_keys_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)
        self.gaussian_keys_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)

        self.MAX_NUM_RENDERED = MAX_NUM_RENDERED
        self.MAX_NUM_TILES = MAX_NUM_TILES
        self.SORT_BUFFER_SIZE = flash_gaussian_splatting.ops.get_sort_buffer_size(MAX_NUM_RENDERED)
        self.list_sorting_space = torch.zeros(self.SORT_BUFFER_SIZE, device=scene.device, dtype=torch.int8)
        self.ranges = torch.zeros((MAX_NUM_TILES, 2), device=scene.device, dtype=torch.int32)
        self.curr_offset = torch.zeros(1, device=scene.device, dtype=torch.int32)

        # 40 bytes
        self.points_xy = torch.zeros((scene.num_vertex, 2), device=scene.device, dtype=torch.float32)
        self.rgb_depth = torch.zeros((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)
        self.conic_opacity = torch.zeros((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)

    # 前向传播（应用层封装）
    def forward(self, scene, camera, bg_color):
        # 属性预处理 + 键值绑定
        self.curr_offset.fill_(0)
        flash_gaussian_splatting.ops.preprocess(scene.position, scene.shs, scene.opacity, scene.cov3d,
                                                camera.width, camera.height, 16, 16,
                                                camera.position, camera.rotation,
                                                camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                                                self.points_xy, self.rgb_depth, self.conic_opacity,
                                                self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                self.curr_offset)

        # 键值对数量判断 + 处理键值对过多的异常情况
        num_rendered = int(self.curr_offset.cpu()[0])
        # print(num_rendered)
        if num_rendered >= self.MAX_NUM_RENDERED:
            raise "Too many k-v pairs!"

        flash_gaussian_splatting.ops.sort_gaussian(num_rendered, camera.width, camera.height, 16, 16,
                                                   self.list_sorting_space,
                                                   self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                   self.gaussian_keys_sorted, self.gaussian_values_sorted)
        # 排序 + 像素着色 + 混色阶段
        out_color = torch.zeros((camera.height, camera.width, 3), device=scene.device, dtype=torch.int8)
        flash_gaussian_splatting.ops.render_16x16(num_rendered, camera.width, camera.height,
                                                  self.points_xy, self.rgb_depth, self.conic_opacity,
                                                  self.gaussian_keys_sorted, self.gaussian_values_sorted,
                                                  self.ranges, bg_color, out_color)
        return out_color


def savePpm(image, path):
    image = image.cpu()
    assert image.dim() >= 3
    assert image.size(2) == 3
    with open(path, 'wb') as f:
        f.write(b'P6\n' + f'{image.size(1)} {image.size(0)}\n255\n'.encode() + image.numpy().tobytes())


def render_scene(test_performance=True):

    device = torch.device('cuda:0')
    bg_color = torch.zeros(3, dtype=torch.float32)  # black

    scene = Scene(device)
    scene.loadPly(model_path)

    with open(camera_path, 'r') as camera_file:
        cameras_json = json.loads(camera_file.read())

    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    MAX_NUM_RENDERED = 2 ** 27
    MAX_NUM_TILES = 2 ** 20
    rasterizer = Rasterizer(scene, MAX_NUM_RENDERED, MAX_NUM_TILES)
    for camera_json in cameras_json:
        camera = Camera(camera_json)
        print("image name = %s" % camera.img_name)

        image = rasterizer.forward(scene, camera, bg_color)  # warm up
        fps = []
        if test_performance:
            n = 20
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                image = rasterizer.forward(scene, camera, bg_color)  # test performance
            torch.cuda.synchronize()
            t1 = time.time()
            print("elapsed time = %f ms" % ((t1 - t0) / n * 1000))
            print("fps = %f" % (n / (t1 - t0)))
            fps.append(n / (t1 - t0))
        image_path = os.path.join(image_dir, "%s.ppm" % camera.img_name)
        savePpm(image, image_path)

    avg_fps = np.mean(fps) if fps else 0    
    with open('fps.csv', 'a') as f:
        f.write(f'{scene_name}, {avg_fps}\n')
       
            

if __name__ == "__main__":

    scenes = ['bonsai', 'room', 'bicycle', 'counter', 'flowers', 'garden', 'stump', 'treehill', 'kitchen', 'playroom', 'drjohnson', 'train', 'truck']
    compressed = [True, False]
    for scene_name in scenes:
        for cmp in compressed:

            image_dir = f'./test_out/{scene_name}_cmp' if cmp else f'./test_out/{scene_name}_orig'
            
            model_path = f'/scratch/cogs25/gaussian-splatting/compressed_models/{scene_name}_comp.ply' if cmp else f'/scratch/cogs25/gaussian-splatting/models/{scene_name}/point_cloud/iteration_30000/point_cloud.ply'
            camera_path = f'/scratch/cogs25/gaussian-splatting/models/{scene_name}/cameras.json'
            render_scene(True)

