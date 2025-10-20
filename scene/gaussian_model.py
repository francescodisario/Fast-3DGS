#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import lzma
import pickle
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.compress_utils import compress_png, decompress_png, sort_param_dict
import time

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x: torch.Tensor) -> torch.Tensor:
        return x.half().float()

    @staticmethod
    def backward(_, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output
    

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.qa = False

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.qa:
            return self.scaling_qa(self.scaling_activation(self._scaling))
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self.qa:
            return self.rotation_activation(self.rotation_qa(self._rotation))
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        # if self.qa:
        #     return self.xyz_qa(self._xyz)
        return self._xyz
    
    @property
    def get_features(self):
        if self.qa:
            features_dc = self.features_dc_qa(self._features_dc)
            features_rest = self.features_rest_qa(self._features_rest)
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        if self.qa:
            return self.opacity_qa(self.opacity_activation(self._opacity))
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
            
    def set_qa(self):
        self.qa = True
        self.features_dc_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.features_rest_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.opacity_qa = FakeQuantizationHalf.apply
        self.scaling_qa = FakeQuantizationHalf.apply
        self.rotation_qa = FakeQuantizationHalf.apply
        self.xyz_qa = FakeQuantizationHalf.apply
        

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    @torch.no_grad()
    def save_lzma(self, path, sort_morton=True):
        
        if sort_morton:
            self._sort_morton()
        if isinstance(path, str):
            mkdir_p(os.path.dirname(os.path.abspath(path)))

        #dtype = torch.half if half_precision else torch.float32

        save_dict = dict()

        save_dict["quantization"] = self.qa

        # save position
        # if self.qa:
        #     save_dict["xyz"] = self.get_xyz.detach().half().cpu().numpy()
        # else:
        #     save_dict["xyz"] = self._xyz.detach().cpu().numpy()

        # save color features
        if self.qa:
            features_dc_q = torch.quantize_per_tensor(
                self._features_dc.detach(),
                self.features_dc_qa.scale,
                self.features_dc_qa.zero_point,
                self.features_dc_qa.dtype,
            ).int_repr()
            save_dict["features_dc"] = features_dc_q.cpu().numpy()
            save_dict["features_dc_scale"] = self.features_dc_qa.scale.cpu().numpy()
            save_dict[
                "features_dc_zero_point"
            ] = self.features_dc_qa.zero_point.cpu().numpy()

            features_rest_q = torch.quantize_per_tensor(
                self._features_rest.detach(),
                self.features_rest_qa.scale,
                self.features_rest_qa.zero_point,
                self.features_rest_qa.dtype,
            ).int_repr()
            save_dict["features_rest"] = features_rest_q.cpu().numpy()
            save_dict["features_rest_scale"] = self.features_rest_qa.scale.cpu().numpy()
            save_dict[
                "features_rest_zero_point"
            ] = self.features_rest_qa.zero_point.cpu().numpy()
            
            save_dict['xyz'] = self._xyz.detach().cpu().numpy()
            save_dict['opacity'] = self._opacity.half().detach().cpu().numpy()
            save_dict['scaling'] = self._scaling.half().detach().cpu().numpy()
            save_dict['rotation'] = self._rotation.half().detach().cpu().numpy()
            
        else:
            save_dict["features_dc"] = self._features_dc.detach().cpu().numpy()
            save_dict["features_rest"] = self._features_rest.detach().cpu().numpy()
            save_dict['xyz'] = self._xyz.cpu().numpy()
            save_dict['opacity'] = self._opacity.detach().cpu().numpy()
            save_dict['scaling'] = self._scaling.detach().cpu().numpy()
            save_dict['rotation'] = self._rotation.detach().cpu().numpy()
            
        save_dict['active_degree'] = self.active_sh_degree
            
        serialized_data = pickle.dumps(save_dict)

        compressed_data = lzma.compress(serialized_data, preset=9)

        compressed_size_bytes = len(compressed_data)

        compressed_size_mb = compressed_size_bytes / (1024 * 1024)  # Convert bytes to MB

        print(f"Size of the compressed data: {compressed_size_mb:.2f} MB")
        
        with open(path, 'wb') as file:
            file.write(compressed_data)
            
        return compressed_size_mb
    
    
    def load_lzma(self, path: str, device: str = "cuda", fp16: bool = True, drop_quant: bool = True):
        with open(path, "rb") as f:
            compressed = f.read()
        data = pickle.loads(lzma.decompress(compressed))

        target_dtype = torch.float16 if fp16 else torch.float32

        def _to_t(npa, dtype=None):
            t = torch.from_numpy(npa).to(device)
            return t.to(dtype) if dtype is not None else t

        qa = bool(data.get("quantization", False))
        have_q_dc = all(k in data for k in ["features_dc", "features_dc_scale", "features_dc_zero_point"])
        have_q_fr = all(k in data for k in ["features_rest", "features_rest_scale", "features_rest_zero_point"])

        if qa and have_q_dc and have_q_fr:
            dc_int   = _to_t(np.asarray(data["features_dc"]), dtype=torch.int32)
            dc_scale = float(np.asarray(data["features_dc_scale"]))
            dc_zp    = int(np.asarray(data["features_dc_zero_point"]))
            dc = (dc_int - dc_zp).to(torch.float32) * dc_scale
            dc = dc.to(target_dtype)

            fr_int   = _to_t(np.asarray(data["features_rest"]), dtype=torch.int32)
            fr_scale = float(np.asarray(data["features_rest_scale"]))
            fr_zp    = int(np.asarray(data["features_rest_zero_point"]))
            fr = (fr_int - fr_zp).to(torch.float32) * fr_scale
            fr = fr.to(target_dtype)
        else:
            dc = _to_t(np.asarray(data["features_dc"]), dtype=target_dtype)
            fr = _to_t(np.asarray(data["features_rest"]), dtype=target_dtype)

        self._features_dc   = nn.Parameter(dc.contiguous(), requires_grad=True)
        self._features_rest = nn.Parameter(fr.contiguous(), requires_grad=True)

        if "xyz" in data:
            self._xyz = nn.Parameter(_to_t(np.asarray(data["xyz"]), dtype=target_dtype).contiguous(), requires_grad=True)
        if "opacity" in data:
            self._opacity = nn.Parameter(_to_t(np.asarray(data["opacity"]), dtype=target_dtype).contiguous(), requires_grad=True)
        if "scaling" in data:
            self._scaling = nn.Parameter(_to_t(np.asarray(data["scaling"]), dtype=target_dtype).contiguous(), requires_grad=True)
        if "rotation" in data:
            self._rotation = nn.Parameter(_to_t(np.asarray(data["rotation"]), dtype=target_dtype).contiguous(), requires_grad=True)

        if "active_degree" in data:
            self.active_sh_degree = int(data["active_degree"])

        if drop_quant:
            self.qa = False
            for m in (getattr(self, "features_dc_qa", None), getattr(self, "features_rest_qa", None)):
                if m is not None:
                    if hasattr(m, "disable_fake_quant"): m.disable_fake_quant()
                    if hasattr(m, "disable_observer"):  m.disable_observer()
        else:

            self.qa = qa

        return True
        

    def _sort_morton(self):
        
        def splitBy3(a):
            x = a & 0x1FFFFF  # we only look at the first 21 bits
            x = (x | x << 32) & 0x1F00000000FFFF
            x = (x | x << 16) & 0x1F0000FF0000FF
            x = (x | x << 8) & 0x100F00F00F00F00F
            x = (x | x << 4) & 0x10C30C30C30C30C3
            x = (x | x << 2) & 0x1249249249249249
            
            return x
        
        def mortonEncode(pos: torch.Tensor) -> torch.Tensor:
            x, y, z = pos.unbind(-1)
            answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
            answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
            
            return answer
        
        with torch.no_grad():
            xyz_q = (
                (2**21 - 1)
                * (self._xyz - self._xyz.min(0).values)
                / (self._xyz.max(0).values - self._xyz.min(0).values)
            ).long()
            order = mortonEncode(xyz_q).sort().indices
            self._xyz = nn.Parameter(self._xyz[order], requires_grad=True)
            self._opacity = nn.Parameter(self._opacity[order], requires_grad=True)

            self._features_rest = nn.Parameter(
                self._features_rest[order], requires_grad=True
            )
            self._features_dc = nn.Parameter(
                self._features_dc[order], requires_grad=True
            )

            self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
            self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)
            
            
    def save_png(self, path):
        path = os.path.join(path, "png")
        mkdir_p(path)
        start_time = time.time()
        opacities = self.get_opacity
        N = opacities.numel()
        n_sidelen = int(N**0.5)
        n_crop = N - n_sidelen**2
        if n_crop:
            index = torch.argsort(opacities.squeeze(), descending=True)
            mask = torch.zeros(N, dtype=torch.bool, device=opacities.device).scatter_(
                0, index[:-n_crop], True
            )
            self.prune(mask.squeeze())
        meta = {}
        param_dict = {
            "xyz": self._xyz,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self.get_rotation,
        }
        param_dict = sort_param_dict(param_dict, n_sidelen)
        for k in param_dict.keys():
            if param_dict[k] is not None:
                if k == "sb_params":
                    for i in range(self.sb_number):
                        meta[f"sb_{i}_color"] = compress_png(
                            path, f"sb_{i}_color", param_dict[k][:, i, :3], n_sidelen
                        )

                        meta[f"sb_{i}_lobe"] = compress_png(
                            path, f"sb_{i}_lobe", param_dict[k][:, i, 3:], n_sidelen
                        )
                elif k == "xyz":
                    meta[k] = compress_png(path, k, param_dict[k], n_sidelen, bit=32)
                else:
                    meta[k] = compress_png(path, k, param_dict[k], n_sidelen)

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)
        end_time = time.time()
        print(f"Compression time: {end_time - start_time:.2f} seconds")