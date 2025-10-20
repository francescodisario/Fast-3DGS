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

import copy
import os
import numpy as np
import torch
from random import randint

import torchvision
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from scene.cameras import Camera

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
    
    
class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x: torch.Tensor) -> torch.Tensor:
        return x.half().float()

    @staticmethod
    def backward(_, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def write(string):
    with open(os.path.join(args.model_path, 'results.csv'), 'a') as f:
        print(string, file=f)


def geometric_progression(tmin, tmax, L):
    b = np.exp((np.log(tmax) - np.log(tmin)) / (L - 1))
    print(b)
    arr = np.array([i for i in range(L)])
    return ((b ** arr) * tmin).astype(np.int32)


def accumulate_gradients(scene, opt, render_fun, renderArgs):
    print('Accumulating gradients...')
    for viewpoint_cam in scene.getTrainCameras():

        render_pkg = render_fun(viewpoint_cam, scene.gaussians, *renderArgs)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward(retain_graph=True)
    print('Done')
    


def fine_tune(scene, opt, start_iter, dataset, pipe, ft_iters=1000):
    
    gaussians = scene.gaussians
    gaussians.training_setup(opt)
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    iter_start.record()
    
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(start_iter, start_iter+ft_iters), desc="Training progress")

    for iteration in range(start_iter, start_iter+ft_iters):
        
        gaussians.update_learning_rate(iteration)
        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)


        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss += 0.05 * gaussians._features_rest.abs().mean()
        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                

        gaussians.exposure_optimizer.step()
        gaussians.exposure_optimizer.zero_grad(set_to_none = True)
        if use_sparse_adam:
            visible = radii > 0
            gaussians.optimizer.step(visible, radii.shape[0])
            gaussians.optimizer.zero_grad(set_to_none = True)
        else:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

        

def prune_gaussians(gaussians, k):
    indices = torch.topk(
                    gaussians._opacity.grad.norm(dim=1) +
                    gaussians._xyz.grad.norm(dim=1) +
                    gaussians._scaling.grad.norm(dim=1) +
                    gaussians._rotation.grad.norm(dim=1) +
                    gaussians._features_rest.grad.norm(dim=-1).norm(dim=-1) +
                    gaussians._features_dc.grad.norm(dim=-1).norm(dim=-1),
                    k=k, largest=False)[1].squeeze()
    mask = torch.ones([gaussians._xyz.shape[0]]).bool()
    mask[indices] = False
    return mask


@torch.no_grad()
def apply_mask(gaussians, mask):
    gaussians._xyz = torch.nn.Parameter(gaussians._xyz[mask].detach().clone())
    gaussians._opacity = torch.nn.Parameter(gaussians._opacity[mask].detach().clone())
    gaussians._features_dc = torch.nn.Parameter(gaussians._features_dc[mask].detach().clone())
    gaussians._features_rest = torch.nn.Parameter(gaussians._features_rest[mask].detach().clone())
    gaussians._scaling = torch.nn.Parameter(gaussians._scaling[mask].detach().clone())
    gaussians._rotation = torch.nn.Parameter(gaussians._rotation[mask].detach().clone())
    return gaussians
     
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    gaussians.load_ply(os.path.join(args.pretrained_dir, 'point_cloud', f'iteration_30000', 'point_cloud.ply'))
    first_iter = 30_000
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    baseline_psnr, baseline_ssim = eval(scene, render, first_iter, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), eval_lpips=False, dump_images=True, pre=True)
    
    
    num_gaussians = gaussians._xyz.shape[0]
    pruning_steps = 30
    prune_percent = 0.05

    first_iter += 1
    
    iteration = first_iter
    ft_iters = 1000
    
    print('Setting QAT...')
    gaussians.set_qa()
    print('Evaluating...')
    eval(scene, render, first_iter, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp))
    fine_tune(scene, opt, iteration, dataset, pipe, ft_iters=ft_iters)
    iteration += ft_iters
    print('Evaluating after ft (QAT)')
    eval(scene, render, first_iter, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp))
    
    
    print('Iterative Pruning - STARTED')
    
    gaussians_prev = None
    while True:
        accumulate_gradients(scene, opt, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp))
        k = int(gaussians._xyz.shape[0] * prune_percent)
        mask = prune_gaussians(gaussians, k)
        scene.gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians_prev = copy.deepcopy(gaussians)
        print(f'Pruned {k} gaussians. Tot Gaussians = {gaussians._xyz.shape[0]}')
        gaussians = apply_mask(gaussians, mask)
        print('Eval before ft')
        eval(scene, render, first_iter, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp))
        fine_tune(scene, opt, iteration, dataset, pipe, ft_iters=ft_iters)
        iteration += ft_iters
        print('Eval after ft')
        psnr, ssim = eval(scene, render, first_iter, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp))
        if psnr <= baseline_psnr - 0.1:
            fine_tune(scene, opt, iteration, dataset, pipe, ft_iters=5000)
            eval(scene, render, first_iter, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp))
            size_mb = gaussians.save_lzma(os.path.join(args.model_path, 'compressed.ply'))
            #gaussians_prev.save_lzma(os.path.join(args.model_path, 'compressed_0.lzma'))
            print(f"baseline psnr = {baseline_psnr}, baseline_ssim={baseline_ssim}, iters={iteration}")
            print('Loading...')
            gaussians.load_lzma(os.path.join(args.model_path, 'compressed.ply'), fp16=False)
            print('Eval after loading')
            eval(scene, render, first_iter, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), eval_lpips=True, dump_images=True, pre=False)
            write(f"size_mb, {size_mb}")
            return


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        eval(scene, renderFunc, iteration, renderArgs)
        torch.cuda.empty_cache()

@torch.no_grad
def eval(scene, renderFunc, iteration, renderArgs, eval_lpips=False, dump_images=False, pre=False):
    validation_configs = (
        {'name': 'test',  'cameras': sorted(scene.getTestCameras(), key=lambda x: x.colmap_id)},
        {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],}
    )

    ret_psnr = 0.0
    ret_ssim = 0.0
    
    os.makedirs(os.path.join('render_out', f'./{scene_name}', 'pre'), exist_ok=True)
    os.makedirs(os.path.join('render_out', f'./{scene_name}', 'post'), exist_ok=True)


    for config in validation_configs:
        
        if config['name'] == 'train':
            continue
        cams = config['cameras']
        if not cams: 
            continue

        l1_test = psnr_test = ssim_test = lpips_test = 0.0

        render_time_s = 0.0
        render_frames = 0

        WARMUP = 2

        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        for idx, viewpoint in enumerate(cams):
            torch.cuda.synchronize()
            start_event.record()
            out = renderFunc(viewpoint, scene.gaussians, *renderArgs)  
            end_event.record()
            torch.cuda.synchronize()
            ms = start_event.elapsed_time(end_event)  

            if idx >= WARMUP:  
                render_time_s += ms / 1000.0
                render_frames += 1

            image   = torch.clamp(out["render"], 0.0, 1.0)
            
            if dump_images:
                path = os.path.join('render_out', f'./{scene_name}', 'pre', f'{idx}.png') if pre is True else os.path.join('render_out', f'./{scene_name}', 'post', f'{idx}.png')
                torchvision.utils.save_image(image, path)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            l1_test   += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            if eval_lpips:
                lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

        n = len(cams)
        l1_test   /= n
        psnr_test /= n
        ssim_test /= n
        if eval_lpips:
            lpips_test /= n

        if render_frames > 0 and render_time_s > 0:
            ms_per_frame = (render_time_s / render_frames) * 1000.0
            fps_render = render_frames / render_time_s
        else:
            ms_per_frame = float('nan')
            fps_render = float('nan')

        print(f"\n[ITER {iteration}] Evaluating {config['name']}: "
              f"L1 {l1_test:.4f}  PSNR {psnr_test:.3f}  SSIM {ssim_test:.4f}  "
              f"Render FPS {fps_render:.2f}  ({ms_per_frame:.2f} ms/frame)")

        if config['name'] == 'test':
            ret_psnr = psnr_test
            ret_ssim = ssim_test
            write(f'ITER={iteration},{config["name"]},{psnr_test},{ssim_test},{lpips_test},{scene.gaussians._xyz.shape[0]},{fps_render},{ms_per_frame}')

    return ret_psnr, ret_ssim


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--pretrained_dir", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    scenes = { 
            'nerf_real_360': ['bonsai', 'bicycle', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill'],
            
            'tandt' : ['truck', 'train'],
        
            'db': ['drjohnson', 'playroom'] 
            }
    
    for dataset_name in scenes:
        for scene_name in scenes[dataset_name]:
            args.source_path = f'/scratch/nerf/dataset/{dataset_name}/{scene_name}'
            args.pretrained_dir = f'/scratch/cogs25/gaussian-splatting/models/{scene_name}'  
            args.model_path = f"/scratch/cogs25/compressed_ckpts/{scene_name}/"
            args.eval = True
            
            if scene_name in ["bicycle", "flowers", "garden", "stump", "treehill"]:
                args.images = 'images_4'
            elif scene_name in ["room", "counter", "kitchen", "bonsai"]:
               args.images = 'images_2'
        
            training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
