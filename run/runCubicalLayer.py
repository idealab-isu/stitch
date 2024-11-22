import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import shutil
from models.dataset import DatasetNP, DatasetNP_TDA
from models.fields import NPullNetwork
import argparse
from pyhocon import ConfigFactory
import datetime
from shutil import copyfile
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from models.utils import get_root_logger, print_log
import math
import mcubes
from pyhocon import ConfigFactory
import warnings
import json
# from torchviz import make_dot
# import graphviz

warnings.filterwarnings("ignore")

import gudhi as gd


from models.cubicalLayer import ConnectedComponentLoss

import wandb
wandb.login()

class Runner:
    def __init__(self, args, conf_path, mode='train'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self._train_backup_done = False
        self._validate_backup_done = False

        self.conf = ConfigFactory.parse_string(conf_text)
        self.use_wandb = self.conf.get_bool('general.use_wandb')
        self.conf['dataset.np_data_name'] = self.conf['dataset.np_data_name']
        # self.conf['dataset.data_dir']
        self.dataset_name = self.conf['dataset.dataset_type']

        if self.use_wandb:
            logging_config = self.conf.as_plain_ordered_dict()
            wandb.init(project=self.conf['general.project_name'], config=logging_config, name=args.dataname)

        if self.dataset_name == 'srb':
            self.gt_path_metrics = os.path.join(self.conf['dataset.data_dir'] + '/ground_truth/', args.dataname + '.ply')
            self.input_path_metrics = os.path.join(self.conf['dataset.data_dir'] + '/scans/', args.dataname + '.ply')
        elif self.dataset_name == 'famous':
            temp_name = args.dataname.split(".xyz")[0]
            self.gt_path_metrics = os.path.join(self.conf['dataset.data_dir'] + '/03_meshes/', temp_name + '.ply')
            self.input_path_metrics = os.path.join(self.conf['dataset.data_dir'] + '/04_pts_ply/', args.dataname + '.ply')
        elif self.dataset_name == 'abc':
            temp_name = args.dataname.split(".xyz")[0]
            self.gt_path_metrics = os.path.join(self.conf['dataset.data_dir'] + '/03_meshes/', temp_name + '.ply')
            self.input_path_metrics = None
        elif self.dataset_name == 'dfaust':
            pass

        self.base_exp_dir = self.conf['general.base_exp_dir'] + args.dir
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        self.mode = mode
        self.save_sdf = self.conf.get_bool('train.save_sdf')
        self.train_sdf_grid_res = self.conf.get_int('train.train_sdf_grid_res')
        self.eval_sdf_grid_res = self.conf.get_int('train.eval_sdf_grid_res')
        print('save_sdf: ', self.save_sdf)
        print('train_sdf_grid_res: ', self.train_sdf_grid_res)
        print('eval_sdf_grid_res: ', self.eval_sdf_grid_res)
        self.sigma_val = self.conf.get_float('train.sigma_val')
        if self.mode == 'train':
            self.dataset_np = DatasetNP(self.conf['dataset'], args.dataname, self.dataset_name, self.sigma_val)
        else:
            # get TDA radius parameter
            self.persistence_radius = self.conf.get_float('train.persistence_radius')
            self.persistence_dim = self.conf.get_int('train.persistence_dim')
            self.persistence_lambda_1 = self.conf.get_float('train.persistence_lambda_1')
            self.persistence_lambda_2 = self.conf.get_float('train.persistence_lambda_2')
            self.eikonal_lambda = self.conf.get_float('train.eikonal_lambda')
            print('persistence_radius: ', self.persistence_radius)
            print('persistence_dim: ', self.persistence_dim)
            print('persistence_lambda_1: ', self.persistence_lambda_1)
            print('persistence_lambda_2: ', self.persistence_lambda_2)
            print('eikonal_lambda: ', self.eikonal_lambda)
            self.dataset_np = DatasetNP_TDA(self.conf['dataset'], args.dataname, self.dataset_name, self.persistence_radius, self.persistence_dim, self.sigma_val, self.base_exp_dir)

        self.dataname = args.dataname
        self.iter_step = 0

        # Training parameters
        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.eval_num_points = self.conf.get_int('train.eval_num_points')
        self.metric_eval_pts = self.conf.get_int('train.metric_sample_pts')
        self.curriculum_start = self.conf.get_int('train.curriculum_start')
        self.curriculum_interval = self.conf.get_int('train.curriculum_interval')

        print("curriculum_start: ", self.curriculum_start)
        print("curriculum_interval: ", self.curriculum_interval)

        # Networks
        self.sdf_network = NPullNetwork(**self.conf['model.sdf_network']).to(self.device)

        print('network: ', self.sdf_network)
        self.optimizer = torch.optim.Adam(self.sdf_network.parameters(), lr=self.learning_rate)

        # Backup codes and configs for debug
        # if self.mode == 'train' or self.mode == 'train_tda':
        self.file_backup(self.mode)
    
    def visualize_full_backward(self):
        loss_fns = [self.compute_]

    def train(self):
        self.file_backup(self.mode)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size
        if self.mode == 'train_tda' or self.mode == 'curriculum':
            connectedCompLoss = ConnectedComponentLoss(sdf_res=self.train_sdf_grid_res, maxdim=self.persistence_dim, base_dir=self.base_exp_dir)
        

        res_step = self.maxiter - self.iter_step

        def hook_fn(name):
            def hook(grad):
                # print(f"Gradient of {name}: mean={grad.abs().mean().item():.6f}, std={grad.std().item():.6f}")
                pass
            return hook

        for iter_i in tqdm(range(res_step)):
            self.update_learning_rate_np(iter_i)

            if self.mode == 'train':
                points, samples, point_gt = self.dataset_np.np_train_data(batch_size)
            else:
                points, samples, point_gt, all_dim_barcodes = self.dataset_np.np_train_data(batch_size)
                
            samples.requires_grad = True

            gradients_sample = self.sdf_network.gradient(samples).squeeze()
            sdf_sample = self.sdf_network.sdf(samples)
            grad_norm = F.normalize(gradients_sample, dim=1)
            sample_moved = samples - grad_norm * sdf_sample

            loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1).mean()
            
            # Initialize topology losses to zero
            topology_loss1 = topology_loss2 = top_loss = torch.tensor(0.0, device=loss_sdf.device)
            loss = loss_sdf

            apply_tda = False
            if self.mode == 'train_tda':
                apply_tda = True
            elif self.mode == 'curriculum':
                apply_tda = (self.iter_step >= self.curriculum_start) and (self.iter_step % self.curriculum_interval == 0)

            if apply_tda:
                sdf_reshaped = sdf_sample.reshape(self.train_sdf_grid_res, self.train_sdf_grid_res, self.train_sdf_grid_res)
                sdf_reshaped.register_hook(hook_fn("sdf_reshaped"))
                
                topology_loss1, topology_loss2 = connectedCompLoss(sdf_reshaped, self.iter_step)
                top_loss = self.persistence_lambda_1 * topology_loss1 + self.persistence_lambda_2 * topology_loss2
                loss = loss_sdf + top_loss

                print(f"Applying TDA loss at iteration {self.iter_step}")
                print(f"topology_loss1 = {topology_loss1.item()} topology_loss2 = {topology_loss2.item()}")
                print(f"top_loss = {top_loss.item()} loss_sdf = {loss_sdf.item()} total_loss = {loss.item()}")

               

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.iter_step += 1

            if self.iter_step % self.report_freq == 0:
                    
                log_dict = {
                    'iter': self.iter_step,
                    'total_loss': loss.item(),
                    'loss_sdf': loss_sdf.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }

                if apply_tda:
                    log_dict.update({
                        'top_loss': top_loss.item(),
                        'topology_loss1': topology_loss1.item(),
                        'topology_loss2': topology_loss2.item(),
                        'topology_loss1_scaled': (self.persistence_lambda_1 * topology_loss1).item(),
                        'topology_loss2_scaled': (self.persistence_lambda_2 * topology_loss2).item()
                    })
                if self.use_wandb:
                    wandb.log(log_dict)
                else:
                    print(log_dict)

            if apply_tda or (self.mode == "curriculum" and self.iter_step % self.curriculum_interval == 0 and self.iter_step >= self.curriculum_start):
                print_log(f'iter:{self.iter_step:8>d} cd_l1 = {loss_sdf.item()} lr={self.optimizer.param_groups[0]["lr"]}', logger=logger)
                print_log(f'iter:{self.iter_step:8>d} total_loss = {loss.item()} loss_sdf = {loss_sdf.item()} top_loss = {top_loss.item()} topology_loss1 = {topology_loss1.item()} topology_loss2 = {topology_loss2.item()}', logger=logger)
                print_log(f'iter:{self.iter_step:8>d} scaled_topology_loss1 = {(self.persistence_lambda_1 * topology_loss1).item()} scaled_topology_loss2 = {(self.persistence_lambda_2 * topology_loss2).item()}', logger=logger)
                # self.save_tda_plot(zero_dim_barcodes[0])
                # img = plt.imread(os.path.join(self.base_exp_dir, f"cc_{self.iter_step}.jpg"))
                # wandb.log({"persistence diagram": wandb.Image(img)})

            if self.iter_step % self.val_freq == 0 and self.iter_step != 0:
                self.validate_mesh(resolution=self.eval_sdf_grid_res, threshold=args.mcubes_threshold, point_gt=point_gt, iter_step=self.iter_step, logger=logger, save_sdf=self.save_sdf)

            if self.iter_step % self.save_freq == 0 and self.iter_step != 0: 
                self.save_checkpoint()

        self.generate_metrics(self.gt_path_metrics, self.input_path_metrics, self.metric_eval_pts)
        if self.use_wandb:
            wandb.finish()
    
    def normalize_pointcloud(self, points):
        """
        Normalize the points of the point cloud to fit within a unit cube [0, 1]^3
        while preserving the aspect ratio.
        """
        # Calculate the bounding box of the points
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)

        # Find the maximum range to maintain aspect ratio
        max_range = np.max(max_bound - min_bound)
        
        # Calculate scale factor for normalization to [0, 1] range
        scale = 1.0 / max_range  # Scale factor to fit the points within [0, 1]

        # Normalize the points (shift by min_bound and scale)
        normalized_points = (points - min_bound) * scale

        return normalized_points
    
    def torch_distance_matrix(self, A, B, device='cuda', batch_size=1024):
        """
        Compute a distance matrix in a memory-efficient manner using PyTorch.
        """
        A = torch.tensor(A, device=device)
        B = torch.tensor(B, device=device)

        dist_matrix = torch.zeros(A.size(0), B.size(0), device=device)

        for i in range(0, A.size(0), batch_size):
            end_i = min(i + batch_size, A.size(0))
            for j in range(0, B.size(0), batch_size):
                end_j = min(j + batch_size, B.size(0))
                diff = A[i:end_i].unsqueeze(1) - B[j:end_j].unsqueeze(0)
                dist_matrix[i:end_i, j:end_j] = torch.sqrt((diff ** 2).sum(2))

        return dist_matrix
    
    def torch_chamfer_distance(self, A, B, device='cuda'):
        """
        Compute the two-sided Chamfer distance between two sets of points, A and B, using PyTorch for GPU acceleration.
        """
        distances_A_to_B = self.torch_distance_matrix(A, B, device=device)
        distances_B_to_A = self.torch_distance_matrix(B, A, device=device)
        min_A_to_B = distances_A_to_B.min(dim=1)[0]
        min_B_to_A = distances_B_to_A.min(dim=1)[0]
        cd_one_sided_AB = min_A_to_B.mean()
        cd_one_sided_BA = min_B_to_A.mean()
        cd_two_sided = 0.5 * (cd_one_sided_AB + cd_one_sided_BA)
        return cd_one_sided_AB.item(), cd_one_sided_BA.item(), cd_two_sided.item()

    def torch_hausdorff_distance(self, A, B, device='cuda'):
        """
        Compute the Hausdorff distance between two sets of points, A and B, using PyTorch.
        """
        A = torch.tensor(A, device=device)
        B = torch.tensor(B, device=device)
        distances_A_to_B = self.torch_distance_matrix(A, B, device=device)
        distances_B_to_A = self.torch_distance_matrix(B, A, device=device)
        min_A_to_B = distances_A_to_B.min(dim=1)[0]
        min_B_to_A = distances_B_to_A.min(dim=1)[0]
        h_distance_AB = min_A_to_B.max()
        h_distance_BA = min_B_to_A.max()
        h_distance = max(h_distance_AB, h_distance_BA)
        return h_distance_AB, h_distance_BA, h_distance

    def generate_metrics(self, gt_path, input_path, metric_sample_pts):
        print('Generating metrics...')

        if self.dataset_name == 'srb':
            print(f'GT path: {gt_path}')
            print(type(gt_path))
            gt = trimesh.load(gt_path) # incase of SRB, gt is a point cloud
            input_pc = trimesh.load(input_path)
            pred_mesh = trimesh.load(os.path.join(self.base_exp_dir, "outputs/00040000_0.0.ply"))
            gt = self.normalize_pointcloud(np.asarray(gt.vertices))
            input_pc = self.normalize_pointcloud(np.asarray(input_pc.vertices))
            pred_pts = self.normalize_pointcloud(np.asarray(pred_mesh.vertices))

            gt_samples = gt[np.random.choice(gt.shape[0], metric_sample_pts, replace=False), :]
            input_samples = input_pc[np.random.choice(input_pc.shape[0], metric_sample_pts, replace=False), :]
            pred_pts_samples = pred_pts[np.random.choice(pred_pts.shape[0], metric_sample_pts, replace=False), :]

            cd_one_sided_gt_to_pred, cd_one_sided_pred_to_gt, cd_two_sided_gt_and_pred = self.torch_chamfer_distance(gt_samples, pred_pts_samples)
            hd_one_sided_gt_to_pred, hd_one_sided_pred_to_gt, hd_two_sided_gt_and_pred = self.torch_hausdorff_distance(gt_samples, pred_pts_samples)

            cd_one_sided_input_to_pred, cd_one_sided_pred_to_input, cd_two_sided_input_and_pred = self.torch_chamfer_distance(input_samples, pred_pts_samples)
            hd_one_sided_input_to_pred, hd_one_sided_pred_to_input, hd_two_sided_input_and_pred = self.torch_hausdorff_distance(input_samples, pred_pts_samples)
            
            # save metrics to a text file and round to 4 decimal places
            save_name = os.path.join(self.base_exp_dir, "metrics.txt")
            with open(save_name, 'w') as f:
                f.write(f'cd_one_sided_gt_to_pred: {cd_one_sided_gt_to_pred:.4f}\n')
                f.write(f'cd_one_sided_pred_to_gt: {cd_one_sided_pred_to_gt:.4f}\n')
                f.write(f'cd_two_sided_gt_and_pred: {cd_two_sided_gt_and_pred:.4f}\n')
                f.write(f'hd_one_sided_gt_to_pred: {hd_one_sided_gt_to_pred:.4f}\n')
                f.write(f'hd_one_sided_pred_to_gt: {hd_one_sided_pred_to_gt:.4f}\n')
                f.write(f'hd_two_sided_gt_and_pred: {hd_two_sided_gt_and_pred:.4f}\n')
                f.write(f'cd_one_sided_input_to_pred: {cd_one_sided_input_to_pred:.4f}\n')
                f.write(f'cd_one_sided_pred_to_input: {cd_one_sided_pred_to_input:.4f}\n')
                f.write(f'cd_two_sided_input_and_pred: {cd_two_sided_input_and_pred:.4f}\n')
                f.write(f'hd_one_sided_input_to_pred: {hd_one_sided_input_to_pred:.4f}\n')
                f.write(f'hd_one_sided_pred_to_input: {hd_one_sided_pred_to_input:.4f}\n')
                f.write(f'hd_two_sided_input_and_pred: {hd_two_sided_input_and_pred:.4f}\n')
            print('Metrics saved!')
        elif self.dataset_name == 'abc':
            print(f'GT path: {gt_path}')
            print(type(gt_path))
            gt = trimesh.load(gt_path) # incase of SRB, gt is a point cloud
            input_pc = trimesh.load(input_path)
            pred_mesh = trimesh.load(os.path.join(self.base_exp_dir, "outputs/00040000_0.0.ply"))
            gt = self.normalize_pointcloud(np.asarray(gt.vertices))
            input_pc = self.normalize_pointcloud(np.asarray(input_pc.vertices))
            pred_pts = self.normalize_pointcloud(np.asarray(pred_mesh.vertices))

            gt_samples = gt[np.random.choice(gt.shape[0], metric_sample_pts, replace=False), :]
            input_samples = input_pc[np.random.choice(input_pc.shape[0], metric_sample_pts, replace=False), :]
            pred_pts_samples = pred_pts[np.random.choice(pred_pts.shape[0], metric_sample_pts, replace=False), :]

            cd_one_sided_gt_to_pred, cd_one_sided_pred_to_gt, cd_two_sided_gt_and_pred = self.torch_chamfer_distance(gt_samples, pred_pts_samples)
            hd_one_sided_gt_to_pred, hd_one_sided_pred_to_gt, hd_two_sided_gt_and_pred = self.torch_hausdorff_distance(gt_samples, pred_pts_samples)

            cd_one_sided_input_to_pred, cd_one_sided_pred_to_input, cd_two_sided_input_and_pred = self.torch_chamfer_distance(input_samples, pred_pts_samples)
            hd_one_sided_input_to_pred, hd_one_sided_pred_to_input, hd_two_sided_input_and_pred = self.torch_hausdorff_distance(input_samples, pred_pts_samples)
            
            # save metrics to a text file and round to 4 decimal places
            save_name = os.path.join(self.base_exp_dir, "metrics.txt")
            with open(save_name, 'w') as f:
                f.write(f'cd_one_sided_gt_to_pred: {cd_one_sided_gt_to_pred:.4f}\n')
                f.write(f'cd_one_sided_pred_to_gt: {cd_one_sided_pred_to_gt:.4f}\n')
                f.write(f'cd_two_sided_gt_and_pred: {cd_two_sided_gt_and_pred:.4f}\n')
                f.write(f'hd_one_sided_gt_to_pred: {hd_one_sided_gt_to_pred:.4f}\n')
                f.write(f'hd_one_sided_pred_to_gt: {hd_one_sided_pred_to_gt:.4f}\n')
                f.write(f'hd_two_sided_gt_and_pred: {hd_two_sided_gt_and_pred:.4f}\n')
                f.write(f'cd_one_sided_input_to_pred: {cd_one_sided_input_to_pred:.4f}\n')
                f.write(f'cd_one_sided_pred_to_input: {cd_one_sided_pred_to_input:.4f}\n')
                f.write(f'cd_two_sided_input_and_pred: {cd_two_sided_input_and_pred:.4f}\n')
                f.write(f'hd_one_sided_input_to_pred: {hd_one_sided_input_to_pred:.4f}\n')
                f.write(f'hd_one_sided_pred_to_input: {hd_one_sided_pred_to_input:.4f}\n')
                f.write(f'hd_two_sided_input_and_pred: {hd_two_sided_input_and_pred:.4f}\n')
            print('Metrics saved!')
        elif self.dataset_name == 'famous':
            print(f'GT path: {gt_path}')
            print(type(gt_path))
            gt = trimesh.load(gt_path)
            input_pc = trimesh.load(input_path)
            pred_mesh = trimesh.load(os.path.join(self.base_exp_dir, "outputs/00040000_0.0.ply"))
            gt = self.normalize_pointcloud(np.asarray(gt.vertices))
            input_pc = self.normalize_pointcloud(np.asarray(input_pc.vertices))
            pred_pts = self.normalize_pointcloud(np.asarray(pred_mesh.vertices))

            gt_samples = gt[np.random.choice(gt.shape[0], metric_sample_pts, replace=False), :]
            input_samples = input_pc[np.random.choice(input_pc.shape[0], metric_sample_pts, replace=False), :]
            pred_pts_samples = pred_pts[np.random.choice(pred_pts.shape[0], metric_sample_pts, replace=False), :]

            cd_one_sided_gt_to_pred, cd_one_sided_pred_to_gt, cd_two_sided_gt_and_pred = self.torch_chamfer_distance(gt_samples, pred_pts_samples)
            hd_one_sided_gt_to_pred, hd_one_sided_pred_to_gt, hd_two_sided_gt_and_pred = self.torch_hausdorff_distance(gt_samples, pred_pts_samples)

            cd_one_sided_input_to_pred, cd_one_sided_pred_to_input, cd_two_sided_input_and_pred = self.torch_chamfer_distance(input_samples, pred_pts_samples)
            hd_one_sided_input_to_pred, hd_one_sided_pred_to_input, hd_two_sided_input_and_pred = self.torch_hausdorff_distance(input_samples, pred_pts_samples)
            
            # save metrics to a text file and round to 4 decimal places
            save_name = os.path.join(self.base_exp_dir, "metrics.txt")
            with open(save_name, 'w') as f:
                f.write(f'cd_one_sided_gt_to_pred: {cd_one_sided_gt_to_pred:.4f}\n')
                f.write(f'cd_one_sided_pred_to_gt: {cd_one_sided_pred_to_gt:.4f}\n')
                f.write(f'cd_two_sided_gt_and_pred: {cd_two_sided_gt_and_pred:.4f}\n')
                f.write(f'hd_one_sided_gt_to_pred: {hd_one_sided_gt_to_pred:.4f}\n')
                f.write(f'hd_one_sided_pred_to_gt: {hd_one_sided_pred_to_gt:.4f}\n')
                f.write(f'hd_two_sided_gt_and_pred: {hd_two_sided_gt_and_pred:.4f}\n')
                f.write(f'cd_one_sided_input_to_pred: {cd_one_sided_input_to_pred:.4f}\n')
                f.write(f'cd_one_sided_pred_to_input: {cd_one_sided_pred_to_input:.4f}\n')
                f.write(f'cd_two_sided_input_and_pred: {cd_two_sided_input_and_pred:.4f}\n')
                f.write(f'hd_one_sided_input_to_pred: {hd_one_sided_input_to_pred:.4f}\n')
                f.write(f'hd_one_sided_pred_to_input: {hd_one_sided_pred_to_input:.4f}\n')
                f.write(f'hd_two_sided_input_and_pred: {hd_two_sided_input_and_pred:.4f}\n')
            print('Metrics saved!')
        
    def save_tda_plot(self, cc_barcodes):
        figure_1 = plt.figure()
        cc_diag = gd.plot_persistence_diagram(cc_barcodes)
        file_path = os.path.join(self.base_exp_dir, f"cc_{self.iter_step}.jpg")
        plt.savefig(file_path)
        plt.close(figure_1)


    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def validate_mesh(self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None, save_sdf=False):
        self.file_backup('validate_mesh')
        bound_min = torch.tensor(self.dataset_np.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset_np.object_bbox_max, dtype=torch.float32)
        os.makedirs(os.path.join(self.base_exp_dir, 'outputs'), exist_ok=True)
        mesh, vertices, sdf = self.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, query_func=lambda pts: self.sdf_network.sdf(pts))

        mesh.export(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}.ply'.format(self.iter_step,str(threshold))))
        np.save(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}.npy'.format(self.iter_step,str(threshold))), np.asarray(vertices))
        if self.mode == 'validate_mesh' and threshold == 0.0:
            np.save(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}_sdf.npy'.format(self.iter_step,str(threshold))), np.asarray(sdf))


    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        print('Creating mesh with threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh, vertices, u
    
    def extract_fields(self, bound_min, bound_max, resolution, query_func):
        N = 32
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        return u

    def file_backup(self, mode):
        if mode in ['train', 'train_tda', 'curriculum'] and self._train_backup_done:
            print(f"{mode} backup already performed. Skipping.")
            return
        elif mode == 'validate_mesh' and self._validate_backup_done:
            print("Validate mesh backup already performed. Skipping.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir_name = f'recording_{mode}_{timestamp}'
        backup_dir = os.path.join(self.base_exp_dir, backup_dir_name)
        os.makedirs(backup_dir, exist_ok=True)

        file_types = ('.py', '.conf', '.txt', '.json', '.yaml', '.yml')

        dirs_to_backup = [
            'models',
            'utils',
            'run',
        ]

        dirs_to_exclude = [
            'outs',
            'outs_tda',
            'slurm',
            'wandb',
            'confs',
        ]

        for dir_path in dirs_to_backup:
            for root, dirs, files in os.walk(dir_path):
                # Exclude directories
                dirs[:] = [d for d in dirs if d not in dirs_to_exclude]

                for file in files:
                    if file.endswith(file_types):
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(backup_dir, os.path.relpath(src_path))
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copy2(src_path, dst_path)
                        print(f"Backed up: {src_path} -> {dst_path}")

        # Backup configuration file
        if hasattr(self, 'conf_path') and os.path.exists(self.conf_path):
            conf_backup_path = os.path.join(backup_dir, 'config.conf')
            shutil.copy2(self.conf_path, conf_backup_path)
            print(f"Backed up config: {self.conf_path} -> {conf_backup_path}")

        # Save basic run information
        info_file_path = os.path.join(backup_dir, 'run_info.txt')
        with open(info_file_path, 'w') as f:
            f.write(f"Mode: {mode}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Max Iterations: {self.maxiter}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Learning Rate: {self.learning_rate}\n")
            # Add any other relevant information
        print(f"Run info saved to: {info_file_path}")

        print(f"Backup completed in: {backup_dir}")
        
        if mode in ['train', 'train_tda', 'curriculum']:
            self._train_backup_done = True
        elif mode == 'validate_mesh':
            self._validate_backup_done = True

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        
        self.iter_step = checkpoint['iter_step']
            
    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
        
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/cubicalLayer.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcubes_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dir', type=str)
    parser.add_argument('--dataname', type=str)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf, args.mode)

    if args.mode in ['train', 'train_tda', 'curriculum']:
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.file_backup('validate_mesh')
        checkpoint_name = 'ckpt_{:0>6d}.pth'.format(40000)
        runner.load_checkpoint(checkpoint_name)
        threshs = [-0.001,-0.0025,-0.005,-0.01,-0.02,0.0,0.001,0.0025,0.005,0.01,0.02, -0.001953125, 0.001953125, 0.00390625, -0.00390625]
        for thresh in threshs:
            runner.validate_mesh(resolution=256, threshold=thresh, iter_step=40000)