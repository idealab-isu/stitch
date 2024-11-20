
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import trimesh
import gudhi as gd

def search_nearest_point(point_batch, point_gt):
    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

    distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12) 
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

    return dis_idx

def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = [list(map(float, line.strip().split()[:3])) for line in lines]
    pointcloud = np.array(data)  # Convert list of lists to a NumPy array

    return pointcloud

def process_data(data_dir, dataname, sigma_val):
    if os.path.exists(os.path.join(data_dir, dataname) + '.ply'):
        pointcloud = trimesh.load(os.path.join(data_dir, dataname) + '.ply').vertices
        pointcloud = np.asarray(pointcloud)
    elif os.path.exists(os.path.join(data_dir, dataname) + '.xyz'):
        pointcloud = read_xyz(os.path.join(data_dir, dataname + '.xyz'))
        print('input point cloud shape:', pointcloud.shape)
    elif os.path.exists(os.path.join(data_dir, dataname) + '.npy'):
        pointcloud = np.load(os.path.join(data_dir, dataname) + '.npy')
    else:
        print('Only support .xyz ,.ply, and .npy data. Please adjust your data.')
        exit()
    shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale

    POINT_NUM = pointcloud.shape[0] // 60
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60
    QUERY_EACH = 1000000//POINT_NUM_GT

    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False)
    pointcloud = pointcloud[point_idx,:]
    # print(np.max(pointcloud[:,0]),np.max(pointcloud[:,1]),np.max(pointcloud[:,2]),np.min(pointcloud[:,0]),np.min(pointcloud[:,1]),np.min(pointcloud[:,2]))
    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud,100,axis=0):
        d = ptree.query(p,51)
        sigmas.append(d[0][:,-1])
            
    sigmas = np.concatenate(sigmas)
    # sigmas = sigmas*2
    sample = []
    sample_near = []

    for i in range(QUERY_EACH):
        scale = 0.25 * np.sqrt(POINT_NUM_GT / 20000)
        # scale = sigma_val
        print('scale: ', scale)
        tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        tt = tt.reshape(-1,POINT_NUM,3)

        sample_near_tmp = []
        for j in range(tt.shape[0]):
            nearest_idx = search_nearest_point(torch.tensor(tt[j]).float().cuda(), torch.tensor(pointcloud).float().cuda())
            nearest_points = pointcloud[nearest_idx]
            nearest_points = np.asarray(nearest_points).reshape(-1,3)
            sample_near_tmp.append(nearest_points)
        sample_near_tmp = np.asarray(sample_near_tmp)
        sample_near_tmp = sample_near_tmp.reshape(-1,3)
        sample_near.append(sample_near_tmp)
        
    sample = np.asarray(sample)
    print('sample shape:', np.asarray(sample).shape)
    sample_near = np.asarray(sample_near)
    print('sample_near shape:', np.asarray(sample_near).shape)

    np.savez(os.path.join(data_dir, dataname)+'.npz', sample = sample, point = pointcloud, sample_near = sample_near)


class DatasetNP:
    def __init__(self, conf, dataname, dataset_name, sigma_val):
        super(DatasetNP, self).__init__()
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.np_data_name = dataname + '.npz'

        if dataset_name == 'srb':
            if os.path.exists(os.path.join(self.data_dir + '/scans/', self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir + '/scans/', dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir + '/scans/', self.np_data_name))
        
        elif dataset_name == 'srb_modified':
            if os.path.exists(os.path.join(self.data_dir + '/modified_scans/', self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir + '/modified_scans/', dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir + '/modified_scans/', self.np_data_name))
        
        elif dataset_name == 'srb_randomAblationScalar':
            if os.path.exists(os.path.join(self.data_dir + '/scans/', self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir + '/scans/', dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir + '/scans/', self.np_data_name))

        elif dataset_name == 'famous':
            if os.path.exists(os.path.join(self.data_dir + '/04_pts/', self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir + '/04_pts/', dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir + '/04_pts/', self.np_data_name))

        elif dataset_name == 'abc':
            if os.path.exists(os.path.join(self.data_dir + '/04_pts/', self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir + '/04_pts/', dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir + '/04_pts/', self.np_data_name))
        elif dataset_name == 'dfaust':
            if os.path.exists(os.path.join(self.data_dir, self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir, dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir, self.np_data_name))
        elif dataset_name == 'custom':
            if os.path.exists(os.path.join(self.data_dir, self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir, dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir, self.np_data_name))
        


        
        self.point = np.asarray(load_data['sample_near']).reshape(-1,3)
        self.sample = np.asarray(load_data['sample']).reshape(-1,3)
        self.point_gt = np.asarray(load_data['point']).reshape(-1,3)
        self.sample_points_num = self.sample.shape[0]-1

        self.object_bbox_min = np.array([np.min(self.point[:,0]), np.min(self.point[:,1]), np.min(self.point[:,2])]) -0.05
        self.object_bbox_max = np.array([np.max(self.point[:,0]), np.max(self.point[:,1]), np.max(self.point[:,2])]) +0.05
        print('Data bounding box:',self.object_bbox_min,self.object_bbox_max)
    
        self.point = torch.from_numpy(self.point).to(self.device).float()
        self.sample = torch.from_numpy(self.sample).to(self.device).float()
        self.point_gt = torch.from_numpy(self.point_gt).to(self.device).float()
       
        print('NP Load data: End')

    def np_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num//10, batch_size, replace = False)
        index = index_fine * 10 + index_coarse
        points = self.point[index]
        sample = self.sample[index]
        
        return points, sample, self.point_gt
        

class DatasetNP_TDA:
    def __init__(self, conf, dataname, dataset_name, persistence_radius, persistence_dim, sigma_val, save_dir):
        super(DatasetNP_TDA, self).__init__()
        print('Using DatasetNP_TDA...')
        self.device = torch.device('cuda')
        self.conf = conf
        self.save_dir = save_dir

        self.data_dir = conf.get_string('data_dir')
        self.np_data_name = dataname + '.npz'

        if dataset_name == 'srb':
            if os.path.exists(os.path.join(self.data_dir + '/scans/', self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir + '/scans/', dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir + '/scans/', self.np_data_name))
        
        elif dataset_name == 'srb_modified':
            if os.path.exists(os.path.join(self.data_dir + '/modified_scans/', self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir + '/modified_scans/', dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir + '/modified_scans/', self.np_data_name))
        
        elif dataset_name == 'srb_randomAblationScalar':
            if os.path.exists(os.path.join(self.data_dir + '/scans/', self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir + '/scans/', dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir + '/scans/', self.np_data_name))

        elif dataset_name == 'famous':
            if os.path.exists(os.path.join(self.data_dir + '/04_pts/', self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir + '/04_pts/', dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir + '/04_pts/', self.np_data_name))

        elif dataset_name == 'abc':
            if os.path.exists(os.path.join(self.data_dir + '/04_pts/', self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir + '/04_pts/', dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir + '/04_pts/', self.np_data_name))
        elif dataset_name == 'dfaust':
            if os.path.exists(os.path.join(self.data_dir, self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir, dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir, self.np_data_name))
        elif dataset_name == 'custom':
            if os.path.exists(os.path.join(self.data_dir, self.np_data_name)):
                print('Data existing. Loading data...')
            else:
                print('Data not found. Processing data...')
                process_data(self.data_dir, dataname, sigma_val)
            load_data = np.load(os.path.join(self.data_dir, self.np_data_name))
        
        self.point = np.asarray(load_data['sample_near']).reshape(-1,3)
        self.sample = np.asarray(load_data['sample']).reshape(-1,3)
        self.point_gt = np.asarray(load_data['point']).reshape(-1,3)
        self.sample_points_num = self.sample.shape[0]-1

        self.object_bbox_min = np.array([np.min(self.point[:,0]), np.min(self.point[:,1]), np.min(self.point[:,2])]) -0.05
        self.object_bbox_max = np.array([np.max(self.point[:,0]), np.max(self.point[:,1]), np.max(self.point[:,2])]) +0.05
        print('Data bounding box:',self.object_bbox_min,self.object_bbox_max)
    
        self.point = torch.from_numpy(self.point).to(self.device).float()
        self.sample = torch.from_numpy(self.sample).to(self.device).float()
        self.point_gt = torch.from_numpy(self.point_gt).to(self.device).float()
        self.all_dim_barcodes, self.gt_pers_barcodes = self.compute_pers_diagram(persistence_radius, self.point_gt, persistence_dim) ##### newly added for persistance diagram #####
        
        # save ground truth persistence diagram
        self.plot_pers_diagram(self.gt_pers_barcodes)


        
        print('NP Load data: End')
    #### newly added for persistance diagram #####
    def compute_pers_diagram(self, radius_thresh, pc, maxdim):
        skeleton = gd.RipsComplex(points=pc.detach().cpu().numpy(), max_edge_length=radius_thresh)
        simplex_tree = skeleton.create_simplex_tree(max_dimension=maxdim)
        barcodes = simplex_tree.persistence()
        all_dims_barcodes = []
        for dim in range(maxdim+1):
            dim_barcodes = simplex_tree.persistence_intervals_in_dimension(dim)
            all_dims_barcodes.append(dim_barcodes)
        # zero_dim_barcodes_pd = simplex_tree.persistence_intervals_in_dimension(0)
        # one_dim_barcodes_pd = simplex_tree.persistence_intervals_in_dimension(1)
        # two_dim_barcodes_pd = simplex_tree.persistence_intervals_in_dimension(2)
        # all_dims_barcodes = [zero_dim_barcodes_pd, one_dim_barcodes_pd, two_dim_barcodes_pd]
        return all_dims_barcodes, barcodes
        # return zero_dim_barcodes_pd
    
    def plot_pers_diagram(self, zero_dim_barcodes_pd):
        figure = plt.figure()
        gt_pers_diag = gd.plot_persistence_diagram(zero_dim_barcodes_pd)
        file_path = os.path.join(self.save_dir, "gt_persistent_diagram.jpg")
        plt.savefig(file_path)
        plt.close(figure)
    ##############################################

    def np_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num//10, batch_size, replace = False)
        index = index_fine * 10 + index_coarse
        points = self.point[index]
        sample = self.sample[index]
        return points, sample, self.point_gt, self.all_dim_barcodes ##### newly added for persistance diagram #####
