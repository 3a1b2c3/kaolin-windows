# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import Callable
import logging as log

import torch
from torch.utils.data import Dataset

import kaolin.ops.spc as spc_ops
import wisp.ops.mesh as mesh_ops
import wisp.ops.spc as wisp_spc_ops
from wisp.datasets.transforms import Transform3d


class SDFDataset(Dataset):
    """Base class for single mesh datasets with points sampled only at a given octree sampling region.
    """

    def __init__(self, 
        sample_mode       : list = ['rand', 'rand', 'near', 'near', 'trace'],
        num_samples       : int = 100000,
        get_normals       : bool = False,
        sample_tex        : bool = False,
        matrix            : torch.Tensor = None #  (supporting per instance model matrix is on our short-term roadmap)
    ):
        """Construct dataset. This dataset also needs to be initialized.

        Args:
            sample_mode (list of str): List of different sample modes. 
                                       See `mesh_ops.point_sample` for more details.
            num_samples (int): Number of data points to keep in the working set. Whatever samples are not in 
                               the working set can be sampled with resample.
            get_normals (bool): If True, will also return normals (estimated by the SDF field).
            sample_tex (bool): If True, will also sample RGB from the nearest texture.
        """
        self.sample_mode = sample_mode
        self.num_samples = num_samples
        self.get_normals = get_normals
        self.sample_tex = sample_tex
        self.initialization_mode = None
        self.matrix = matrix
    '''  
            matrix: A tensor of shape (4, 4) or of shape (minibatch, 4, 4)
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
s                the specified `device` and `dtype`
                        M = [
                [Rxx, Ryx, Rzx, 0],
                [Rxy, Ryy, Rzy, 0],
                [Rxz, Ryz, Rzz, 0],
                [Tx,  Ty,  Tz,  1],
            ]
    ''' 
    def transform(self, points: torch.Tensor, matrix : torch.Tensor, normals: torch.Tensor =None):
        t = Transform3d(matrix=matrix)#.translate(torch.zeros(3,3))
        normals_transformed = normals
        if t:
            points_transformed = t.transform_points(points)    # => (N, P, 3)
            #print(points_transformed.shape, points.shape)
            if normals is not None:
                normals_transformed = t.transform_normals(normals)  # => (N, P, 3)
            return points_transformed, normals_transformed
        return points, normals
    '''
     Suppose that t is a Transform3d;
    then we can do the following:

    .. code-block:: python

        N = len(t)
        points = torch.randn(N, P, 3)
        normals = torch.randn(N, P, 3)
        points_transformed = t.transform_points(points)    # => (N, P, 3)
        normals_transformed = t.transform_normals(normals)  # => (N, P, 3)

    https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html
    pytorch3d.transforms.Transform3d(dtype: torch.dtype = torch.float32, device: Union[str, torch.device] = 'cpu', matrix: Optional[torch.Tensor] = None

        M = [
                [Rxx, Ryx, Rzx, 0],
                [Rxy, Ryy, Rzy, 0],
                [Rxz, Ryz, Rzz, 0],
                [Tx,  Ty,  Tz,  1],
            ]

from pytorch3d.transforms import Transform3D
transform3 = Transform3d().rotate(torch.stack([torch.eye(3)]*3)).translate(torch.zeros(3,3))
transform1 = Transform3d()
transform4 = transform1.stack(transform3)
print(len(transform3))
print(len(transform1))
print(len(transform4))
transform4.transform_points(torch.zeros(4,5,3))

N = len(t)
points = torch.randn(N, P, 3)
normals = torch.randn(N, P, 3)
points_transformed = t.transform_points(points)    # => (N, P, 3)
normals_transformed = t.transform_normals(normals)  # => (N, P, 3)

    \# example: rotate around y axis 90 degrees
relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(
    torch.tensor([0, np.pi/2, 0]), "XYZ"
)
            view_matrix[:3, -1] = torch.matmul(-view_matrix[:3, :3], poses[i][:3, -1])
    c = a@b #For dot product
         view_matrix = view_matrix @ retranslate @ rot_yaw @ translate
    def transform(self, matrix):
    """ Apply a transformation defined by a given matrix. """
    self.nodes = np.dot(self.nodes, matrix)
This uses the numpy function dot(), which multiplies two matrices. We now write a function in wireframe.py to create a translation matrix:

def translationMatrix(dx=0, dy=0, dz=0):
    """ Return matrix for translation along vector (dx, dy, dz). """
    
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [dx,dy,dz,1]])
    '''

    def init_from_mesh(self, dataset_path, mode_norm='aabb'):#'sphere', normalize=False):#DEBUG
        """Initializes the dataset by sampling SDFs from a mesh.

        Args:
            dataset_path (str): Path to OBJ file.
            mode_norm (str): The mode at which the mesh will be normalized in [-1, 1] space.
        """
        self.initialization_mode = "mesh"
        
        if self.sample_tex:
            out = mesh_ops.load_obj(dataset_path, load_materials=True)
            self.V, self.F, self.texv, self.texf, self.mats = out
        else:
            self.V, self.F = mesh_ops.load_obj(dataset_path)
        
        self.V, self.F = mesh_ops.normalize(self.V, self.F, mode_norm)
        """
                 self.coords = data["coords"]
                self.coords_center = data["coords_center"]
                self.coords_scale = data["coords_scale"]
        """
        self.mesh = self.V[self.F]
        self.resample()
        
    def init_from_grid(self, grid, samples_per_voxel=32):
        """Initializes the dataset by sampling SDFs from an OctreeGrid created from a mesh.

        Args:
            grid (wisp.models.grids.OctreeGrid): An OctreeGrid class initialized from mesh.
            samples_per_voxel (int): Number of data points to sample per voxel.
                                     Right now this class will sample upto 3x the points in reality since it will
                                     augment the samples with surface samples. Only used if the SDFs are sampled
                                     from a grid.
        """
        if grid.__class__.__name__ != "OctreeGrid" and "OctreeGrid" not in [pclass.__name__ for pclass in grid.__class__.__bases__]:
            raise Exception("Only the OctreeGrid class or derivatives are supported for this initialization mode")
    
        if not hasattr(grid, 'blas') and hasattr(grid.blas, 'V'):
            raise Exception("Only the OctreeGrid class or derivatives initialized from mesh are supported for this initialization mode")

        if self.get_normals:
            raise Exception("Grid initialization does not currently support normals")

        self.initialization_mode = "grid"
        self.samples_per_voxel = samples_per_voxel
        
        level = grid.active_lods[-1]

        # Here, corners mean "the bottom left corner of the voxel to sample from"
        corners = spc_ops.unbatched_get_level_points(grid.blas.points, grid.blas.pyramid, level)

        # Two pass sampling to figure out sample size
        self.pts_ = []
        for mode in self.sample_mode: 
            if mode == "rand":
                # Sample the points.
                self.pts_.append(wisp_spc_ops.sample_spc(corners, level, self.samples_per_voxel).cpu())
        for mode in self.sample_mode:
            if mode == "rand":
                pass
            elif mode == "near":
                self.pts_.append(mesh_ops.sample_near_surface(grid.blas.V.cuda(), 
                                                   grid.blas.F.cuda(), 
                                                   self.pts_[0].shape[0], 
                                                   variance=1.0/(2**level)).cpu())
            elif mode == "trace":
                self.pts_.append(mesh_ops.sample_surface(grid.blas.V.cuda(),
                                               grid.blas.F.cuda(),
                                               self.pts_[0].shape[0])[0].cpu())
            else:
                raise Exception(f"Sampling mode {mode} not implemented")

        # Filter out points which do not belong to the narrowband
        self.pts_ = torch.cat(self.pts_, dim=0)
        self.pidx = grid.query(self.pts_.cuda(), 0)
        self.pts_ = self.pts_[self.pidx>-1]
    
        # Sample distances and textures.
        if self.sample_tex:
            self.rgb_, self.hit_pts_, self.d_ = mesh_ops.closest_tex(grid.blas.V, grid.blas.F, 
                    grid.blas.texv, grid.blas.texf, grid.blas.mats, self.pts_)
        else:
            log.info(f"Computing SDFs for {self.pts_.shape[0]} samples (may take a while)..")
            self.d_ = mesh_ops.compute_sdf(grid.blas.V, grid.blas.F, self.pts_)
            assert(self.d_.shape[0] == self.pts_.shape[0])
        
        log.info(f"Total Samples: {self.pts_.shape[0]}")
        
        self.resample()

    def resample(self):
        """Resamples a new working set of SDFs.
        """
        
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")
        
        elif self.initialization_mode == "mesh":
            # Compute new sets of SDFs entirely

            self.nrm = None
            if self.get_normals:
                self.pts, self.nrm = mesh_ops.sample_surface(self.V, self.F, self.num_samples*len(self.sample_mode))
                self.nrm = self.nrm.cpu()
            else:
                self.pts = mesh_ops.point_sample(self.V, self.F, self.sample_mode, self.num_samples)

            if self.sample_tex:
                self.rgb, _, self.d = mesh_ops.closest_tex(self.V.cuda(), self.F.cuda(), 
                                               self.texv, self.texf, self.mats, self.pts)
                self.rgb = self.rgb.cpu()
            else:
                self.d = mesh_ops.compute_sdf(self.V.cuda(), self.F.cuda(), self.pts.cuda())   

            self.d = self.d.cpu()
            self.pts = self.pts.cpu()

            log.info(f"Resampling...")
        
        elif self.initialization_mode == "grid":
            # Choose a new working set of SDFs
            self.pts = self.pts_
            self.d = self.d_
            
            if self.sample_tex:
                self.rgb = self.rgb_
                self.hit_pts = self.hit_pts_

            _idx = torch.randperm(self.pts.shape[0] - 1, device='cuda')

            self.pts = self.pts[_idx]
            self.d = self.d[_idx]
            if self.sample_tex:
                self.rgb = self.rgb[_idx]
                self.hit_pts = self.hit_pts[_idx]

            total_samples = self.num_samples
            self.pts = self.pts[:total_samples]
            self.d = self.d[:total_samples]
            if self.sample_tex:
                self.rgb = self.rgb[:total_samples]
                self.rgb = self.rgb.cpu()
                self.hit_pts = self.hit_pts[:total_samples]
                self.hit_pts = self.hit_pts.cpu()

            self.d = self.d.cpu()
            self.pts = self.pts.cpu()

    # pts1[0] 
    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")
        
        # TODO(ttakikawa): Do this channel-wise instead
        out_normals = self.get_normals
        #out_normal = self.transform(out_normals, self.matrix)
        if out_normals and self.sample_tex:
            return self.pts[idx], self.d[idx], self.nrm[idx], self.rgb[idx]
        elif self.get_normals:
            return self.pts[idx], self.d[idx], self.nrm[idx]
        elif self.sample_tex:
            return self.pts[idx], self.d[idx], self.rgb[idx]
        else:
            if self.matrix.shape:
                out_pts, _n = self.transform(self.pts, self.matrix)
                #print(idx, type(out_pts), " __getitem_", self.pts.shape)
                return out_pts[idx], self.d[idx]
            return self.pts[idx], self.d[idx]

    def __len__(self):
        """Return length of dataset (number of _samples_)."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")

        return self.pts.size()[0]
