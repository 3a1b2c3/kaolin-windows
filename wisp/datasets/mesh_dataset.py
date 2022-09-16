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


class MeshDataset(Dataset):
    """Base class for single mesh datasets with points sampled only at a given octree sampling region.
    """
    minV = 0
    maxV = 0

    def __init__(self, 
        sample_mode       : list = ['rand', 'rand', 'near', 'near', 'trace'],
        num_samples       : int = 100000,
        get_normals       : bool = False,
        sample_tex        : bool = False,
        normalize         : bool = True,
        transform         : Callable = None,
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

        self.mesh = self.V[self.F]
        self.resample()


    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")
        # TODO(ttakikawa): Do this channel-wise instead
        if self.get_normals and self.sample_tex:
            return self.pts[idx], self.d[idx], self.nrm[idx], self.rgb[idx]
        elif self.get_normals:
            return self.pts[idx], self.d[idx], self.nrm[idx]
        elif self.sample_tex:
            return self.pts[idx], self.d[idx], self.rgb[idx]
        else:
            return self.pts[idx], self.d[idx]

    def __len__(self):
        """Return length of dataset (number of _samples_)."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")

        return self.pts.size()[0]

    def transform(self, points: torch.Tensor, matrix : torch.Tensor, normals: torch.Tensor=None):
        t = Transform3d(matrix=matrix)#.translate(torch.zeros(3,3))
        normals_transformed = normals
        if t:
            points_transformed = t.transform_points(points)    # => (N, P, 3)
            #print(points_transformed.shape, points.shape)
            if normals is not None:
                normals_transformed = t.transform_normals(normals)  # => (N, P, 3)
            return points_transformed, normals_transformed
        return points, normals