# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging as log
import os

from wisp.utils import PsDebugger, PerfTimer
from wisp.ops.spc import sample_spc

import wisp.ops.spc as wisp_spc_ops
import wisp.ops.grid as grid_ops

from wisp.models.grids import BLASGrid
from wisp.models.decoders import BasicDecoder

import kaolin.ops.spc as spc_ops
from kaolin.ops.spc import points_to_morton, morton_to_points, unbatched_points_to_octree

from wisp.accelstructs import OctreeAS

OPATH = os.path.normpath(os.path.join(__file__, "../../../../data/test/obj/1.obj"))

'''
# indexing by masking follow naturally the morton order
What is Morton Order (Z-Order, Lebesgue Curve)
morton (torch.LongTensor): The morton codes of quantized 3D points,
of shape :math:`(\text{num_points})`.
// Uses the morton buffer to construct an octree. It is the user's responsibility to allocate
// space for these zero-init buffers, and for the morton buffer, to allocate the buffer from the back 
// with the occupied positions.
'''

def build(octree):
    points, pyramid, prefix = spc_utils.octree_to_spc(self.octree)
    points_dual, self.pyramid_dual = spc_utils.create_dual(self.points, self.pyramid)

def mergeOctrees(points_hierarchy1, points_hierarchy2, pyramid1, pyramid2,
            features1, features2, level):

    points1 = points_hierarchy1[pyramid1[-1, 1]:pyramid1[-1, 1] + pyramid1[-1, 0]]
    points2 = points_hierarchy2[pyramid2[-1, 1]:pyramid2[-1, 1] + pyramid2[-1, 0]]
    all_points = torch.cat([points_hierarchy1, points_hierarchy2], dim=0)
    unique, unique_keys, unique_counts = torch.unique(all_points.contiguous(), dim=0,
                                                                                        return_inverse=True, return_counts=True)
    morton, keys = torch.sort(points_to_morton(unique.contiguous()).contiguous())
    points = morton_to_points(morton.contiguous())
    merged_octree = unbatched_points_to_octree(points, level, sorted=True)

    all_features = torch.cat([features1, features2], dim=0)
    feat = torch.zeros(unique.shape[0], all_features.shape[1], device=all_features.device).double()
    # Here we just do an average when both octrees have features on the same coordinate
    feat = feat.index_add_(0, unique_keys, all_features.double()) / unique_counts[..., None].double()
    feat = feat.to(all_features.dtype)
    merged_features = feat[keys]
    return merged_octree, merged_features

class HashGrid(BLASGrid):
    """This is a feature grid where the features are defined in a codebook that is hashed.
    """

    def __init__(self, 
        feature_dim        : int,
        interpolation_type : str   = 'linear',
        multiscale_type    : str   = 'cat',
        feature_std        : float = 0.0,
        feature_bias       : float = 0.0,
        codebook_bitwidth  : int   = 16,
        blas_level         : int   = 7, # octree
        **kwargs
    ):
        """Initialize the hash grid class.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            interpolation_type (str): The type of interpolation function.
            multiscale_type (str): The type of multiscale aggregation. Usually 'sum' or 'cat'.
                                   Note that 'cat' will change the decoder input dimension.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The mean of the Gaussian distribution.
            codebook_bitwidth (int): The bitwidth of the codebook.
            blas_level (int): The level of the octree to be used as the BLAS.
        
        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.interpolation_type = interpolation_type
        self.multiscale_type = multiscale_type

        self.feature_std = feature_std
        self.feature_bias = feature_bias
        self.codebook_bitwidth = codebook_bitwidth
        self.blas_level = blas_level

        self.kwargs = kwargs
    
        ############ here
        self.blas = OctreeAS()
        self.blasMesh = OctreeAS()
        self.blasMesh.init_from_mesh(OPATH, 1, True, samples=1000000)

        # pointcloud_to_octree(pointcloud, level, attributes=None, dilate=0):
        self.blas.init_dense(self.blas_level)
        #    return point_hierarchy[pyramid[1, level]:pyramid[1, level + 1]]
        self.dense_points = spc_ops.unbatched_get_level_points(self.blas.points, self.blas.pyramid, self.blas_level).clone()
        self.num_cells = self.dense_points.shape[0]
        self.occupancy = torch.ones(self.num_cells) * 20.0 #check pyramide

    def init_from_octree(self, base_lod, num_lods):
        """Builds the multiscale hash grid with an octree sampling pattern.
        """
        octree_lods = [base_lod + x for x in range(num_lods)]
        resolutions = [2**lod for lod in octree_lods]
        self.init_from_resolutions(resolutions)

    def init_from_geometric(self, min_width, max_width, num_lods, vertices=None, faces=None):
        """Build the multiscale hash grid with a geometric sequence.

        This is an implementation of the geometric multiscale grid from 
        instant-ngp (https://nvlabs.github.io/instant-ngp/).

        See Section 3 Equations 2 and 3 for more details.
        """
        b = np.exp((np.log(max_width) - np.log(min_width)) / num_lods) 
        resolutions = [int(np.floor(min_width*(b**l))) for l in range(num_lods)]
        self.init_from_resolutions(resolutions, vertices, faces)
    
    def init_from_resolutions(self, resolutions, vertices=None, faces=None):
        """Build a multiscale hash grid from a list of resolutions.
        """
        self.resolutions = resolutions
        self.num_lods = len(resolutions)
        self.active_lods = [x for x in range(self.num_lods)]
        self.max_lod = self.num_lods - 1

        log.info(f"Active Resolutions: {self.resolutions}")
        
        self.codebook_size = 2**self.codebook_bitwidth

        self.codebook = nn.ParameterList([])
        for res in resolutions:
            num_pts = res**3
            fts = torch.zeros(min(self.codebook_size, num_pts), self.feature_dim)
            fts += torch.randn_like(fts) * self.feature_std
            self.codebook.append(nn.Parameter(fts))

    def freeze(self):
        """Freezes the feature grid.
        """
        self.codebook.requires_grad_(False)

    def interpolate(self, coords, lod_idx, pidx=None):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            lod_idx  (int): int specifying the index to ``active_lods`` 
            pidx (torch.LongTensor): Primitive indices of shape [batch]. Unused here.

        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        timer = PerfTimer(activate=False, show_memory=False)

        batch, num_samples, _ = coords.shape
        
        #print("coords.shape1 ", coords.shape,  coords[0][0])
        # coords.shape1  torch.Size([840, 1, 3]) tensor([[ 0.7880, -0.9838,  0.9873]], device='cuda:0')
        feats = grid_ops.hashgrid(coords, self.resolutions, self.codebook_bitwidth, lod_idx, self.codebook)

        if self.multiscale_type == 'cat':
            return feats.reshape(batch, num_samples, -1)
        elif self.multiscale_type == 'sum':
            return feats.reshape(batch, num_samples, len(self.resolutions), self.feature_dim).sum(-2)
        else:
            raise NotImplementedError

    def raymarch(self, rays, level=None, num_samples=64, raymarch_type='voxel'):
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: the OctreeGrid raymarch samples over the coarsest LOD where features are available.
        """
        return self.blas.raymarch(rays,
                                  level=self.blas_level, num_samples=num_samples, raymarch_type=raymarch_type)
