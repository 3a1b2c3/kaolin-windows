# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import os

from kaolin.ops.spc.points import points_to_morton, morton_to_points, unbatched_points_to_octree
from kaolin.rep.spc import Spc
from kaolin.ops.spc import points_to_morton, morton_to_points, unbatched_points_to_octree, unbatched_get_level_points

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
def test(level, features):
    # Avoid duplications if cells occupy more than one point
    unique, unique_keys, unique_counts = torch.unique(points.contiguous(), dim=0,
                                                      return_inverse=True, return_counts=True)

    # Create octree hierarchy
    morton, keys = torch.sort(points_to_morton(unique.contiguous()).contiguous())
    points = morton_to_points(morton.contiguous())
    octree = unbatched_points_to_octree(points, level, sorted=True)

    # Organize features for octree leaf nodes
    feat = None
    if features is not None:
        # Feature collision of multiple points sharing the same cell is consolidated here.
        # Assumes mean averaging
        feat_dtype = features.dtype
        is_fp = features.is_floating_point()

        # Promote to double precision dtype to avoid rounding errors
        feat = torch.zeros(unique.shape[0], features.shape[1], device=features.device).double()
        feat = feat.index_add_(0, unique_keys, features.double()) / unique_counts[..., None].double()
        if not is_fp:
            feat = torch.round(feat)
        feat = feat.to(feat_dtype)
        feat = feat[keys]

    # A full SPC requires octree hierarchy + auxilary data structures
    lengths = torch.tensor([len(octree)], dtype=torch.int32)   # Single entry batch
    return Spc(octrees=octree, lengths=lengths, features=feat)


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


# get features
def interpolate(grid_ops, coords, lod_idx, pidx=None):
    self.resolutions, self.codebook_bitwidth, lod_idx, self.codebook
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            lod_idx  (int): int specifying the index to ``active_lods`` 
            pidx (torch.LongTensor): Primitive indices of shape [batch]. Unused here.

        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        batch, num_samples, _ = coords.shape
        
        # add features THIS add sphere -------------
        feats = grid_ops.hashgrid(coords, self.resolutions, self.codebook_bitwidth, lod_idx, self.codebook)

        if self.multiscale_type == 'cat':#
            return feats.reshape(batch, num_samples, -1)
        elif self.multiscale_type == 'sum':
            return feats.reshape(batch, num_samples, len(self.resolutions), self.feature_dim).sum(-2)
        else:
            raise NotImplementedError