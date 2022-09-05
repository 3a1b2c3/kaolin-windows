# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch

def normalize(
    V : torch.Tensor,
    F : torch.Tensor,
    mode : str):
    """Normalizes a mesh.

    Args:
        V (torch.FloatTensor): Vertices of shape [V, 3]
        F (torch.LongTensor): Faces of shape [F, 3]
        mode (str): Different methods of normalization.

      normalized space [-1, 1].

    Returns:
        (torch.FloatTensor, torch.LongTensor):
        - Normalized Vertices
        - Faces
    """

    if mode == 'sphere':

        V_max, _ = torch.max(V, dim=0)
        V_min, _ = torch.min(V, dim=0)
        V_center = (V_max + V_min) / 2.
        V = V - V_center

        # Find the max distance to origin
        max_dist = torch.sqrt(torch.max(torch.sum(V**2, dim=-1)))
        V_scale = 1. / max_dist
        V *= V_scale
        return V, F

    elif mode == 'aabb':
        
        V_min, _ = torch.min(V, dim=0)
        V = V - V_min

        max_dist = torch.max(V)
        V *= 1.0 / max_dist

        V = V * 2.0 - 1.0

        return V, F

    elif mode == 'planar':
        
        V_min, _ = torch.min(V, dim=0)
        V = V - V_min

        x_max = torch.max(V[...,0])
        z_max = torch.max(V[...,2])

        V[...,0] *= 1.0 / x_max
        V[...,2] *= 1.0 / z_max

        max_dist = torch.max(V)
        V[...,1] *= 1.0 / max_dist
        #V *= 1.0 / max_dist

        V = V * 2.0 - 1.0

        y_min = torch.min(V[...,1])

        V[...,1] -= y_min

        return V, F

    elif mode == 'none':

        return V, F


def unnormalize(
    V : torch.Tensor,
    F : torch.Tensor,
    V_min,
    max_dist):
    """Normalizes a mesh.
        q =  ( 2.0*(x-min_x)/(max_x-min_x) - 1.0
       2.0*(y-min_y)/(max_y-min_y) - 1.0
       2.0*(z-min_z)/(max_z-min_z) - 1.0 )
now q equals p translated to the interval (-1,-1,-1) -- (+1,+1,+1).

Do whatever you need to do on this intermediate grid.

Convert all coordinates q = (xx, yy, zz) back to the original grid by doing the inverse operation:

p =  ( (xx+1.0)*(max_x-min_x)/2.0 + min_x
       (yy+1.0)*(max_y-min_y)/2.0 + min_y
       (zz+1.0)*(max_z-min_z)/2.0 + min_z
    """
    f = (max_dist - V_min)/2.0
    V = V * f

    return V, F






