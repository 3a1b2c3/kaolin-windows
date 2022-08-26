# -*- coding: utf-8 -*-

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from contextlib import contextmanager
import os, sys

import torch

from kaolin.io import utils
from kaolin.io import obj

from wisp.core.primitives import PrimitivesPack
from wisp.ops.spc.conversions import mesh_to_spc#, mesh_to_octree
from wisp.accelstructs import OctreeAS
from wisp.models.grids.hash_grid import HashGrid
from wisp.models.nefs.nerf import NeuralRadianceField
from .spc_utils import get_level_points_from_octree


OPATH = os.path.normpath(os.path.join(__file__, "../../../data/test/obj/1.obj"))
OPATH = os.path.normpath(r"D:/workspace/INTEGRATION/kaolin-wisp/data/test/obj/1.obj")
'''
kaolin-wisp\wisp\ops\spc\conversions.py
def mesh_to_spc(vertices, faces, level):
    """Construct SPC from a mesh.

    Args:
        vertices (torch.FloatTensor): Vertices of shape [V, 3]
        faces (torch.LongTensor): Face indices of shape [F, 3]
        level (int): The level of the octree

    Returns:
        (torch.ByteTensor, torch.ShortTensor, torch.LongTensor, torch.BoolTensor):
        - the octree tensor
        - point hierarchy
        - pyramid
        - prefix
    """
    octree = mesh_to_octree(vertices, faces, level)
    points, pyramid, prefix = octree_to_spc(octree)
    return octree, points, pyramid, prefix 
            import wisp.ops.spc as wisp_spc_ops
        octree = wisp_spc_ops.mesh_to_octree(self.V, self.F, level)
'''
def get_obj(f=OPATH, scale=10):
    mesh = obj.import_mesh(f,
             with_materials=True, with_normals=True,
             error_handler=obj.skip_error_handler,
             heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_naive_homogenize)

    vertices = mesh.vertices.cpu()
    faces = mesh.faces.cpu()
    return vertices, faces 

def points_to_layer(vertices, points_layers_to_draw, colorT):  
    for i in range(0, len(vertices)):
        points_layers_to_draw.add_points(vertices[i], colorT)
    return points_layers_to_draw


def get_obj_layers(f=OPATH, color = [[1, 0, 0, 1], [0, 0, 1, 1]], scale=1, level=10):
    level = 1

    vertices, faces = get_obj(f, scale)
    if not len(vertices):
        return []
    """ 
    uvs_list=[mesh.uvs.cpu()],
    face_uvs_idx_list=[mesh.face_uvs_idx.cpu()],
    uvs = mesh.uvs.cuda().unsqueeze(0)
    face_uvs_idx = mesh.face_uvs_idx.cuda()
    face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
    face_uvs.requires_grad = False
    texture_map = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
                            requires_grad=True)
    diffuse_color = mesh.materials[0]['map_Kd'] 
    """
    layers_to_draw = [PrimitivesPack()]
    start = torch.FloatTensor()
    end = torch.FloatTensor()
    colorT = torch.FloatTensor(color[0])   
    for i in range(0, len(faces)):
        face = faces[i]
        start = vertices[face[0]]
        end = vertices[face[1]]
        layers_to_draw[0].add_lines(start, end, colorT)
        start = vertices[face[1]]
        end = vertices[face[2]]
        layers_to_draw[0].add_lines(start, end, colorT)
        start = vertices[face[2]]
        end = vertices[face[0]]
        layers_to_draw[0].add_lines(start, end, colorT)

     # add points
    points_layers_to_draw = [PrimitivesPack()]
    colorT = torch.FloatTensor(color[1])   
    # points_to_layer(vertices, points_layers_to_draw, colorT):
    for i in range(0, len(vertices)):
        points_layers_to_draw[0].add_points(vertices[i], colorT)
    return layers_to_draw, points_layers_to_draw


def get_OctreeAS(f=OPATH, levels=7):
    blasMesh = OctreeAS()
    blasMesh.init_from_mesh(OPATH, levels, True, num_samples=1000000)
    return blasMesh
    #octree, points, pyramid, prefix = mesh_to_spc(mesh.vertices, mesh.faces, level)
    spc = mesh_to_spc(vertices, faces, 10)

    return layers_to_draw, points_layers_to_draw, spc

#        feature_dim      : int   = 16,            feature_dim (int): The dimension of the features stored on the grid.
def get_HashGrid(max_grid_res=16, num_lods=16):
    h = HashGrid(16)
    # pipeline.nef.grid.init_from_geometric(16, args.max_grid_res, args.num_lods)
    h.init_from_geometric(16, max_grid_res, num_lods)
    #feats  = h.interpolate(coords, lod_idx, pidx=None)
    #feats = h.grid.interpolate(coords, lod_idx)#.reshape(-1, self.effective_feature_dim)
    #h.init_from_octree(base_lod, num_lods)
    return h

"""
    def rgba(self, coords, ray_d, pidx=None, lod_idx=None):
        Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): packed tensor of shape [batch, num_samples, 3]
            ray_d (torch.FloatTensor): packed tensor of shape [batch, 3]
            pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     Unused in the current implementation.
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor}:
                - RGB tensor of shape [batch, num_samples, 3] 
            


>>> a = torch.arange(4.)
>>> torch.reshape(a, (2, 2))
tensor([[ 0.,  1.],
        [ 2.,  3.]])
>>> b = torch.tensor([[0, 1], [2, 3]])
>>> torch.reshape(b, (-1,))
tensor([ 0,  1,  2,  3])
THCudaTensor_resize2d(tensor, oH, oW);
THCudaTensor_resize3d(tensor, 1, oH, oW); // no copy
THCudaTensor_resize2d(tensor, 1, oH*oW); // copy!
    example_2D_list = [
        [[ 0.9266, -0.9980,  0.9962]],
        [[ 0.9465, -0.9803,  0.9978]],
        [[ 0.9946, -0.9445,  0.9856]]
    ]

    coords = torch.tensor(example_2D_list)
    # __batch, num_samples 3 1
            coords (torch.FloatTensor): 3D coordinates of shape [batch, num_samples, 3]
b = a.unsqueeze(0) # adds one extra dimension of extent 1
If you add at the 1 position, it will be (3,1), which means 3 rows and 1 column.
<becomes 1 x 3 x326 x 326
    a.view(1,5)
    .unsqueeze(0) # adds one extra dimension of extent 1
    torch.Size([141467, 1, 3])  _1_batch, num_samples  141467 1
torch.Size([141467, 1, 3]) __batch, num_samples 141467 1 WPORKIN

"""
def get_features_HashGrid(coords, hashGrid, lod_idx=15):
    coords = coords[:100] #540
    print("before", coords.shape)
    coords = coords.reshape(2, 50, 3)
    batch, num_samples, _ = coords.shape
    print( batch, " batch: ", num_samples, "after",  coords.shape)
    #torch.Size([273, 1, 3]) __batch, num_samples 273 1
    #torch.Size([1, 141712, 3]) actual  1 141712


    batch, num_samples, _ = coords.shape
    print(coords.shape, " _1_batch, num_samples ",  batch, num_samples )      
    #feats = hashGrid.interpolate(coords, lod_idx, pidx=None)
    #feats = h.grid.interpolate(coords, lod_idx)#.reshape(-1, self.effective_feature_dim)
    #h.init_from_octree(base_lod, num_lods)
    # len( coords),len( ray_d)
    #    def rgba(self, coords, ray_d, pidx=None, lod_idx=None):
    hashGrid.freeze()
    #return feats


def get_NeuralRadianceField(f=OPATH, base_lod=1, num_lods=7):
    n = NeuralRadianceField()
    #def rgba(self, coords, ray_d, pidx=None, lod_idx=None):
    return n

def octree_to_layers(octree, level, colorT, layers_to_draw=None):
    points = get_level_points_from_octree(octree, level)
    if points:
        print("\n ___ points: ",  points.shape)
        if layers_to_draw is None:
            layers_to_draw = [PrimitivesPack()]
            points_to_layer(points, layers_to_draw[0], colorT)
        else:
            points_to_layer(points, layers_to_draw, colorT)
    return layers_to_draw

"""
'mats', 'max_level', 'octree', 'points', 'prefix', 'pyramid', 'query', 'raymarch', 'raytrace', 'texf', 'texv']
        ...max_level 1 torch.Size([9, 3])
        self.octree = octree # add mesh
        self.points, self.pyramid, self.prefix = wisp_spc_ops.octree_to_spc(self.octree)
        self.initialized = True
        self.max_level = self.pyramid.shape[-1] - 2
"""