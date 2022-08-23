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
import os

import torch

from kaolin.io import utils
from kaolin.io import obj

from wisp.core.primitives import PrimitivesPack
from wisp.ops.spc.conversions import mesh_to_spc#, mesh_to_octree
from wisp.accelstructs import OctreeAS
from .spc_utils import get_level_points_from_octree

OPATH = os.path.normpath(os.path.join(__file__, "../../../data/test/obj/1.obj"))

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
    blasMesh.init_from_mesh(OPATH, levels, True, samples=1000000)
    return blasMesh
    #octree, points, pyramid, prefix = mesh_to_spc(mesh.vertices, mesh.faces, level)
    spc = mesh_to_spc(vertices, faces, 10)

    return layers_to_draw, points_layers_to_draw, spc

def octree_to_layers(octree, level, colorT, layers_to_draw=None):
    points = get_level_points_from_octree(octree, level)
    print("\n ___ points: ",  points[0][0][0], points.shape)
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