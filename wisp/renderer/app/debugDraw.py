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
from typing import Optional, Type, Callable, Dict, List, Tuple

import torch
from glumpy import app, gloo, gl, ext

from kaolin.render.camera import Camera
from kaolin.io import utils
from kaolin.io import obj
import kaolin.ops.spc as spc_ops

from wisp.core.primitives import PrimitivesPack
from wisp.framework import WispState, watch
from wisp.renderer.core import RendererCore
from wisp.renderer.core.control import CameraControlMode, WispKey, WispMouseButton
from wisp.renderer.core.control import FirstPersonCameraMode, TrackballCameraMode, TurntableCameraMode
from wisp.renderer.gizmos import Gizmo, WorldGrid, AxisPainter, PrimitivesPainter
from wisp.renderer.gui import WidgetRendererProperties, WidgetGPUStats, WidgetSceneGraph, WidgetImgui
from wisp.ops.spc.conversions import mesh_to_spc
#from wisp.ops.pointcloud import create_pointcloud_from_images, normalize_pointcloud

from wisp.ops.test_mesh import get_obj_layers, get_OctreeAS, octree_to_layers, get_HashGrid, get_features_HashGrid
from wisp.ops.spc_utils import create_dual, octree_to_spc, get_level_points_from_octree
from wisp.ops.spc_formatting import describe_octree



'''
        # Normalize channel to [0, 1]
        channels_kit = self.state.graph.channels
        channel_info = channels_kit.get(selected_output_channel, create_default_channel())
        normalized_channel = channel_info.normalize_fn(rb_channel.clone())  # Clone to protect from modifications

        # To RGB (in normalized space)
        # TODO (operel): incorporate color maps
        channel_dim = normalized_channel.shape[-1]
        if channel_dim == 1:
            rgb = torch.cat((normalized_channel, normalized_channel, normalized_channel), dim=-1)
        elif channel_dim == 2:
            rgb = torch.cat((normalized_channel, normalized_channel, torch.zeros_like(normalized_channel)), dim=-1)
        elif channel_dim == 3:
            rgb = normalized_channel
        else:
            raise ValueError('Cannot display channels with more than 3 dimensions over the canvas.')
            "imgs": rgbs, "masks": masks,

 Since the occupancy information is [compressed]
 (https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.spc.html?highlight=spc#octree) and 
 [packed](https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html?highlight=packed#packed), accessing level-specific information consistently involves
cumulative summarization of the number of "1" bits. <br>
It makes sense to calculate this information once and then cache it. <br>
The `pyramid` field does exactly that: it keeps summarizes the number of occupied cells per level, and their cumsum, for fast level-indexing.
'''


def getDebugCloud(dataSet, wisp_state, level=3):
    #print("\n____initwisp_state.channels ", wisp_state.graph.channels["rgb"])
    c = dataSet.coords
   # print("c:", c)
    rays = dataSet.data['rays']
    rgbs = dataSet.data["imgs"] 
    masks = dataSet.data["masks"]
    depths = dataSet.data["masks"] #wisp_state.graph.channels["depth"] #Channel depth is usually a distance to the surface hit point
    # add points
    dpoints_layers_to_draw = [PrimitivesPack()]
    points_layers_to_draw = [PrimitivesPack()]
    colorT = torch.FloatTensor([0, 1, 1, 1]) 
    #octree_to_layers(wisp_state.graph.neural_pipelines['test-ngp-nerf-interactive'].nef.grid.blas.octree, level, colorT, dpoints_layers_to_draw[0])

    N = rays.origins.shape[0]
    for j in range(0, len(rays)):
        # print(rays.shape, "rays[j]", rays[j].shape)
        for i in range(0, len(rays[j].origins)):
            #points.append(rays[i].origins) 
            #points_layers_to_draw[0].add_points(points[i], colorT)
            #i, j:  39999 0 200 40000
            #print("i, j: ", i, j, len(rays),  len(rays[j].origins)) #i, j:  0 0 200 40000
            points_layers_to_draw[j].add_lines(rays[j][i].origins, rays[j][i].origins + rays[j][i].dirs, colorT)
        break

    '''
    points = wisp_state.graph.neural_pipelines['test-ngp-nerf-interactive'].nef.grid.dense_points
    #points1 = point_hierarchy1[pyramid1[-1, 1]:pyramid1[-1, 1] + pyramid1[-1, 0]
    colorT = torch.FloatTensor([1, 1, 1, 1]) 
    for i in range(0, len(points)): 
        dpoints_layers_to_draw[0].add_points(points[i], colorT)
    '''
    # wisp_state.graph.neural_pipelines['test-ngp-nerf-interactive'].nef.grid.occupancy
    return points_layers_to_draw, dpoints_layers_to_draw

  
""" 
        rgbs (list of torch.FloatTensor): List of RGB tensors of shape [H, W, 3].
        masks (list of torch.FloatTensor): List of mask tensors of shape [H, W, 1].
        rays (list of wisp.core.Rays): List of rays.origins and rays.dirs of shape [H, W, 3].
        depths (list of torch.FloatTensor): List of depth tensors of shape [H, W, 1].


 print(" d1: ", dir( self.dataset))
        print("__________dataset1:", self.dataset.keys()) #__________dataset: dict_keys(['imgs', 'masks', 'rays', 'cameras'])
        self.dataset['rays']
        print("\n____initwisp_state.channels ", wisp_state.graph.channels.keys())
        #def create_pointcloud_from_images(rgbs, masks, rays, depths):
        sys.exit()
        '''
        print("\n____initwisp_state.channels ", wisp_state.graph.channels.keys())
        print("\n____init_wisp_state.neural_pipelines1", dir(wisp_state.graph.channels["depth"])) # Channel
        print("\n____init_wisp_state.neural_pipelines2", wisp_state.graph.channels["hit"])
        print("\n____init_wisp_state.nesural_pipelines3", wisp_state.graph.neural_pipelines['test-ngp-nerf-interactive'].named_parameters)
        print("\n____init_wisp_state.neural_pipelines4", dir(wisp_state.graph.neural_pipelines['test-ngp-nerf-interactive'].nef))
        print("\n____init_wisp_state.neural_pipelines6", dir(wisp_state.graph.neural_pipelines['test-ngp-nerf-interactive'].nef.grid.dense_points))
        print("\n____init_wisp_state.neural_pipelines7", wisp_state.graph.neural_pipelines['test-ngp-nerf-interactive'].nef.grid.dense_points)
        print("\n____init_wisp_state.neural_pipelines8", wisp_state.graph.neural_pipelines['test-ngp-nerf-interactive'].nef.grid.occupancy) #([20., 20., 20.,  ..., 20., 20., 20.])
        #print("\n____init_wisp_state.neural_pipelines5", wisp_state.graph.neural_pipelines['test-ngp-nerf-interactive'].state_dict)
        #rays
        #create_pointcloud_from_images(wisp_state.graph.channels["rgb"])

          self.dense_points = spc_ops.unbatched_get_level_points(self.blas.points, self.blas.pyramid, self.blas_level).clone()
        self.num_cells = self.dense_points.shape[0]
        self.occupancy


', 'buffers', 'children', 'cpu', 'cuda', 'double', 'dump_patches', 'eval', 'extra_repr', 
'float', 'forward', 'get_buffer', 
'get_extra_state', 'get_parameter', 'get_submodule', 'half', 'ipu', 'load_state_dict', 'modules', 'named_buffers', 'named_children',
 'named_modules', 'named_parameters', 'nef', 'parameters', 'register_backward_hook', 'register_buffer', 'register_forward_hook',
  'register_forward_pre_hook', 'register_full_backward_hook', 'register_load_state_dict_post_hook', 
'register_module', 'register_parameter', 'requires_grad_', 'set_extra_state', 'share_memory', 'state_dict', 
'to', 'to_empty', 
'tracer', 'train', 'training', 'type', 'xpu', 'zero_grad']
            self.render_res_x = None
        self.render_res_y = None
        self.output_width = None
        self.output_height = None
        self._last_state = dict()
        self._data_layers 
        
         (nef): NeuralRadianceField(  self._data_layers
    (grid): HashGrid(
      (codebook): ParameterList(
             (codebook): ParameterList(
          (0): Parameter containing: [torch.cuda.FloatTensor of size 4096x2 (GPU 0)]
          (1): Parameter containing: [torch.cuda.FloatTensor of size 9261x2 (GPU 0)]
          (2): Parameter containing: [torch.cuda.FloatTensor of size 24389x2 (GPU 0)]
          (3): Parameter containing: [torch.cuda.FloatTensor of size 59319x2 (GPU 0)]
          (4): Parameter containing: [torch.cuda.FloatTensor of size 148877x2 (GPU 0)]
          (5): Parameter containing: [torch.cuda.FloatTensor of size 373248x2 (GPU 0)]
          (6): Parameter containing: [torch.cuda.FloatTensor of size 524288x2 (GPU 0)]
          (7): Parameter containing: [torch.cuda.FloatTensor of size 524288x2 (GPU 0)]
          (8): Parameter containing: [torch.cuda.FloatTensor of size 524288x2 (GPU 0)]
          (9): Parameter containing: [torch.cuda.FloatTensor of size 524288x2 (GPU 0)]
          (10): Parameter containing: [torch.cuda.FloatTensor of size 524288x2 (GPU 0)]
          (11): Parameter containing: [torch.cuda.FloatTensor of size 524288x2 (GPU 0)]
          (12): Parameter containing: [torch.cuda.FloatTensor of size 524288x2 (GPU 0)]
          (13): Parameter containing: [torch.cuda.FloatTensor of size 524288x2 (GPU 0)]
          (14): Parameter containing: [torch.cuda.FloatTensor of size 524288x2 (GPU 0)]
          (15): Parameter containing: [torch.cuda.FloatTensor of size 524288x2 (GPU 0)]
      )
"""
