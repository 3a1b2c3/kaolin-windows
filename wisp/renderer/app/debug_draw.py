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

from kaolin.render.camera import Camera
from kaolin.io import utils, obj
import kaolin.ops.spc as spc_ops

from wisp.core.primitives import PrimitivesPack
from wisp.framework import WispState, watch
from wisp.renderer.gizmos import PrimitivesPainter
from wisp.ops.spc.conversions import mesh_to_spc
#from wisp.ops.pointcloud import create_pointcloud_from_images, normalize_pointcloud

from wisp.ops.test_mesh import get_obj_layers, get_OctreeAS, octree_to_layers, get_HashGrid, get_features_HashGrid
from wisp.ops.spc_utils import create_dual, octree_to_spc, get_level_points_from_octree
from wisp.ops.spc_formatting import describe_octree
from wisp.renderer.gizmos import PrimitivesPainter

testNgpNerfInteractive = "test-ngp-nerf-interactive"

GREEN = torch.FloatTensor([0, 1, 0, 1])
RED = torch.FloatTensor([1, 0, 0, 1]) 

"""
    Holds the settings of a single Bottom-Level renderer.
    Wisp supports joint rendering of various pipelines (NeRF, SDFs, meshes, and so forth),
    where each pipeline is wrapped by a bottom level renderer configured by this state object.
    The state object exists throughout the lifecycle of the renderer / pipeline,
    and is used to determine how the bottom level renderer of an existing pipeline is constructed.
"""

class DebugData(object):
    data = {
        'coords' : { 'points' : None },
        'mesh' :  { 'points' : None, 'lines' : None },
        'rays' :  { 'points' : None, 'lines' : None },
        'octree' : { 'points' : None }
    }
    data_train = {
        'coords' : { 'points' : None },
        'features' : { 'points' : None }
    }
    dataset = None

    def add_mesh_points_lines(self, colorT = GREEN):
        layers, layers_to_draw = get_obj_layers()
        # add points
        self.data['mesh']['points'] = PrimitivesPainter()
        self.data['mesh']['points'].redraw(layers_to_draw)

        # draw mesh
        self.data['mesh']['lines'] = PrimitivesPainter()
        self.data['mesh']['lines'].redraw(layers)

    #         cloudLayer, points_layers_to_draw = getDebugCloud(self.debug_data.dataset, self.wisp_state)
    def add_rays_points_lines(self, dataSet, colorT = GREEN):
        """
        'coords', 'data', 'dataset_num_workers', 'get_images', 'get_img_samples', 'img_shape', 'init',
         'mip', 'multiview_dataset_format', 'num_imgs', 'root', 'transform"""
        """"""
        c = dataSet.coords
        rays = dataSet.data.get('rays')
        rgbs = dataSet.data["imgs"] 
        masks = dataSet.data["masks"]
        #depths = dataSet.data["depth"] #wisp_state.graph.channels["depth"] #Channel depth is usually a distance to the surface hit point
        # add points
        points_layers_to_draw = [PrimitivesPack()]
        layers_to_draw = [PrimitivesPack()]

        N = rays.origins.shape[0]
        for j in range(0, len(rays)):
            for i in range(0, len(rays[j].origins)):
                layers_to_draw[j].add_lines(rays[j][i].origins, rays[j][i].origins + rays[j][i].dirs, colorT)
                points_layers_to_draw[j].add_points(rays[j][i].origins, colorT)
            break

        # add points
        self.data['rays']['points'] = PrimitivesPainter()
        self.data['rays']['points'].redraw(points_layers_to_draw)

        self.data['rays']['lines'] = PrimitivesPainter()
        self.data['rays']['lines'].redraw(layers_to_draw)

    def add_feature_points(self, wisp_state, lodix=15, colorT=GREEN):
        """
        'coords', 'data', 'dataset_num_workers', 'get_images', 'get_img_samples', 
        'img_shape', 'init', 'mip', 'multiview_dataset_format', 'num_imgs', 'root', 
        transform == rays
        needs to train to exist
        print("\n____init_wisp_state.neural_pipelines6", dir(wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef.grid.dense_points))
        print("\n____init_wisp_state.neural_pipelines7", wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef.grid.dense_points)
        print("\n____init_wisp_state.neural_pipelines8", wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef.grid.occupancy)
        
        ridx, pidx, samples, depths, deltas, boundary = nef.grid.raymarch(rays, 
                level=nef.grid.active_lods[lod_idx], num_samples=num_steps, raymarch_type=raymarch_type)
        """
        packedRFTracer = wisp_state.graph.neural_pipelines[testNgpNerfInteractive].tracer
        bl_state = wisp_state.graph.bl_renderers #dict_keys([testNgpNerfInteractive])
        neuralRadianceField = wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef
        features = wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef.features
        #            coords (torch.FloatTensor): packed tensor of shape [batch, num_samples, 3]
        try:
            #features = features.unsqueeze(0)
            coords = wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef.coords
            coords = torch.reshape(packedRFTracer.coords, (-1, 3)) 
            #coords = torch.reshape(coords, (-1, 3))
            #print(n, "__coords[0:1, 0:2, :3]", coords.shape, coords[0])
            #torch.Size([86, 32]) No ___1coords torch.Size([86, 1, 3])
            #coords (torch.FloatTensor): packed tensor of shape [batch, num_samples, 3] space?
            points_layers_to_draw = [PrimitivesPack()]
            for _i, x in enumerate(coords):
                #print(x)
                #for j in range(0, len(coords)):
                points_layers_to_draw[0].add_points(x, colorT)
            #print(len(coords),"___2coords", coords[0],  len(points_layers_to_draw.points))
            # add points
            if not self.data['features']['points']:
                self.data['features']['points'] = PrimitivesPainter()
            self.data['features']['points'].redraw(points_layers_to_draw)
            print(len(coords), features.shape, "0coords", coords.shape, points_layers_to_draw.points)
        except Exception as e:
            print(e, " ___1coords", type(wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef))


    def add_coords_points(self, wisp_state, colorT = GREEN):
        packedRFTracer = wisp_state.graph.neural_pipelines[testNgpNerfInteractive].tracer
        points_layers_to_draw = [PrimitivesPack()]
        try:
            coords = wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef.coords
            coords = torch.reshape(packedRFTracer.coords, (-1, 3)) 
            #coords = torch.reshape(coords, (-1, 3))
            for _i, x in enumerate(coords):
                points_layers_to_draw[0].add_points(x, colorT)
            # add points
            self.data['coords']['points'] = PrimitivesPainter()
            self.data['coords']['points'].redraw(points_layers_to_draw)
            self.data_train['coords']['points'] = PrimitivesPainter()
            self.data_train['coords']['points'].redraw(points_layers_to_draw)
        except Exception as e:
            print("No ___1coords", e, type(wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef))

    def add_octree(self, colorT = GREEN, levels=2, scale=False):
        octreeAS = get_OctreeAS(levels)
        h = get_HashGrid()
        f = get_features_HashGrid(octreeAS.points, h, lod_idx=15)
        print(octreeAS.points.shape, " __octreeAS.points ")#,  batch, num_samples )  
        o_layer = octree_to_layers(octreeAS.octree, levels, colorT)
        # points:  torch.Size([24535, 3])
        print("...max_level", octreeAS.max_level)#, octreeAS.points[0][0], octreeAS.points.shape)

        self.data['octree']['points'] = PrimitivesPainter()
        self.data['octree']['points'].redraw(o_layer)

    def add_all(self, wisp_state=None):
        self.add_mesh_points_lines()
        self.add_octree()
        if self.dataset and self.dataset.data.get('rays'):
            self.add_rays_points_lines(self.dataset)
        if wisp_state and wisp_state.graph.neural_pipelines.get(testNgpNerfInteractive):
            self.add_coords_points(wisp_state)

def init_debug_state(wisp_state, debug_data):
    for k1, v1 in debug_data.data_train.items():
        for k, _v in v1.items():
            wisp_state.debug[k1 + '_' + k] = False
    for k1, v1 in debug_data.data.items():
        for k, _v in v1.items():
            wisp_state.debug[k1 + '_' + k] = False

def render_debug(debug_data, wisp_state, camera):
    debug_data.add_coords_points(wisp_state)

    for k1, v1 in debug_data.data.items():
        for k, _v in v1.items():
            if wisp_state.debug.get(k1 + '_' + k):
                if debug_data.data.get(k1).get(k):
                    debug_data.data.get(k1).get(k).render(camera)

    for k1, v1 in debug_data.data_train.items():
        for k, _v in v1.items():
            if wisp_state.debug.get(k1 + '_' + k):
                if (k1 in ['coords']):
                    debug_data.add_coords_points(wisp_state)
                if debug_data.data_train.get(k1).get(k):
                    debug_data.data_train.get(k1).get(k).render(camera)


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

 Since the occupancy information is [compressed]
 (https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.spc.html?highlight=spc#octree) and 
 [packed](https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html?highlight=packed#packed), accessing level-specific information consistently involves
cumulative summarization of the number of "1" bits. <br>
It makes sense to calculate this information once and then cache it. <br>
The `pyramid` field does exactly that: it keeps summarizes the number of occupied cells per level, and their cumsum, for fast level-indexing.

        rgbs (list of torch.FloatTensor): List of RGB tensors of shape [H, W, 3].
        masks (list of torch.FloatTensor): List of mask tensors of shape [H, W, 1].
        rays (list of wisp.core.Rays): List of rays.origins and rays.dirs of shape [H, W, 3].
        depths (list of torch.FloatTensor): List of depth tensors of shape [H, W, 1].


 print(" d1: ", dir( self.dataset))
        print("__________dataset1:", self.dataset.keys()) #__________dataset: dict_keys(['imgs', 'masks', 'rays', 'cameras'])
        self.dataset['rays']
        print("\n____initwisp_state.channels ", wisp_state.graph.channels.keys())
        #def create_pointcloud_from_images(rgbs, masks, rays, depths):

        print("\n____initwisp_state.channels ", wisp_state.graph.channels.keys())
        print("\n____init_wisp_state.neural_pipelines1", dir(wisp_state.graph.channels["depth"])) # Channel
        print("\n____init_wisp_state.neural_pipelines2", wisp_state.graph.channels["hit"])
        print("\n____init_wisp_state.nesural_pipelines3", wisp_state.graph.neural_pipelines[testNgpNerfInteractive].named_parameters)
        print("\n____init_wisp_state.neural_pipelines4", dir(wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef))
        print("\n____init_wisp_state.neural_pipelines6", dir(wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef.grid.dense_points))
        print("\n____init_wisp_state.neural_pipelines7", wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef.grid.dense_points)
        print("\n____init_wisp_state.neural_pipelines8", wisp_state.graph.neural_pipelines[testNgpNerfInteractive].nef.grid.occupancy) #([20., 20., 20.,  ..., 20., 20., 20.])
        #print("\n____init_wisp_state.neural_pipelines5", wisp_state.graph.neural_pipelines[testNgpNerfInteractive].state_dict)
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
'''
