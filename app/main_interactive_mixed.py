# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import sys
import argparse
import yaml
import torch



'''

with open("example.yaml", "r") as stream:
    try:
        print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)
py ./app/main_interactive.py --config configs/mesh.yaml --dataset-path D:/workspace/INTEGRATION/kaolin-wisp/data/test/obj/1.obj
py ./app/main_interactive.py --config configs/ngp_nerf_interactive.yaml --dataset-path D:/workspace/INTEGRATION/kaolin-wisp/data/test/results_test_nored_200
py ./app/main_interactive.py --config configs/nglod_sdf_interactive.yaml --dataset-path D:/workspace/INTEGRATION/kaolin-wisp/data/test/obj/1.obj

py ./app/main_interactive_mixed.py --config configs/nglod_sdf_interactive.yaml --dataset-path D:/workspace/INTEGRATION/kaolin-wisp/data/test/obj/1.obj
py ./app/main_interactive_mixed.py --config configs/ngp_nerf_interactive.yaml --dataset-path  D:/workspace/INTEGRATION/kaolin-wisp/data/test/results_test_nored_200
Be aware that for now, the neural fields we optimize are assumed to exist in the normalized range of -1 to 1
 (supporting per instance model matrix is on our short-term roadmap)
'''
argsSdf = {'trainer_type': 'SDFTrainer', 'exp_name': 'test-nglod-sdf', 'perf': False, 'detect_anomaly': False, 'config': 'configs/nglod_sdf.yaml', 'grid_type': 'OctreeGrid', 'interpolation_type': 'linear', 'as_type': 'none', 'raymarch_type': 'voxel', 'multiscale_type': 'sum', 'feature_dim': 16, 'feature_std': 0.01, 'feature_bias': 0.0, 'noise_std': 0.0, 
'num_lods': 6, 'base_lod': 2, 'max_grid_res': 2048, 'tree_type': 'quad', 'codebook_bitwidth': 8, 'embedder_type': 'none', 'pos_multires': 10, 'view_multires': 4, 'nef_type': 'NeuralSDF', 'layer_type': 'none', 'activation_type': 'relu', 'decoder_type': 'basic', 'num_layers': 1, 'hidden_dim': 128, 'out_dim': 1, 'skip': None, 'pretrained': None, 'position_input': True, 'dataset_type': 'sdf', 'dataset_path': 'D:\\workspace\\INTEGRATION\\kaolin-wisp\\data\\test\\obj\\1.obj', 'dataset_num_workers': -1, 
'sample_mode': ['rand', 'near', 'near', 'trace', 'trace'], 'get_normals': False, 'num_samples': 500000, 
'num_samples_on_mesh': 10000000, 'sample_tex': False, 'mode_mesh_norm': 'sphere', 'samples_per_voxel': 32, 
'multiview_dataset_format': 'standard', 'num_rays_sampled_per_img': 4096, 'bg_color': 'white', 'mip': None, 'optimizer_type': 'adam',
 'lr': 0.001, 'weight_decay': 0, 'grid_lr_weight': 1.0, 'rgb_loss': 1.0, 'epochs': 10, 'batch_size': 512, 'resample': True, 'only_last': True,
 'resample_every': 1, 'model_format': 'full', 'save_as_new': False, 'save_every': 5, 'render_every': 5, 'log_2d': True, 
 'log_dir': '_results/logs/runs/', 'grow_every': -1, 'prune_every': -1, 'random_lod': False, 'growth_strategy': 'increase', 'valid_only': False, 
 'valid_every': 20, 'render_res': [1024, 1024], 'render_batch': 0, 'camera_origin': [-2.8, 2.3, -2.8], 'camera_lookat': [0, 0, 0], 
 'camera_fov': 30, 'camera_proj': 'persp', 'camera_clamp': [0, 10], 'tracer_type': 'PackedSDFTracer', 'num_steps': 128, 'step_size': 0.8, 'min_dis': 0.0003, 'matcap_path': 'data/matcaps/matcap_plastic_yellow.jpg', 'ao': False, 'shadow': True, 'shading_mode': 'matcap', 'log_level': 20}

argsNef = {'trainer_type': 'MultiviewTrainer', 'exp_name': 'test-ngp-nerf-interactive', 'perf': False, 'detect_anomaly': False, 'config': 'configs/ngp_nerf_interactive.yaml', 'grid_type': 'HashGrid', 'interpolation_type': 'linear', 'as_type': 'none', 'raymarch_type': 'ray', 'multiscale_type': 'cat', 'feature_dim': 2, 'feature_std': 0.01, 'feature_bias': 0.0, 'noise_std': 0.0, 'num_lods': 16, 'base_lod': 2, 'max_grid_res': 2048, 'tree_type': 'geometric', 'codebook_bitwidth': 19, 'embedder_type': 'positional', 'pos_multires': 10, 'view_multires': 4, 'nef_type': 'NeuralRadianceField', 'layer_type': 'none', 'activation_type': 'relu', 'decoder_type': 'basic', 'num_layers': 1, 'hidden_dim': 128, 'out_dim': 4, 'skip': None, 'pretrained': None, 'position_input': False, 'dataset_type': 'multiview', 'dataset_path': 'd:\\workspace\\INTEGRATION\\kaolin-wisp\\data\\test\\results_test_nored_200', 'dataset_num_workers': -1, 'sample_mode': ['rand', 'near', 'near', 'trace', 'trace'], 
'get_normals': False, 'num_samples': 100000, 'num_samples_on_mesh': 1000000, 'sample_tex': False, 'mode_mesh_norm': 'sphere', 'samples_per_voxel': 256, 'multiview_dataset_format': 'standard', 'num_rays_sampled_per_img': 4096, 'bg_color': 'black', 'mip': 2, 'optimizer_type': 'rmsprop', 'lr': 0.001, 'weight_decay': 0, 'grid_lr_weight': 100.0, 'rgb_loss': 1.0, 'epochs': 50, 'batch_size': 1, 'resample': False, 'only_last': False, 'resample_every': 1, 'model_format': 'full', 'save_as_new': False, 'save_every': -1, 'render_every': -1, 'log_2d': False, 'log_dir': '_results/logs/runs/', 'grow_every': -1, 'prune_every': -1, 'random_lod': False, 'growth_strategy': 'increase', 'valid_only': False, 'valid_every': -1, 'render_res': [1024, 1024], 'render_batch': 4000, 'camera_origin': [-3.0, 0.65, -3.0], 'camera_lookat': [0, 0, 0], 'camera_fov': 30, 'camera_proj': 'persp', 'camera_clamp': [0, 10], 'tracer_type': 'PackedRFTracer', 'num_steps': 512, 'step_size': 1.0, 'min_dis': 0.0003, 'matcap_path': 'data/matcaps/matcap_plastic_yellow.jpg', 'ao': False, 'shadow': False, 'shading_mode': 'rb', 'log_level': 20}
argsSdfInter = {'trainer_type': 'SDFTrainer', 'exp_name': 'test-nglod-sdf-interactive', 'perf': False, 'detect_anomaly': False, 
'config': 'configs/nglod_sdf_interactive.yaml', 'grid_type': 'OctreeGrid', 'interpolation_type': 'linear', 'as_type': 'none', 'raymarch_type': 'voxel', 'multiscale_type': 'sum', 'feature_dim': 16, 'feature_std': 0.01, 'feature_bias': 0.0, 'noise_std': 0.0, 'num_lods': 6, 'base_lod': 2, 'max_grid_res': 2048, 'tree_type': 'quad', 'codebook_bitwidth': 8, 'embedder_type': 'none', 'pos_multires': 10, 'view_multires': 4, 'nef_type': 'NeuralSDF', 'layer_type': 'none', 'activation_type': 'relu', 'decoder_type': 'basic', 'num_layers': 1, 'hidden_dim': 128, 'out_dim': 1, 'skip': None, 'pretrained': None, 'position_input': True, 'dataset_type': 'sdf', 'dataset_path': 'D:\\workspace\\INTEGRATION\\kaolin-wisp\\data\\test\\obj\\1.obj', 'dataset_num_workers': -1, 'sample_mode': ['rand', 'near', 'near', 'trace', 'trace'], 'get_normals': False, 'num_samples': 500000, 'num_samples_on_mesh': 10000000, 'sample_tex': False, 'mode_mesh_norm': 'sphere', 'samples_per_voxel': 32, 'multiview_dataset_format': 'standard', 'num_rays_sampled_per_img': 4096, 'bg_color': 'white', 'mip': None, 'optimizer_type': 'adam', 'lr': 0.001, 'weight_decay': 0, 'grid_lr_weight': 1.0, 'rgb_loss': 1.0, 'epochs': 10, 'batch_size': 512, 'resample': True, 'only_last': True, 'resample_every': 1, 'model_format': 'full', 'save_as_new': False, 'save_every': -1, 'render_every': -1, 
'log_2d': True, 'log_dir': '_results/logs/runs/', 'grow_every': -1, 'prune_every': -1, 'random_lod': False, 'growth_strategy': 'increase', 'valid_only': False, 'valid_every': -1, 'render_res': [1024, 1024], 'render_batch': 0, 'camera_origin': [-2.8, 2.3, -2.8], 'camera_lookat': [0, 0, 0], 'camera_fov': 30, 'camera_proj': 'persp', 'camera_clamp': [0, 10], 'tracer_type': 'PackedSDFTracer', 'num_steps': 128, 'step_size': 0.8, 'min_dis': 0.0003, 'matcap_path': 'data/matcaps/matcap_plastic_yellow.jpg', 'ao': False, 'shadow': True, 'shading_mode': 'matcap', 'log_level': 20}

class Object(object):
    dataset_type = "sdf"
    nef_type = None
    tracer_type = None
    pretrained = False
    sample_mode = None
    num_samples = 50000 
    num_samples_on_mesh = 100000
    num_lods = 6
    base_lod = 2 
    get_normals = False
    sample_tex = False 
    valid_only = False
    interpolation_type = 'linear'
    samples_per_voxel = 32
    log_dir = '_results/logs/runs/'
    dataset_path = r"D:/workspace/INTEGRATION/kaolin-wisp/data/test/obj/1.obj"
    extra_args = { "perf" : False,
     "dataset_path" : r"D:/workspace/INTEGRATION/kaolin-wisp/data/test/obj/1.obj" }
    epochs = 10
    batch_size = 512
    lr = 0.001
    grid_lr_weight = 1.0
    weight_decay = None
    exp_name = 'test-nglod-sdf-interactive'
    save_every = -1
    render_every = -1
    valid_every =  -1
    trainer_type = 'SDFTrainer'


sdF_args = Object()
sdF_args.nef_type = 'NeuralSDF'   
sdF_args.num_lods = 6
sdF_args.base_lod = 2 
sdF_args.samples_per_voxel = 32
sdF_args.num_samples = 5000
sdF_args.epochs = 10
sdF_args.batch_size = 512
sdF_args.save_every = -1
sdF_args.render_every = -1
sdF_args.valid_every =  -1
sdF_args.num_samples_on_mesh = 10000
sdF_args.tracer_type = 'PackedSDFTracer'
sdF_args.sample_mode = ['rand', 'near', 'near', 'trace', 'trace']
sdF_args.camera_clamp=[0, 10] 
sdF_args.camera_fov=30
sdF_args.camera_lookat=[0, 0, 0]
sdF_args.camera_origin=[-2.8, 2.3, -2.8]
sdF_args.camera_proj='persp'
sdF_args.codebook_bitwidth=8
sdF_args.config='configs/nglod_sdf_interactive.yaml'
sdF_args.dataset_num_workers=-1,
sdF_args.dataset_path='D:/workspace/INTEGRATION/kaolin-wisp/data/test/obj/1.obj'
sdF_args.dataset_type='sdf'
sdF_args.decoder_type='basic', 
sdF_args.detect_anomaly=False
sdF_args.embedder_type='none'
sdF_args.epochs=10
sdF_args.exp_name='test-nglod-sdf-interactive'
sdF_args.feature_bias=0.0
sdF_args.feature_dim=16
sdF_args.feature_std=0.01
sdF_args.get_normals=False
sdF_args.grid_lr_weight=1.0
sdF_args.grid_type='OctreeGrid'
sdF_args.grow_every=-1
sdF_args.growth_strategy='increase'
sdF_args.hidden_dim=128
sdF_args.interpolation_type='linear'
sdF_args.layer_type='none'
sdF_args.log_2d=True
sdF_args.log_dir='_results/logs/runs/'
sdF_args.log_level=20
sdF_args.lr=0.001 
sdF_args.matcap_path='data/matcaps/matcap_plastic_yellow.jpg'
sdF_args.max_grid_res=2048
sdF_args.min_dis=0.0003
sdF_args.mip=None
sdF_args.mode_mesh_norm='sphere'
sdF_args.model_format='full'
sdF_args.multiscale_type='sum'
sdF_args.multiview_dataset_format='standard'
sdF_args.nef_type='NeuralSDF'
sdF_args.noise_std=0.0
sdF_args.num_layers=1
sdF_args.num_lods=6
sdF_args.num_rays_sampled_per_img=4096
sdF_args.num_samples=5000
sdF_args.num_samples_on_mesh=10000
sdF_args.num_steps=128
sdF_args.only_last=True
sdF_args.optimizer_type='adam'
sdF_args.out_dim=1
sdF_args.perf=False
sdF_args.pos_multires=10
sdF_args.position_input=True
sdF_args.pretrained=None
sdF_args.prune_every=-1
sdF_args.random_lod=False
sdF_args.raymarch_type='voxel'
sdF_args.render_batch=0
sdF_args.render_every=-1 
sdF_args.render_res=[1024, 1024]
sdF_args.resample=True
sdF_args.resample_every=1
sdF_args.rgb_loss=1.0
sdF_args.sample_mode=['rand', 'near', 'near', 'trace', 'trace'] 
sdF_args.sample_tex=False
sdF_args.samples_per_voxel=32
sdF_args.save_as_new=False
sdF_args.save_every=-1
sdF_args.shading_mode='matcap'
sdF_args.shadow=True
sdF_args.skip=None
sdF_args.step_size=0.8 
sdF_args.tracer_type='PackedSDFTracer'
sdF_args.trainer_type='SDFTrainer'
sdF_args.tree_type='quad'
sdF_args.valid_every=-1
sdF_args.valid_only=False 
sdF_args.view_multires=4
sdF_args.weight_decay=0

#Namespace(activation_type='relu', ao=False, as_type='none', base_lod=2, batch_size=512, bg_color='white',
#  camera_clamp=[0, 10], camera_fov=30, camera_lookat=[0, 0, 0], camera_origin=[-2.8, 2.3, -2.8], camera_proj='persp', 
# codebook_bitwidth=8, config='configs/nglod_sdf_interactive.yaml', dataset_num_workers=-1,
#  dataset_path='D:/workspace/INTEGRATION/kaolin-wisp/data/test/obj/1.obj', dataset_type='sdf', decoder_type='basic', 
# detect_anomaly=False, embedder_type='none', epochs=10, exp_name='test-nglod-sdf-interactive', feature_bias=0.0, feature_dim=16, feature_std=0.01, get_normals=False, grid_lr_weight=1.0, grid_type='OctreeGrid', grow_every=-1, growth_strategy='increase', hidden_dim=128, interpolation_type='linear', layer_type='none', log_2d=True, log_dir='_results/logs/runs/', log_level=20, lr=0.001, matcap_path='data/matcaps/matcap_plastic_yellow.jpg', max_grid_res=2048, min_dis=0.0003, mip=None, mode_mesh_norm='sphere', model_format='full', multiscale_type='sum', multiview_dataset_format='standard', nef_type='NeuralSDF', noise_std=0.0, num_layers=1, num_lods=6, num_rays_sampled_per_img=4096, num_samples=5000, num_samples_on_mesh=10000, num_steps=128, only_last=True, optimizer_type='adam', out_dim=1, perf=False, pos_multires=10, position_input=True, pretrained=None, prune_every=-1, random_lod=False, raymarch_type='voxel', render_batch=0, render_every=-1, render_res=[1024, 1024], resample=True, resample_every=1, rgb_loss=1.0, sample_mode=['rand', 'near', 'near', 'trace', 'trace'], sample_tex=False, samples_per_voxel=32, save_as_new=False, save_every=-1, shading_mode='matcap', shadow=True, skip=None, step_size=0.8, tracer_type='PackedSDFTracer', trainer_type='SDFTrainer', tree_type='quad', valid_every=-1, valid_only=False, view_multires=4, weight_decay=0)
sdF_arg_dict = {'trainer_type': 'SDFTrainer', 'exp_name': 'test-nglod-sdf-interactive', 'perf': False, 'detect_anomaly': False, 
'config': 'configs/nglod_sdf_interactive.yaml', 'grid_type': 'OctreeGrid', 'interpolation_type': 'linear', 'as_type': 'none', 'raymarch_type': 'voxel', 'multiscale_type': 'sum', 'feature_dim': 16, 'feature_std': 0.01, 'feature_bias': 0.0, 'noise_std': 0.0, 'num_lods': 6, 'base_lod': 2, 'max_grid_res': 2048, 'tree_type': 'quad', 'codebook_bitwidth': 8, 'embedder_type': 'none', 'pos_multires': 10, 'view_multires': 4, 'nef_type': 'NeuralSDF', 'layer_type': 'none', 'activation_type': 'relu', 'decoder_type': 'basic', 'num_layers': 1, 'hidden_dim': 128, 'out_dim': 1, 'skip': None, 'pretrained': None, 'position_input': True, 'dataset_type': 'sdf', 'dataset_path': 'D:/workspace/INTEGRATION/kaolin-wisp/data/test/obj/1.obj', 'dataset_num_workers': -1, 'sample_mode': ['rand', 'near', 'near', 'trace', 'trace'], 'get_normals': False, 'num_samples': 5000, 'num_samples_on_mesh': 10000, 'sample_tex': False, 'mode_mesh_norm': 'sphere', 'samples_per_voxel': 32, 'multiview_dataset_format': 'standard', 'num_rays_sampled_per_img': 4096, 'bg_color': 'white', 'mip': None, 'optimizer_type': 'adam', 'lr': 0.001, 'weight_decay': 0, 'grid_lr_weight': 1.0, 'rgb_loss': 1.0, 'epochs': 10, 'batch_size': 512, 'resample': True, 'only_last': True, 'resample_every': 1, 'model_format': 'full', 'save_as_new': False, 'save_every': -1, 'render_every': -1, 'log_2d': True, 'log_dir': '_results/logs/runs/', 'grow_every': -1, 'prune_every': -1, 'random_lod': False, 'growth_strategy': 'increase', 'valid_only': False, 'valid_every': -1, 'render_res': [1024, 1024], 'render_batch': 0, 'camera_origin': [-2.8, 2.3, -2.8], 'camera_lookat': [0, 0, 0], 'camera_fov': 30, 'camera_proj': 'persp', 'camera_clamp': [0, 10], 'tracer_type': 'PackedSDFTracer', 'num_steps': 128, 'step_size': 0.8, 'min_dis': 0.0003, 'matcap_path': 'data/matcaps/matcap_plastic_yellow.jpg', 'ao': False, 'shadow': True, 'shading_mode': 'matcap', 'log_level': 20}

if __name__ == "__main__":
    from cuda_guard import setup_cuda_context
    setup_cuda_context()     # Must be called before any torch operations take place

    import os
    import app_utils
    import logging as log
    from wisp.trainers import *
    from wisp.config_parser import parse_options, argparse_to_str, get_modules_from_config, \
        get_optimizer_from_config
    from wisp.framework import WispState

    # Usual boilerplate
    parser = parse_options(return_parser=True)
    app_utils.add_log_level_flag(parser)
    app_group = parser.add_argument_group('app')
    # Add custom args if needed for app
    args, args_str = argparse_to_str(parser)
    app_utils.default_log_setup(args.log_level)
    #___args ArgumentParser(prog='main_interactive_mixed.py', usage=None, description='ArgumentParser for kaolin-wisp.',
    #  formatter_class=<class 'argparse.HelpFormatter'>, conflict_handler='error', add_help=True)
    # Create the parser

    '''  
            matrix: A tensor of shape (4, 4) or of shape (minibatch, 4, 4)
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
                the specified `device` and `dtype`
                        M = [
                [Rxx, Ryx, Rzx, 0],
                [Rxy, Ryy, Rzy, 0],
                [Rxz, Ryz, Rzz, 0],
                [Tx,  Ty,  Tz,  1],
            ]
    ''' 
    M_data = [  [.3, 0., 0., 0], 
                [0., .3, 0., 0], 
                [0., 0., .3, 0], 
                [0., 0., 0., 1]
        ]
    sdF_args.matrix = torch.Tensor( M_data)
    pipeline1, train_dataset1, device = get_modules_from_config(sdF_args)
    #print(" pipeline", pipeline) #neural_pipelines.items():
    
    pipeline, train_dataset, device = get_modules_from_config(args)


    optim_cls, optim_params = get_optimizer_from_config(args)
    scene_state = WispState()
    #    def __init__(self, pipeline, dataset, num_epochs, batch_size,

    #   for renderer_id, neural_pipeline in scene_graph.neural_pipelines.items():
    trainerSDf = globals()[sdF_args.trainer_type](pipeline1, train_dataset1, 
                                     sdF_args.epochs, 
                                     sdF_args.batch_size,
                                      optim_cls, 
                                      sdF_args.lr, 
                                      sdF_args.weight_decay,
                                      sdF_args.grid_lr_weight, 
                                      optim_params, 
                                      sdF_args.log_dir, 
                                      device,
                                      exp_name=sdF_args.exp_name, 
                                      info=args_str, 
                                      extra_args=sdF_arg_dict,
                                      render_every=sdF_args.render_every, 
                                      save_every=sdF_args.save_every,
                                      scene_state=scene_state)
    print("_______scene_state1: ", scene_state.graph.neural_pipelines.keys(), args_str)

    trainer = globals()[args.trainer_type](pipeline, train_dataset, args.epochs, args.batch_size,
                                      optim_cls, args.lr, args.weight_decay,
                                      args.grid_lr_weight, optim_params, args.log_dir, device,
                                      exp_name=args.exp_name, info=args_str, 
                                      extra_args=vars(args),
                                      render_every=args.render_every, save_every=args.save_every,
                                      scene_state=scene_state)
    print("_______scene_state2: ", scene_state.graph.neural_pipelines.keys())

    if not os.environ.get('WISP_HEADLESS') == '1':
        from wisp.renderer.app.optimization_app import OptimizationApp
        scene_state.renderer.device = trainer.device  # Use same device for trainer and renderer
        #    def __init__(self, wisp_state: WispState, trainer_step_func: Callable[[], None], experiment_name: str, dataset=None):
        
        def joint_train_step():
            trainer.iterate()
            trainerSDf.iterate()

        renderer = OptimizationApp(wisp_state=scene_state,
                                        trainer_step_func=joint_train_step, #[ trainer.iterate, trainerSDf.iterate],
                                        experiment_name="wisp trainer",
                                        dataset=train_dataset1 # debug_data only
                                        )
        renderer.run()
    else:
        log.info("Running headless. For the app, set WISP_HEADLESS=0")
        if args.valid_only:
            trainer.validate()
        else:
            trainer.train()
