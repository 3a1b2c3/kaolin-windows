# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import sys
import argparse
'''
___args <class 'argparse.Namespace'>
 py ./app/main_interactive_mixed.py --config configs/nglod_sdf_interactive.yaml --dataset-path D:\workspace\INTEGRATION\kaolin-wisp\data\test\obj\1.obj
 py ./app/main_interactive_mixed.py --config configs/ngp_nerf_interactive.yaml --dataset-path d:\workspace\INTEGRATION\kaolin-wisp\data\test\results_test_nored_200

'''
argsSdf =  {'trainer_type': 'SDFTrainer', 'exp_name': 'test-nglod-sdf', 'perf': False, 'detect_anomaly': False, 'config': 'configs/nglod_sdf.yaml', 'grid_type': 'OctreeGrid', 'interpolation_type': 'linear', 'as_type': 'none', 'raymarch_type': 'voxel', 'multiscale_type': 'sum', 'feature_dim': 16, 'feature_std': 0.01, 'feature_bias': 0.0, 'noise_std': 0.0, 'num_lods': 6, 'base_lod': 2, 'max_grid_res': 2048, 'tree_type': 'quad', 'codebook_bitwidth': 8, 'embedder_type': 'none', 'pos_multires': 10, 'view_multires': 4, 'nef_type': 'NeuralSDF', 'layer_type': 'none', 'activation_type': 'relu', 'decoder_type': 'basic', 'num_layers': 1, 'hidden_dim': 128, 'out_dim': 1, 'skip': None, 'pretrained': None, 'position_input': True, 'dataset_type': 'sdf', 'dataset_path': 'D:\\workspace\\INTEGRATION\\kaolin-wisp\\data\\test\\obj\\1.obj', 'dataset_num_workers': -1, 'sample_mode': ['rand', 'near', 'near', 'trace', 'trace'], 'get_normals': False, 'num_samples': 500000, 'num_samples_on_mesh': 10000000, 'sample_tex': False, 'mode_mesh_norm': 'sphere', 'samples_per_voxel': 32, 'multiview_dataset_format': 'standard', 'num_rays_sampled_per_img': 4096, 'bg_color': 'white', 'mip': None, 'optimizer_type': 'adam', 'lr': 0.001, 'weight_decay': 0, 'grid_lr_weight': 1.0, 'rgb_loss': 1.0, 'epochs': 10, 'batch_size': 512, 'resample': True, 'only_last': True, 'resample_every': 1, 'model_format': 'full', 'save_as_new': False, 'save_every': 5, 'render_every': 5, 'log_2d': True, 'log_dir': '_results/logs/runs/', 'grow_every': -1, 'prune_every': -1, 'random_lod': False, 'growth_strategy': 'increase', 'valid_only': False, 'valid_every': 20, 'render_res': [1024, 1024], 'render_batch': 0, 'camera_origin': [-2.8, 2.3, -2.8], 'camera_lookat': [0, 0, 0], 'camera_fov': 30, 'camera_proj': 'persp', 'camera_clamp': [0, 10], 'tracer_type': 'PackedSDFTracer', 'num_steps': 128, 'step_size': 0.8, 'min_dis': 0.0003, 'matcap_path': 'data/matcaps/matcap_plastic_yellow.jpg', 'ao': False, 'shadow': True, 'shading_mode': 'matcap', 'log_level': 20}
argsNef = {'trainer_type': 'MultiviewTrainer', 'exp_name': 'test-ngp-nerf-interactive', 'perf': False, 'detect_anomaly': False, 'config': 'configs/ngp_nerf_interactive.yaml', 'grid_type': 'HashGrid', 'interpolation_type': 'linear', 'as_type': 'none', 'raymarch_type': 'ray', 'multiscale_type': 'cat', 'feature_dim': 2, 'feature_std': 0.01, 'feature_bias': 0.0, 'noise_std': 0.0, 'num_lods': 16, 'base_lod': 2, 'max_grid_res': 2048, 'tree_type': 'geometric', 'codebook_bitwidth': 19, 'embedder_type': 'positional', 'pos_multires': 10, 'view_multires': 4, 'nef_type': 'NeuralRadianceField', 'layer_type': 'none', 'activation_type': 'relu', 'decoder_type': 'basic', 'num_layers': 1, 'hidden_dim': 128, 'out_dim': 4, 'skip': None, 'pretrained': None, 'position_input': False, 'dataset_type': 'multiview', 'dataset_path': 'd:\\workspace\\INTEGRATION\\kaolin-wisp\\data\\test\\results_test_nored_200', 'dataset_num_workers': -1, 'sample_mode': ['rand', 'near', 'near', 'trace', 'trace'], 'get_normals': False, 'num_samples': 100000, 'num_samples_on_mesh': 1000000, 'sample_tex': False, 'mode_mesh_norm': 'sphere', 'samples_per_voxel': 256, 'multiview_dataset_format': 'standard', 'num_rays_sampled_per_img': 4096, 'bg_color': 'black', 'mip': 2, 'optimizer_type': 'rmsprop', 'lr': 0.001, 'weight_decay': 0, 'grid_lr_weight': 100.0, 'rgb_loss': 1.0, 'epochs': 50, 'batch_size': 1, 'resample': False, 'only_last': False, 'resample_every': 1, 'model_format': 'full', 'save_as_new': False, 'save_every': -1, 'render_every': -1, 'log_2d': False, 'log_dir': '_results/logs/runs/', 'grow_every': -1, 'prune_every': -1, 'random_lod': False, 'growth_strategy': 'increase', 'valid_only': False, 'valid_every': -1, 'render_res': [1024, 1024], 'render_batch': 4000, 'camera_origin': [-3.0, 0.65, -3.0], 'camera_lookat': [0, 0, 0], 'camera_fov': 30, 'camera_proj': 'persp', 'camera_clamp': [0, 10], 'tracer_type': 'PackedRFTracer', 'num_steps': 512, 'step_size': 1.0, 'min_dis': 0.0003, 'matcap_path': 'data/matcaps/matcap_plastic_yellow.jpg', 'ao': False, 'shadow': False, 'shading_mode': 'rb', 'log_level': 20}
argsSdfInter = {'trainer_type': 'SDFTrainer', 'exp_name': 'test-nglod-sdf-interactive', 'perf': False, 'detect_anomaly': False, 
'config': 'configs/nglod_sdf_interactive.yaml', 'grid_type': 'OctreeGrid', 'interpolation_type': 'linear', 'as_type': 'none', 'raymarch_type': 'voxel', 'multiscale_type': 'sum', 'feature_dim': 16, 'feature_std': 0.01, 'feature_bias': 0.0, 'noise_std': 0.0, 'num_lods': 6, 'base_lod': 2, 'max_grid_res': 2048, 'tree_type': 'quad', 'codebook_bitwidth': 8, 'embedder_type': 'none', 'pos_multires': 10, 'view_multires': 4, 'nef_type': 'NeuralSDF', 'layer_type': 'none', 'activation_type': 'relu', 'decoder_type': 'basic', 'num_layers': 1, 'hidden_dim': 128, 'out_dim': 1, 'skip': None, 'pretrained': None, 'position_input': True, 'dataset_type': 'sdf', 'dataset_path': 'D:\\workspace\\INTEGRATION\\kaolin-wisp\\data\\test\\obj\\1.obj', 'dataset_num_workers': -1, 'sample_mode': ['rand', 'near', 'near', 'trace', 'trace'], 'get_normals': False, 'num_samples': 500000, 'num_samples_on_mesh': 10000000, 'sample_tex': False, 'mode_mesh_norm': 'sphere', 'samples_per_voxel': 32, 'multiview_dataset_format': 'standard', 'num_rays_sampled_per_img': 4096, 'bg_color': 'white', 'mip': None, 'optimizer_type': 'adam', 'lr': 0.001, 'weight_decay': 0, 'grid_lr_weight': 1.0, 'rgb_loss': 1.0, 'epochs': 10, 'batch_size': 512, 'resample': True, 'only_last': True, 'resample_every': 1, 'model_format': 'full', 'save_as_new': False, 'save_every': -1, 'render_every': -1, 'log_2d': True, 'log_dir': '_results/logs/runs/', 'grow_every': -1, 'prune_every': -1, 'random_lod': False, 'growth_strategy': 'increase', 'valid_only': False, 'valid_every': -1, 'render_res': [1024, 1024], 'render_batch': 0, 'camera_origin': [-2.8, 2.3, -2.8], 'camera_lookat': [0, 0, 0], 'camera_fov': 30, 'camera_proj': 'persp', 'camera_clamp': [0, 10], 'tracer_type': 'PackedSDFTracer', 'num_steps': 128, 'step_size': 0.8, 'min_dis': 0.0003, 'matcap_path': 'data/matcaps/matcap_plastic_yellow.jpg', 'ao': False, 'shadow': True, 'shading_mode': 'matcap', 'log_level': 20}

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
    #___args ArgumentParser(prog='main_interactive_mixed.py', usage=None, description='ArgumentParser for kaolin-wisp.',
    #  formatter_class=<class 'argparse.HelpFormatter'>, conflict_handler='error', add_help=True)
    # Create the parser
    parserNerf = argparse.ArgumentParser()
    # Add an argument
    parserNerf.add_argument('--config', default="configs/ngp_nerf_interactive.yaml")
    parserNerf.add_argument('--dataset-path', default="d:/workspace/INTEGRATION/kaolin-wisp/data/test/results_test_nored_200")
    #argsNErf, args_strNErf = argparse_to_str(parserNerf)
    #print("___args", parserNerf)#, vars(args)) #___args <class 'argparse.Namespace'>

    app_utils.default_log_setup(args.log_level)

    pipeline, train_dataset, device = get_modules_from_config(args)
    #pipeline1, train_dataset1, device = get_modules_from_config(argsNErf)
    print(" pipeline", pipeline) #neural_pipelines.items():
    optim_cls, optim_params = get_optimizer_from_config(args)
    scene_state = WispState()
    trainer = globals()[args.trainer_type](pipeline, train_dataset, args.epochs, args.batch_size,
                                      optim_cls, args.lr, args.weight_decay,
                                      args.grid_lr_weight, optim_params, args.log_dir, device,
                                      exp_name=args.exp_name, info=args_str, extra_args=vars(args),
                                      render_every=args.render_every, save_every=args.save_every,
                                      scene_state=scene_state)

    #   for renderer_id, neural_pipeline in scene_graph.neural_pipelines.items():
    trainerSDf = globals()[args.trainer_type](pipeline, train_dataset, args.epochs, args.batch_size,
                                      optim_cls, args.lr, args.weight_decay,
                                      args.grid_lr_weight, optim_params, args.log_dir, device,
                                      exp_name=args.exp_name, info=args_str, extra_args=vars(args),
                                      render_every=args.render_every, save_every=args.save_every,
                                      scene_state=scene_state)
    print("scene_state", scene_state)
    if not os.environ.get('WISP_HEADLESS') == '1':
        from wisp.renderer.app.optimization_app import OptimizationApp
        scene_state.renderer.device = trainer.device  # Use same device for trainer and renderer
        #    def __init__(self, wisp_state: WispState, trainer_step_func: Callable[[], None], experiment_name: str, dataset=None):
        renderer = OptimizationApp(wisp_state=scene_state,
                                        trainer_step_func=trainer.iterate,
                                        experiment_name="wisp trainer",
                                        dataset=train_dataset
                                        )
        renderer.run()
    else:
        log.info("Running headless. For the app, set WISP_HEADLESS=0")
        if args.valid_only:
            trainer.validate()
        else:
            trainer.train()
