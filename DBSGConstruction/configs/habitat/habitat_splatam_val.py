import os
from os.path import join as p_join

primary_device = "cuda:0"

scenes = ["habitat"]
# sequence_name = '4ok3usBNeis'
# sequence_name = '5cdEh9F2hJL'
# sequence_name = 'a8BtkwhxdRV'
# sequence_name = 'bxsVRursffK'
sequence_name = 'CrMo8WxCyVb'



seed = int(4)
scene_name = scenes[0]

map_every = 2
keyframe_every = 5  # 防 OOM：减小关键帧密度，降低映射时显存峰值
mapping_window_size = 6  # 防 OOM：缩小窗口，显著降低每步显存占用
tracking_iters = 200
mapping_iters = 10  # 防 OOM：降低每个 time step 的映射计算负担
scene_radius_depth_ratio = 2

group_name = "habitat_test"
# 验证 CLIP 时：先跑 use_clip=False、run_name 用 _no_clip；再跑 use_clip=True、run_name 用 _clip，最后用 eval_clip_effect.py compare 对比
use_clip = True  # True=启用 CLIP 语义向量（需 pip install open-clip-torch）
run_name = f"{sequence_name}_seed{seed}_clip" if use_clip else f"{sequence_name}_seed{seed}_clip"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=200, # 更频繁输出全局进度
    eval_every=100, # 评估更密：避免只看两个点误判趋势
    scene_radius_depth_ratio=scene_radius_depth_ratio, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=True, # Save Checkpoints
    checkpoint_interval=50, # Checkpoint Interval
    use_wandb=False,
    wandb=dict(
        entity="mcc_nav",
        project="SplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="configs/data/habitat_val",
        gradslam_data_cfg=f"configs/data/habitat/{scene_name}.yaml",
        sequence=sequence_name,
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        num_frames=1000,
    ),
    use_clip=True,
    clip_dim=512,
    clip_model_name="ViT-B-32",
    tracking=dict(
        use_gt_poses=True, # Use GT Poses for Tracking
        forward_prop=False, # TODO Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.002,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        max_new_gaussians_per_frame=1000,  # 防 OOM：限制单帧新增高斯，抑制显存膨胀
        sil_thres=0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True, # TODO Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=40,  # 折中清理长度，避免过长 prune 带来显存峰值
            prune_every=5,
            removal_opacity_threshold=0.007,
            final_removal_opacity_threshold=0.007,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=False, # Enter Interactive Mode after Online Recon Viz
    ),
)
