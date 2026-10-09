"""
SplaTAM / Semantic Gaussian 建图配置：HM3D **train** 导出数据根目录。

与 habitat_splatam_val.py 相同，仅 data.basedir 指向 configs/data/habitat_train。
run_build_val_semantic_gaussians.py 默认仍使用 val 配置，train 建图请传::

  --config configs/habitat/habitat_splatam_train.py
"""
import os
from os.path import join as p_join

primary_device = "cuda:0"

scenes = ["habitat"]
sequence_name = "CrMo8WxCyVb"

seed = int(4)
scene_name = scenes[0]

map_every = 1
keyframe_every = 5
mapping_window_size = 12
tracking_iters = 200
mapping_iters = 40
scene_radius_depth_ratio = 2

group_name = "habitat_test"
use_clip = True
run_name = f"{sequence_name}_seed{seed}_clip" if use_clip else f"{sequence_name}_seed{seed}_clip"

# .../GaussianConstruction/configs/habitat/this_file.py -> .../GaussianConstruction
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAIN_DATA = p_join(_BASE, "configs", "data", "habitat_train")
_GRADSLAM_YAML = p_join(_BASE, "configs", "data", "habitat", f"{scene_name}.yaml")

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=200,
    eval_every=100,
    scene_radius_depth_ratio=scene_radius_depth_ratio,
    mean_sq_dist_method="projective",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    # 省磁盘：不保存中间 params{N}.npz（仅最终 save_params 落盘）
    save_checkpoints=True,
    checkpoint_interval=100,
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
        basedir=_TRAIN_DATA,
        gradslam_data_cfg=_GRADSLAM_YAML,
        sequence=sequence_name,
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        num_frames=500,
    ),
    use_clip=True,
    clip_dim=512,
    clip_model_name="ViT-B-32",
    tracking=dict(
        use_gt_poses=True,
        forward_prop=False,
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
        max_new_gaussians_per_frame=3000,
        sil_thres=0.5,
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
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=40,
            prune_every=5,
            removal_opacity_threshold=0.007,
            final_removal_opacity_threshold=0.007,
            reset_opacities=False,
            reset_opacities_every=500,
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),
    viz=dict(
        render_mode="color",
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=600,
        viz_h=340,
        viz_near=0.01,
        viz_far=100.0,
        view_scale=2,
        viz_fps=5,
        enter_interactive_post_online=False,
    ),
)
