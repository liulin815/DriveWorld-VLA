from dataclasses import dataclass
from typing import List, Tuple, Dict

from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class ImagineWorldConfig:

    trajectory_sampling: TrajectorySampling = TrajectorySampling(
        time_horizon=4, interval_length=0.5
    )
    latent = True
    resnet34_path = '/e2e-data/evad-tech-vla/liulin/WoTE/ckpt/resnet34.pth'
    vlm_path = '/e2e-data/evad-tech-vla/liulin/WoTE/ckpt/ReCogDrive-VLM-2B'
    vlm_type = 'internvl'
    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    checkpoint_path = '/e2e-data/evad-tech-vla/liulin/WoTE/ckpt/stage1.ckpt'
    expert_checkpoint_path = '/e2e-data/evad-tech-vla/liulin/WoTE/ckpt/expert_imagineworld.ckpt'
    train_stage = 'stage1'
    #train_stage = 'stage2'
    #train_stage = 'stage3'

    # lantent DIT config

    dit_img_resolution=8
    dit_in_channels=256
    dit_label_dropout=0.2
    dit_num_classes=10
    dit_type='DiT-L/2'

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    num_keyval = 64 # 8*8
    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    # detect_boxes = True
    # use_bev_semantic = True

    detect_boxes = False
    use_bev_semantic = False
    
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
        7: ("ego_box", [TrackedObjectType.VEHICLE]),  # ego box
    }
    use_ego_box_in_map = True
    ego_box_map_idx = 7

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 8
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])

    # New
    # k-means traj
    num_traj_anchor: int = 256
    
    use_sim_reward: bool = True
    sim_reward_dict_path: str = f'/e2e-data/evad-tech-vla/liulin/WoTE/extra_data/planning_vb/formatted_pdm_score_256.npy'
    cluster_file_path = f'/e2e-data/evad-tech-vla/liulin/WoTE/extra_data/planning_vb/trajectory_anchors_256.npy'
    num_plan_queries: int = 32

    # map loss
    input_target = True
    use_map_loss: bool = True
    use_focal_loss_for_map = True
    bev_semantic_weight: float = 10.0
    future_idx = 11
    fut_bev_semantic_weight: float = 0.1
    focal_loss_alpha = 0.5
    focal_loss_gamma = 2.0
    
    # sampled trajs for supervision
    num_sampled_trajs: int = 1

    # recurrent
    num_fut_timestep = 1
    use_traj_offset = True

    # optmizer
    use_coslr_opt = True
    lr_steps = [70] # not used
    scheduler_type = "MultiStepLR" # not used
    weight_decay: float = 1e-4
    optimizer_type = "AdamW"
    cfg_lr_mult = 0.1
    opt_paramwise_cfg = {
        "name":{
            "image_encoder":{
                "lr_mult": cfg_lr_mult
            }
        }
    }
    max_epochs = 100

    # loss weight
    traj_offset_loss_weight = 1.0
    offset_im_reward_weight = 0.1
    im_loss_weight = 1.0
    metric_loss_weight = 1.0
