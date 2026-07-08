from re import T
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import timm
import time
import lzma
import math
import pickle
from navsim.common.dataclasses import Trajectory
from dataclasses import asdict, dataclass, field

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.enums import StateSE2Index
from navsim.common.dataloader import MetricCacheLoader
from pathlib import Path

import torchvision.models as models
import torch.nn.functional as F
from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs

import os
from datetime import datetime
from navsim.agents.ImagineWorld.ImagineWorld_targets import BoundingBox2DIndex
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from typing import Any, List, Dict, Union
from navsim.agents.ImagineWorld.ImagineWorld_backbone import ImagineWorldBackbone
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)


class CrossBEV(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, nlayers):
        super().__init__()
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), nlayers
        )
    
    def forward(self, embedded_vocab, bev_feature):
        tr_out = self.transformer(embedded_vocab, bev_feature)
        return tr_out



class ImagineRewardModel(nn.Module):
    def __init__(self, 
                 config,
                ):
        super().__init__()

        # Define constants as variables
        STATUS_ENCODING_INPUT_DIM = 4 + 2 + 2
        CLUSTER_CENTERS_FEATURE_DIM = 24

        self._status_encoding = nn.Linear(STATUS_ENCODING_INPUT_DIM, config.tf_d_model)

        num_poses = config.trajectory_sampling.num_poses
        num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 64
        self._keyval_embedding = nn.Embedding(
            num_keyval, config.tf_d_model
        )
        
        self.reward_weights = config.reward_weights if hasattr(config, 'reward_weights') else [0.1, 0.5, 0.5, 1.0]
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"

        self.metric_cache_loader = MetricCacheLoader(Path(config.metric_cache_path))
        self.test_metric_cache_loader = MetricCacheLoader(Path(config.test_metric_cache_path))
        proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self.simulator = PDMSimulator(proposal_sampling)
        self.train_scorer = PDMScorer(proposal_sampling, config.scorer_config)
        self.layernorm_for_bev = nn.LayerNorm(256)

        self.pos_embed = nn.Sequential(
            nn.Linear(CLUSTER_CENTERS_FEATURE_DIM, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        self.cross_cur_bev = CrossBEV(d_model=config.tf_d_model, nhead=config.vadv2_head_nhead, 
                                d_ffn=config.tf_d_ffn, nlayers=config.vadv2_head_nlayers)
        self.cross_fut_bev = CrossBEV(d_model=config.tf_d_model, nhead=config.vadv2_head_nhead, 
                                d_ffn=config.tf_d_ffn, nlayers=config.vadv2_head_nlayers)
        

        self.heads = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'drivable_area_compliance':
                nn.Sequential(
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, 1),
                ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'comfort': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'final_scores': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
        })
        self.save_root_dir = '/e2e-data/evad-tech-vla/liulin/WoTE/gt_score_dir'
    
    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(
                self._checkpoint_path, map_location=torch.device("cpu")
            )["state_dict"]
        
        if "agent.ImagineWorld_model.trajectory_anchors" in state_dict:
            del state_dict["agent.ImagineWorld_model.trajectory_anchors"]

        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)

    def _get_ego_status_feature(self, status_feature: torch.Tensor) -> torch.Tensor:
        """
        Obtain the encoded ego vehicle status features.
        """
        status_encoding = self._status_encoding(status_feature)  # [batch_size, C]
        ego_status_feat = status_encoding[:, None, :]  # [batch_size, 1, C]
        return ego_status_feat

    def _concatenate_ego_and_traj_features(self, ego_status_feat: torch.Tensor, trajectory_anchors_feat: torch.Tensor) -> torch.Tensor:
        """
        Concatenate ego features with trajectory features and encode.
        """
        # Repeat ego_status_feat to match the number of trajectories
        ego_status_feat = ego_status_feat.repeat(1, trajectory_anchors_feat.shape[1], 1)  # [batch_size, num_traj, C]
        # Concatenate
        ego_feat = torch.cat([ego_status_feat, trajectory_anchors_feat], dim=-1)  # [batch_size, num_traj, C + encoded_dim]
        # Encode
        ego_feat = self.encode_ego_feat_mlp(ego_feat)  # [batch_size, num_traj, C']
        ego_feat = ego_feat.unsqueeze(-2)  # [batch_size, num_traj, 1, C']
        return ego_feat
    
    def select_best_trajectory(self, final_rewards, trajectory_anchors, batch_size):
        best_trajectory_idx = torch.argmax(final_rewards, dim=-1)  # Shape: [batch_size]
        poses = trajectory_anchors[best_trajectory_idx]  # Shape: [batch_size, 24]
        poses = poses.view(batch_size, 8, 3)  # Reshape to [batch_size, 8, 3]
        return poses

    def forward_test(self, features, targets=None, result=None, token=None) -> Dict[str, torch.Tensor]:
        # Extract scene feature encoding
        self.is_eval = True
        results = {}
        trajectory_anchors = result['trajectory_anchors'].detach()
        trajectory_offset = result['trajectory_offset'].detach()
        n_samples = 16 # 在一个batch_size中我们选择得分为前16的轨迹进行未来想象
        bs = trajectory_offset.shape[0]
        
        # ---------------------------- feature preparetion -----------------------
        bev_flatten_feature = result['flatten_bev_feature_1'].detach()
        bev_flatten_feature = self.layernorm_for_bev(bev_flatten_feature)
        bev_flatten_feature = bev_flatten_feature + self._keyval_embedding.weight[None, :, :]
        latent_future_feature = result['latent_future'].detach().flatten(-2, -1).permute(0, 2, 1)
        latent_future_feature = latent_future_feature.chunk(2, dim=0)[0]
        #import pdb;pdb.set_trace()
        latent_future_feature = latent_future_feature + self._keyval_embedding.weight[None, :, :]
        latent_future_feature = latent_future_feature.reshape(bs, 16, 64, 256) # [bs, 16, 64, 256]

        # 消融实验用 jfy改 原版跑的时候把noise_std改成0
        # noise_std = 0.0
        # noise = torch.randn_like(latent_future_feature, dtype=latent_future_feature.dtype, device=latent_future_feature.device) * noise_std
        # latent_future_feature = latent_future_feature + noise
        # 消融实验用 jfy改 变成0
        #latent_future_feature_zero=torch.zeros_like(latent_future_feature, dtype=latent_future_feature.dtype, device=latent_future_feature.device)
        #latent_future_feature=latent_future_feature_zero

        expand_trajectory_anchors = trajectory_anchors.unsqueeze(0).repeat(bs, 1, 1, 1)
        output_trajectory = expand_trajectory_anchors + trajectory_offset
        final_rewards = result['final_reward'].detach()
        topk_indices = torch.topk(final_rewards, k=16, dim=1).indices
        topk_trajectories = torch.gather(output_trajectory, dim=1, index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 3))
        # ----------------------------------------------------------
        metric_cache = {}
        gt_score_pickcle = {}
        path = self.metric_cache_loader.metric_cache_paths[token]
        with lzma.open(path, 'rb') as f:
            metric_cache[token] = pickle.load(f)
        gt_rewards, gt_scores = self.reward_fn(topk_trajectories, [token], metric_cache)
        sub_dir = path.split('/')[-4]
        full_dir = os.path.join(self.save_root_dir, sub_dir)
        os.makedirs(full_dir, exist_ok=True)
        gt_score_pickcle['proposals'] = topk_trajectories
        gt_score_pickcle['proposals_idx'] = topk_indices
        gt_score_pickcle['sub_scores'] = gt_rewards
        gt_score_pickcle['pdms'] = gt_scores
        pkl_name = token + '.pkl'
        full_pkl_path = os.path.join(full_dir, pkl_name)
        with open(full_pkl_path, "wb") as f:
            pickle.dump(gt_score_pickcle, f)
        # ----------------------------------------------------------
        topk_trajectories_feature = self.pos_embed(topk_trajectories.flatten(-2,-1))

        B, K, N, C = latent_future_feature.shape
        bev_flatten_feature = bev_flatten_feature.unsqueeze(1).repeat(1, K, 1, 1)
        traj_out = self.cross_cur_bev(topk_trajectories_feature.reshape(B*K, 1, C), bev_flatten_feature.reshape(B*K, N, C))
        traj_out = self.cross_fut_bev(traj_out, latent_future_feature.reshape(B*K, N, C))
        traj_out = traj_out.reshape(B,K,C)
        status_feature = features['status_feature'].detach()
        status_encoding = self._status_encoding(status_feature).unsqueeze(1).repeat(1, K, 1)
        #import pdb;pdb.set_trace()
        traj_out = traj_out + status_encoding

        for k, head in self.heads.items():
            results[k] = head(traj_out)
        no_at_fault_collisions = results['no_at_fault_collisions'].sigmoid()
        drivable_area_compliance = results['drivable_area_compliance'].sigmoid()
        driving_direction_compliance = results['driving_direction_compliance'].sigmoid()
        time_to_collision_within_bound = results['time_to_collision_within_bound'].sigmoid()
        comfort = results['comfort'].sigmoid()
        scores = 0.5 * no_at_fault_collisions.log() + 0.5 * drivable_area_compliance.log() \
                 + 0.5 * driving_direction_compliance.log() + 1.0 * (5.0 * time_to_collision_within_bound + 2.0 * comfort).log()
        scores = scores.squeeze(2)
        best_trajectory_idx = torch.argmax(scores, dim=-1)
        batch_idx = torch.arange(bs, device=topk_trajectories.device)
        best_trajectory = topk_trajectories[batch_idx, best_trajectory_idx]
        #import pdb;pdb.set_trace()
        results["trajectory"] = best_trajectory
        #import pdb;pdb.set_trace()
        return results
    
    def forward_train(self, features, targets=None, result=None) -> Dict[str, torch.Tensor]:
        
        reward = {}
        unique_tokens = targets['token']
        
        trajectory_anchors = result['trajectory_anchors'].detach()
        trajectory_offset = result['trajectory_offset'].detach()
        n_samples = 16 # 在一个batch_size中我们选择得分为前16的轨迹进行未来想象
        bs = trajectory_offset.shape[0]
        
        # ---------------------------- feature preparetion -----------------------
        bev_flatten_feature = result['flatten_bev_feature_1'].detach()
        bev_flatten_feature = self.layernorm_for_bev(bev_flatten_feature)
        bev_flatten_feature = bev_flatten_feature + self._keyval_embedding.weight[None, :, :]
        latent_future_feature = result['latent_future'].detach().flatten(-2, -1).permute(0, 2, 1)
        latent_future_feature = latent_future_feature.chunk(2, dim=0)[0]
        #import pdb;pdb.set_trace()
        latent_future_feature = latent_future_feature + self._keyval_embedding.weight[None, :, :]
        latent_future_feature = latent_future_feature.reshape(bs, 16, 64, 256) # [bs, 16, 64, 256]
        #import pdb;pdb.set_trace()
        # ---------------------------- preparetion end     -----------------------

        expand_trajectory_anchors = trajectory_anchors.unsqueeze(0).repeat(bs, 1, 1, 1)
        output_trajectory = expand_trajectory_anchors + trajectory_offset
        final_rewards = result['final_reward'].detach()
        topk_indices = torch.topk(final_rewards, k=16, dim=1).indices
        topk_trajectories = torch.gather(output_trajectory, dim=1, index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 3))
        metric_cache = {}
        for token in unique_tokens:
            path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, 'rb') as f:
                metric_cache[token] = pickle.load(f)
        rewards, scores = self.reward_fn(topk_trajectories, unique_tokens, metric_cache)
        topk_trajectories_feature = self.pos_embed(topk_trajectories.flatten(-2,-1))
        # 
        B, K, N, C = latent_future_feature.shape
        bev_flatten_feature = bev_flatten_feature.unsqueeze(1).repeat(1, K, 1, 1)
        traj_out = self.cross_cur_bev(topk_trajectories_feature.reshape(B*K, 1, C), bev_flatten_feature.reshape(B*K, N, C))
        traj_out = self.cross_fut_bev(traj_out, latent_future_feature.reshape(B*K, N, C))
        traj_out = traj_out.reshape(B,K,C)
        status_feature = features['status_feature'].detach()
        status_encoding = self._status_encoding(status_feature).unsqueeze(1).repeat(1, K, 1)
        #import pdb;pdb.set_trace()
        traj_out = traj_out + status_encoding

        for k, head in self.heads.items():
            reward[k] = head(traj_out)
        reward.update({'rewards': rewards})
        reward.update({'scores': scores})
        # import pdb;pdb.set_trace()
        return reward


    def reward_fn(
        self,
        pred_traj,
        tokens_list,
        cache_dict,
    ) -> torch.Tensor:
        """Calculates PDM scores for a batch of predicted trajectories."""
        pred_np = pred_traj.detach().cpu().numpy()
        bs = pred_np.shape[0]

        reward_keys = [
            'no_at_fault_collisions',
            'drivable_area_compliance',
            'driving_direction_compliance',
            'ego_progress',
            'time_to_collision_within_bound',
            'comfort',
        ]

        rewards = {
            k: [[] for _ in range(bs)]
            for k in reward_keys + ['score']
        }
        for i, token in enumerate(tokens_list):
            metric_cache = cache_dict[token]
            for j in range(pred_np[i].shape[0]):
                trajectory = Trajectory(pred_np[i][j])
                pdm_result = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=trajectory,
                    future_sampling=self.simulator.proposal_sampling,
                    simulator=self.simulator,
                    scorer=self.train_scorer,
                )
                pdm_score_dict = asdict(pdm_result)
                for k in rewards:
                    rewards[k][i].append(pdm_score_dict[k])

        reward_tensor = torch.stack(
            [
                torch.tensor(rewards[k], device=pred_traj.device, dtype=torch.float32)
                for k in reward_keys
            ],
            dim=-1
        )  # [bs, num_traj, num_rewards]

        score_tensor = torch.tensor(
            rewards['score'], device=pred_traj.device, dtype=torch.float32
        )
        #import pdb;pdb.set_trace()
        #print()
        return reward_tensor, score_tensor


    def weighted_reward_calculation(self, im_rewards, sim_rewards) -> torch.Tensor:
        """
        Calculate the final reward for each trajectory based on the given weights.

        Args:
            im_rewards (torch.Tensor): Imitation rewards for each trajectory. Shape: [batch_size, num_traj]
            sim_rewards (List[torch.Tensor]): List of metric rewards for each trajectory. Each tensor shape: [batch_size, num_traj]
            w (List[float]): List of weights for combining the rewards.

        Returns:
            torch.Tensor: Final weighted reward for each trajectory. Shape: [batch_size, num_traj]
        """
        assert len(sim_rewards) == 5, "Expected 4 metric rewards: S_NC, S_DAC, S_TTC, S_EP, S_COMFORT"
        # Extract metric rewards
        w = self.reward_weights
        S_NC, S_DAC, S_EP, S_TTC, S_COMFORT = sim_rewards
        S_NC, S_DAC, S_EP, S_TTC, S_COMFORT = S_NC.squeeze(-1), S_DAC.squeeze(-1), S_EP.squeeze(-1), S_TTC.squeeze(-1), S_COMFORT.squeeze(-1)
        #self.metric_keys = ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort']
        # Calculate assembled cost based on the provided formula
        assembled_cost = (
            w[0] * torch.log(im_rewards) +
            w[1] * torch.log(S_NC) +
            w[2] * torch.log(S_DAC) +
            w[3] * torch.log(5 * S_TTC + 2 * S_COMFORT + 5 * S_EP)
        )
        return assembled_cost
