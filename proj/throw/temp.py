import sapien
import numpy as np
import gymnasium as gym
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers.record import RecordEpisode
import torch
from transforms3d.euler import euler2quat
from throw.env_cfg import EnvConfig, RewardConfig
from mani_skill.utils.building import actors
from typing import Any, Dict
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("TempTask-v1", max_episode_steps=200, override=True)
class TempTaskEnv(BaseEnv):

    def __init__(self, *args, robot_uids="panda", **kwargs):
        self.env_cfg = EnvConfig()
        self.reward_cfg = RewardConfig()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)



    @property
    def _default_human_render_camera_configs(self):
        # default camera config
        pose = sapien_utils.look_at(eye=self.env_cfg.eye, target=self.env_cfg.target)
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])


    def _build_cube(self):
        # build a cube
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.env_cfg.cube_halfsize,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.env_cfg.cube_halfsize]),
        )
        self.cube.set_mass(self.env_cfg.cube_mass)



    def _build_target(self):
        # build a goal region
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.env_cfg.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )

    def _load_scene(self, options: dict):
        # load table top scene
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        # build cube
        self._build_cube()
        # build goal region
        self._build_target()
        


    def _randomize_cube_position(self, num_envs: int):
        # randomize cube position
        with torch.device(self.device):
            # sample cube position
            xyz = torch.zeros((num_envs, 3))
            xyz[..., 0] = (torch.rand((num_envs)) * 2 - 1) * 0.3 - 0.1
            xyz[..., 1] = torch.rand((num_envs)) * 0.2 + 0.5
            xyz[..., 2] = self.env_cfg.table_height + self.env_cfg.cube_halfsize
        
            cube_pose = Pose.create_from_pq(p=xyz, q=np.array([1, 0, 0, 0]))
            self.cube.set_pose(cube_pose)

    def _randomize_goal_position(self, num_envs: int):
        # randomize goal position
        with torch.device(self.device):
            # sample goal position
            xyz_goal = torch.zeros((num_envs, 3))
            xyz_goal[..., 0] = (torch.rand((num_envs)) * 2 - 1) * 0.3 - 0.1
            xyz_goal[..., 1] = torch.rand((num_envs)) * 0.2 - self.env_cfg.goal_distance + self.env_cfg.goal_radius
            table_top_z = self.env_cfg.table_height
            goal_z = table_top_z + 0.001
            xyz_goal[..., 2] = goal_z
            
            # set goal position
            goal_pose = Pose.create_from_pq(p=xyz_goal,q=euler2quat(0, np.pi / 2, 0))
            self.goal_region.set_pose(goal_pose)




    def _initialize_agent_position(self, num_envs: int):
        # initialize agent position
        with torch.device(self.device):
            agent_poisiton = torch.tensor(self.env_cfg.agent_p)
            agent_p = torch.zeros((num_envs, 3))
            agent_p[..., :2] = agent_poisiton[:2]
            agent_p[..., 2] = self.env_cfg.table_height + agent_poisiton[2]
            agent_pose = Pose.create_from_pq(p=agent_p,q=self.env_cfg.agent_q)
            self.agent.robot.set_pose(agent_pose)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # initialize episode
        with torch.device(self.device):
            b = len(env_idx)
            
            # initialize table scene
            self.table_scene.initialize(env_idx)

            # initialize agent position
            self._initialize_agent_position(b)
            self.default_tcp_pose = self.agent.tcp.pose.p.clone()
            
            # randomize cube and goal position
            self._randomize_cube_position(b)
            self._randomize_goal_position(b)

            if not hasattr(self, 'has_lifted_once'):
                self.has_lifted_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if not hasattr(self, 'has_released'):
                self.has_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if not hasattr(self, 'grasp_step_counter'):
                self.grasp_step_counter = torch.full((self.num_envs,), 0, device=self.device)

            self.has_lifted_once[env_idx] = False
            self.has_released[env_idx] = False
            self.grasp_step_counter[env_idx] = 0

    def evaluate(self) -> dict:
        # get cube and goalposition
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_region.pose.p
        is_grasping = self.agent.is_grasping(self.cube)

        # check if cube is in goal region
        distance_xy_to_goal = torch.linalg.norm(cube_pos[..., :2] - goal_pos[..., :2], dim=1)
        in_goal_region = distance_xy_to_goal < self.env_cfg.goal_radius

        # check if cube has landed
        table_height = self.env_cfg.table_height
        cube_height = cube_pos[..., 2]
        has_landed = cube_height <= table_height + self.env_cfg.cube_halfsize


        # check if cube is lifted to lift target
        is_cube_lifted = cube_height >= self.env_cfg.table_height + self.env_cfg.lift_height
        reach_lift_target = is_grasping & is_cube_lifted

        # check if cube has lifted once
        self.has_lifted_once = self.has_lifted_once | (reach_lift_target)

        # check if cube is released
        just_released = ~is_grasping & self.has_lifted_once & ~self.has_released
        self.has_released = self.has_released | just_released

        success = has_landed & in_goal_region & self.has_lifted_once

        # update grasp step counter
        currently_grasping_after_lift = is_grasping & self.has_lifted_once & ~self.has_released
        self.grasp_step_counter[currently_grasping_after_lift] += 1
        self.grasp_step_counter[~currently_grasping_after_lift] = 0
        max_grasp_steps = self.env_cfg.max_grasp_time
        grasp_timeout = (
            self.has_lifted_once & 
            is_grasping & 
            (self.grasp_step_counter > max_grasp_steps)
        )
  
        fail = grasp_timeout

        return {
            "success": success,
            "distance_xy_to_goal": distance_xy_to_goal,
            "in_goal_region": in_goal_region,
            "has_landed": has_landed,
            "cube_height": cube_height,
            "fail": fail,
            "has_lifted_once": self.has_lifted_once,
            "has_released": self.has_released,
            "just_released": just_released,
        }
    

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        
        if "state" in self._obs_mode:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                goal_pos=self.goal_region.pose.p,
                cube_velocity=self.cube.linear_velocity,
                tcp_velocity=self.agent.tcp.get_linear_velocity(),
                tcp_to_cube_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                cube_to_goal_pos=self.goal_region.pose.p - self.cube.pose.p,
                has_lifted_once=info.get("has_lifted_once", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)),
                has_released=info.get("has_released", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)),
            )
        
        return obs



    def _compute_cube_velocity_reward(self, cube_velocity, goal_pos, cube_pos):
        # Calculate optimal release conditions
        cube_to_goal_xy = goal_pos[:, :2] - cube_pos[:, :2]
        distance_to_goal_xy = torch.linalg.norm(cube_to_goal_xy, dim=1)
        
        # TCP velocity for momentum transfer
        cube_velocity_xy = cube_velocity[:, :2]
        cube_vel_magnitude = torch.linalg.norm(cube_velocity_xy, dim=1)
        cube_vel_direction = cube_velocity_xy / (cube_vel_magnitude.unsqueeze(-1) + 1e-6)
        goal_direction = cube_to_goal_xy / (distance_to_goal_xy.unsqueeze(-1) + 1e-6)
        
        # Reward for cube moving in the right direction before release
        cube_alignment = (cube_vel_direction * goal_direction).sum(dim=1)
        cube_direction_reward = torch.clamp(cube_alignment, 0, 1) * 15.0

        # Reward for TCP velocity magnitude (for momentum transfer)
        cube_speed_reward = torch.tanh(cube_vel_magnitude * 0.5) * 20.0
        
        return cube_direction_reward, cube_speed_reward


    def _compute_tcp_velocity_reward(self, goal_pos: Array, cube_pos: Array):
        # Calculate optimal release conditions
        cube_to_goal_xy = goal_pos[:, :2] - cube_pos[:, :2]
        distance_to_goal_xy = torch.linalg.norm(cube_to_goal_xy, dim=1)
        
        # TCP velocity for momentum transfer
        self.agent.tcp.get_angular_velocity()
        tcp_velocity = self.agent.tcp.get_linear_velocity()
        tcp_velocity_xy = tcp_velocity[:, :2]
        
        # Direction alignment between TCP velocity and cube-to-goal
        tcp_vel_magnitude = torch.linalg.norm(tcp_velocity_xy, dim=1)
        tcp_vel_direction = tcp_velocity_xy / (tcp_vel_magnitude.unsqueeze(-1) + 1e-6)
        goal_direction = cube_to_goal_xy / (distance_to_goal_xy.unsqueeze(-1) + 1e-6)
        
        # Reward for TCP moving in the right direction before release
        tcp_alignment = (tcp_vel_direction * goal_direction).sum(dim=1)
        tcp_direction_reward = torch.clamp(tcp_alignment, 0, 1) * 15.0
        
        # Reward for TCP velocity magnitude (for momentum transfer)
        tcp_speed_reward = torch.tanh(tcp_vel_magnitude * 0.5) * 20.0
        
        # Height-based release timing reward
        optimal_release_height = self.env_cfg.table_height + self.env_cfg.lift_height * 0.8
        height_difference = torch.abs(cube_pos[:, 2] - optimal_release_height)
        height_timing_reward = torch.exp(-height_difference * 2.0) * 5.0

        return tcp_direction_reward, tcp_speed_reward, height_timing_reward


    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        
        is_grasping = self.agent.is_grasping(self.cube)
        reward = torch.zeros(is_grasping.shape[0], device=self.device)
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_region.pose.p
        cube_velocity = self.cube.linear_velocity
        
        # Get evaluation info
        has_lifted_once = info.get("has_lifted_once", torch.zeros_like(is_grasping, dtype=torch.bool))
        has_released = info.get("has_released", torch.zeros_like(is_grasping, dtype=torch.bool))
    
        # stage 1: Reach and grasp cube
        stage1_mask = ~has_lifted_once & ~has_released
        if stage1_mask.any():
            tcp_to_cube_dist = torch.linalg.norm(cube_pos - tcp_pos, dim=1)
            reaching_reward = (1 - torch.tanh(5.0 * tcp_to_cube_dist))
            grasping_reward = is_grasping
            

            # lift cube
            cube_height = cube_pos[..., 2]
            lift_progress = torch.clamp(
                (cube_height - self.env_cfg.table_height - self.env_cfg.cube_halfsize) / self.env_cfg.lift_height,
                0, 1
            )
            lifting_reward = lift_progress * is_grasping.float() * 2.0
            reward[stage1_mask] = (
                reaching_reward[stage1_mask] + 
                grasping_reward[stage1_mask] + 
                lifting_reward[stage1_mask]
            )

        # Stage 2: Prepare for release
        stage2_mask = has_lifted_once & ~has_released
        if stage2_mask.any():

            # get cube velocity# 1. PENALIZE TCP getting too close to goal (prevent extending)
            tcp_to_goal_xy_dist = torch.linalg.norm(goal_pos[:, :2] - tcp_pos[:, :2], dim=1)
            min_throwing_distance = 0.3  # Minimum distance to maintain from goal
            too_close_penalty = torch.exp(-tcp_to_goal_xy_dist / min_throwing_distance) * 20.0


            # Calculate optimal release conditions
            cube_to_goal_xy = goal_pos[:, :2] - cube_pos[:, :2]
            distance_to_goal_xy = torch.linalg.norm(cube_to_goal_xy, dim=1)
            
            # TCP velocity for momentum transfer
            self.agent.tcp.get_angular_velocity()
            tcp_velocity = self.agent.tcp.get_linear_velocity()
            tcp_velocity_xy = tcp_velocity[:, :2]
            
            # Direction alignment between TCP velocity and cube-to-goal
            tcp_vel_magnitude = torch.linalg.norm(tcp_velocity_xy, dim=1)
            tcp_vel_direction = tcp_velocity_xy / (tcp_vel_magnitude.unsqueeze(-1) + 1e-6)
            goal_direction = cube_to_goal_xy / (distance_to_goal_xy.unsqueeze(-1) + 1e-6)
            
            # Reward for TCP moving in the right direction before release
            tcp_alignment = (tcp_vel_direction * goal_direction).sum(dim=1)
            tcp_direction_reward = torch.clamp(tcp_alignment, 0, 1) * 15.0
            
            # Reward for TCP velocity magnitude (for momentum transfer)
            tcp_speed_reward = torch.tanh(tcp_vel_magnitude * 0.5) * 20.0
            
            # Height-based release timing reward
            optimal_release_height = self.env_cfg.table_height + self.env_cfg.lift_height * 0.5
            height_difference = torch.abs(cube_pos[:, 2] - optimal_release_height)
            height_timing_reward = torch.exp(-height_difference * 2.0) * 5.0

            # joint vel reward
            joint_vel = self.agent.robot.get_qvel()[..., :-2]
            joint_vel_magnitude = torch.linalg.norm(joint_vel, dim=1)
            joint_vel_reward = torch.tanh(joint_vel_magnitude * 0.5) * 10.0
            
            # Apply rewards to grasping agents
            grasping_mask = stage2_mask & is_grasping
            reward[grasping_mask] += (
                tcp_direction_reward[grasping_mask] + 
                tcp_speed_reward[grasping_mask] + 
                height_timing_reward[grasping_mask] - too_close_penalty[grasping_mask] + joint_vel_reward[grasping_mask]
            )
            
            # Penalize slow movement when grasping
            reward[grasping_mask] -= torch.exp(-tcp_vel_magnitude[grasping_mask]) * 2.0

        # Stage 3: Just Release
        stage3_mask = info['just_released']
        if stage3_mask.any():
            cube_direction_reward, cube_speed_reward = self._compute_cube_velocity_reward(cube_velocity, goal_pos, cube_pos)
            reward[stage3_mask] += cube_direction_reward[stage3_mask] + cube_speed_reward[stage3_mask]

            # joint vel reward
            joint_vel = self.agent.robot.get_qvel()[..., :-2]
            joint_vel_magnitude = torch.linalg.norm(joint_vel, dim=1)
            joint_vel_reward = torch.tanh(joint_vel_magnitude * 0.5) * 10.0

            reward[stage3_mask] += joint_vel_reward[stage3_mask]

            
        # reward for NOT grasping (i.e., released)
        released = ~is_grasping & has_lifted_once
        reward[released] += 7.0  

        # penalty for still grasping
        still_grasping = is_grasping & has_lifted_once
        reward[still_grasping] -= 7.0  

        # Stage 3: After release
        stage4_mask = has_released
        if stage4_mask.any():
            static_reward = 1 - torch.tanh(
                5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], dim=1)
            )
            reward[stage4_mask] += static_reward[stage4_mask] * 10.0

        
        # success
        if "success" in info:
            reward[info["success"]] += 50.0

        # far penalty
        return reward
    

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 10.0 # Maximum reward from success reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
