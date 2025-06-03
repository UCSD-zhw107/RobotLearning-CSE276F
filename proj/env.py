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
from env_cfg import EnvConfig, RewardConfig
from mani_skill.utils.building import actors
from typing import Any, Dict
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("ThrowCubePandas-v1", max_episode_steps=200, override=True)
class ThrowCubePandasEnv(BaseEnv):

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
            
            # randomize cube and goal position
            self._randomize_cube_position(b)
            self._randomize_goal_position(b)
            
            # initialize throwing state tracking
            self.prev_is_grasping = torch.zeros(b, dtype=torch.bool, device=self.device)


    def _check_release(self):
        # check if cube is released
        is_grasping = self.agent.is_grasping(self.cube)
        just_released = self.prev_is_grasping & ~is_grasping
        self.prev_is_grasping = is_grasping
        
        # velcoity check
        vel_xy = self.cube.linear_velocity[..., :2]
        speed_xy = torch.linalg.norm(vel_xy, dim=-1)
        speed = torch.linalg.norm(self.cube.linear_velocity,  dim=-1)
        vel_ok = (speed_xy > self.env_cfg.throw_vel_xy_threshold) | (speed < self.env_cfg.throw_vel_threshold)

        # height check
        cube_height = self.cube.pose.p[..., 2]
        table_height = self.env_cfg.table_height
        airborne = table_height + self.env_cfg.throw_air_threshold
        height_ok = (cube_height > airborne)

        # check if cube is released
        return just_released & vel_ok & height_ok


    def evaluate(self) -> dict:
        # get cube and goalposition
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_region.pose.p

        # check if cube is in goal region
        distance_xy_to_goal = torch.linalg.norm(cube_pos[..., :2] - goal_pos[..., :2], dim=1)
        in_goal_region = distance_xy_to_goal < self.env_cfg.goal_radius

        # check if cube is thrown
        is_released = self._check_release()

        # check if cube has landed
        table_height = self.env_cfg.table_height
        cube_height = cube_pos[..., 2]
        has_landed = cube_height <= table_height + self.env_cfg.cube_halfsize

        success = in_goal_region & has_landed

        return {
            "success": success,
            "distance_xy_to_goal": distance_xy_to_goal,
            "in_goal_region": in_goal_region,
            "has_landed": has_landed,
            "is_released": is_released,
            "cube_height": cube_height,
        }
    

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        
        if "state" in self._obs_mode:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                cube_velocity=self.cube.linear_velocity,
                goal_pos=self.goal_region.pose.p,
                tcp_velocity=self.agent.tcp.linear_velocity,
            )
        
        return obs


    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # agent obs
        agent_obs = obs['agent']
        extra_obs = obs['extra']
        
        tcp_pos = extra_obs["tcp_pose"][:, :3]
        cube_pos = extra_obs["cube_pose"][:, :3]
        goal_pos = extra_obs["goal_pos"][:, :3]
        
        # stage 1: Reach and grasp cube
        tcp_to_cube_dist = torch.linalg.norm(tcp_pos - cube_pos, dim=1)
        reaching_reward = self.reward_cfg.approach_reward_weight * (1 - torch.tanh(5.0 * tcp_to_cube_dist))
        is_grasping = self.agent.is_grasping(self.cube)
        grasping_reward = self.reward_cfg.grasp_reward_weight * is_grasping

        # stage 2: Lift cube
        cube_height = cube_pos[..., 2]
        lift_reward = self.reward_cfg.lift_reward_weight * (1 - torch.tanh(5.0 * (cube_height - self.reward_cfg.target_lift_height)))
        
        # stage 3: Throw cube
        is_released = info["is_released"]
        # velocity alignment reward
        cube_vel = extra_obs["cube_velocity"]
        velocity_direction = cube_vel / (torch.linalg.norm(cube_vel, dim=-1, keepdim=True) + 1e-6)
        goal_direction = goal_pos - cube_pos
        goal_direction[..., 2] = 0
        goal_direction = goal_direction / (torch.linalg.norm(goal_direction, dim=-1, keepdim=True) + 1e-6)
        alignment = (velocity_direction * goal_direction).sum(dim=-1) * self.reward_cfg.direction_align_reward_weight
        # velocity magnitude reward
        speed = torch.linalg.norm(cube_vel, dim=-1)
        speed_bonus = torch.tanh(0.5 * speed) * self.reward_cfg.throw_reward_weight

        reward = (reaching_reward + grasping_reward + lift_reward) * is_grasping
        reward += (alignment + speed_bonus) * is_released

        # stage 4: Success
        if "success" in info:
            reward[info["success"]] += self.reward_cfg.success_reward_weight
        return reward
    

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 30.0
        return self.compute_dense_reward(obs=obs, action=action, info=info)
