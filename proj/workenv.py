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
from test_env_cfg import TestEnvConfig
from mani_skill.utils.building import actors
from typing import Any, Dict
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("WorkEnv-v1", max_episode_steps=200, override=True)
class WorkEnv(BaseEnv):

    def __init__(self, *args, robot_uids="panda", **kwargs):
        self.env_cfg = TestEnvConfig()
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


    def _build_bin(self):
        bin_builder = self.scene.create_actor_builder()
        # bottom
        bin_builder.add_box_collision(
            half_size=self.env_cfg.bin_bottom_halfsize,
            pose = sapien.Pose(p=[0,0, -self.env_cfg.bin_bottom_halfsize[2]], q=[1, 0, 0, 0]),
        )
        bin_builder.add_box_visual(
            half_size=self.env_cfg.bin_bottom_halfsize,
            pose = sapien.Pose(p=[0,0, -self.env_cfg.bin_bottom_halfsize[2]], q=[1, 0, 0, 0]),
        )
        # walls
        wall_pose_offsets = [
            [0, self.env_cfg.bin_bottom_halfsize[1] - self.env_cfg.bin_wall_halfsize[1], self.env_cfg.bin_wall_halfsize[2]],   # front
            [0, -self.env_cfg.bin_bottom_halfsize[1] + self.env_cfg.bin_wall_halfsize[1], self.env_cfg.bin_wall_halfsize[2]],  # back
            [self.env_cfg.bin_bottom_halfsize[0] - self.env_cfg.bin_wall_halfsize[1], 0, self.env_cfg.bin_wall_halfsize[2]],   # right
            [-self.env_cfg.bin_bottom_halfsize[0] + self.env_cfg.bin_wall_halfsize[1], 0, self.env_cfg.bin_wall_halfsize[2]],  # left
        ]
        for i, offset in enumerate(wall_pose_offsets):
            if i >= 2:  
                quat = sapien.Pose(q=[np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]).q  
            else:  
                quat = [1, 0, 0, 0]  
            bin_builder.add_box_collision(
                half_size=self.env_cfg.bin_wall_halfsize,
                pose=sapien.Pose(p=offset, q=quat)
            )
            bin_builder.add_box_visual(
                half_size=self.env_cfg.bin_wall_halfsize,
                pose=sapien.Pose(p=offset, q=quat),
            )
        bin_builder.initial_pose = sapien.Pose(p=[0.2, 0, 0.1], q=[1, 0, 0, 0])
        self.bin = bin_builder.build_dynamic(name="bin")


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
        # build bin
        self._build_bin()
        


    def _randomize_cube_bin_position(self, num_envs: int):
        # randomize cube position
        with torch.device(self.device):
            # sample cube position
            xyz = torch.zeros((num_envs, 3))
            xyz[..., 0] = (torch.rand((num_envs)) * 2 - 1) * 0.3 - 0.1
            xyz[..., 1] = torch.rand((num_envs)) * 0.2 + 0.5
            xyz[..., 2] = self.env_cfg.table_height + 0.05
        
            bin_pose = Pose.create_from_pq(p=xyz, q=np.array([1, 0, 0, 0]))
            self.bin.set_pose(bin_pose)

            # get cube position
            cube_xyz = xyz + torch.tensor([0, 0, 0.05])
            cube_pose = Pose.create_from_pq(p=cube_xyz, q=np.array([1, 0, 0, 0]))
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
            self._randomize_cube_bin_position(b)
            self._randomize_goal_position(b)



    def _get_bin_grasp_points(self) -> torch.Tensor:
        
        bin_pose = self.bin.pose
        bin_pos = bin_pose.p
        
        
        #wall_height = self.env_cfg.bin_wall_halfsize[2] 
        wall_height = 0.00
        wall_thickness = self.env_cfg.bin_wall_halfsize[1]  
        
        
        grasp_points = {
            'front': bin_pos[..., :] + torch.tensor([0, -self.env_cfg.bin_bottom_halfsize[1], wall_height], device=bin_pos.device),
            'back': bin_pos[..., :] + torch.tensor([0, self.env_cfg.bin_bottom_halfsize[1], wall_height], device=bin_pos.device),
            'right': bin_pos[..., :] + torch.tensor([-self.env_cfg.bin_bottom_halfsize[0] , 0, wall_height], device=bin_pos.device),
            'left': bin_pos[..., :] + torch.tensor([self.env_cfg.bin_bottom_halfsize[0], 0, wall_height], device=bin_pos.device)
        }

        # use front wall grasp point as default
        return grasp_points['left'], grasp_points['right']


    def _is_cube_in_bin(self):
        cube_pose = self.cube.pose.p   # (b, 3)
        bin_pose = self.bin.pose.p     # (b, 3)
        bin_halfsize = torch.tensor(self.env_cfg.bin_bottom_halfsize, device=cube_pose.device)  # (3,)
        lower_bound = bin_pose[..., :2] - bin_halfsize[:2]
        upper_bound = bin_pose[..., :2] + bin_halfsize[:2]
        inside_xy = (cube_pose[..., 0] >= lower_bound[..., 0]) & (cube_pose[..., 0] <= upper_bound[..., 0]) & \
                    (cube_pose[..., 1] >= lower_bound[..., 1]) & (cube_pose[..., 1] <= upper_bound[..., 1])

        expected_z = bin_pose[..., 2] + self.env_cfg.bin_bottom_halfsize[2] + self.env_cfg.cube_halfsize
        z_close = torch.abs(cube_pose[..., 2] - expected_z) < 0.05
        return inside_xy & z_close



    def evaluate(self) -> dict:
        # get cube and goalposition
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_region.pose.p
        bin_pos = self.bin.pose.p
        is_grasping = self.agent.is_grasping(self.bin)

        # check if cube in bin
        is_cube_in_bin = self._is_cube_in_bin()
        
        # check if bin in goal region
        distance_xy_to_goal = torch.linalg.norm(cube_pos[..., :2] - goal_pos[..., :2], dim=1)
        is_bin_in_goal_region = distance_xy_to_goal < self.env_cfg.goal_radius

        # check if bin on table
        is_bin_on_table = bin_pos[..., 2] <= self.env_cfg.table_height + self.env_cfg.bin_wall_halfsize[2] + 1e-3

        # get grasp points
        self.grasp_point_left, self.grasp_point_right = self._get_bin_grasp_points()
        

        success = is_cube_in_bin & is_bin_in_goal_region & is_bin_on_table


        return {
            "success": success,
            "bin_in_goal_region": is_bin_in_goal_region,
            "cube_in_bin": is_cube_in_bin,
            "bin_on_table": is_bin_on_table,
            "is_grasping": is_grasping,
        }
    

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        
        if "state" in self._obs_mode:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                bin_pose=self.bin.pose.raw_pose,
                goal_pos=self.goal_region.pose.p,
                #grasp_point=self.grasp_point,
                #tcp_to_grasp=self.grasp_point - self.agent.tcp.pose.p,
                tcp_to_bin_pos=self.bin.pose.p - self.agent.tcp.pose.p,
                bin_to_goal_pos=self.goal_region.pose.p - self.bin.pose.p,
                cube_to_bin_pos=self.bin.pose.p - self.cube.pose.p,
            )
        
        return obs


    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):

        # basic info
        bin_pos = self.bin.pose.p
        goal_pos = self.goal_region.pose.p
        tcp_pos = self.agent.tcp.pose.p
        #grasp_point = self.grasp_point
        is_grasping = self.agent.is_grasping(self.bin)
        reward = torch.zeros(self.num_envs, device=self.device)
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )

        # Reaching and grasping reward
        # determine which grasp point to use
        tcp_to_grasp_dist_left = torch.linalg.norm(self.grasp_point_left - tcp_pos, dim=1)
        tcp_to_grasp_dist_right = torch.linalg.norm(self.grasp_point_right - tcp_pos, dim=1)
        #tcp_to_grasp_dist = torch.minimum(tcp_to_grasp_dist_left, tcp_to_grasp_dist_right)
        tcp_to_grasp_dist = tcp_to_grasp_dist_left
        reaching_reward = (1 - torch.tanh(5.0 * tcp_to_grasp_dist))
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ) * (tcp_to_grasp_dist >= 0.05) 
        reaching_reward = reaching_reward * 0.5
        grasping_reward = is_grasping * 3.0

        # Cube in Bin reward
        cube_in_bin_reward = info['cube_in_bin'] * is_grasping

        # Goal reaching reward
        bin_to_goal_dist = torch.linalg.norm(goal_pos - bin_pos, dim=1)
        goal_reaching_reward = (1 - torch.tanh(5.0 * bin_to_goal_dist)) * 5.0 * is_grasping
        
        reward = reaching_reward + grasping_reward + cube_in_bin_reward + goal_reaching_reward 

        # check if success
        if 'success' in info:
            reward[info['success']] += 10.0
        
        return reward
    

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 10.0 # Maximum reward from success reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
