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
from env_cfg import EnvConfig
from mani_skill.utils.building import actors


@register_env("ThrowCubePandas-v1", max_episode_steps=200, override=True)
class ThrowCubePandasEnv(BaseEnv):

    def __init__(self, *args, robot_uids="panda", **kwargs):
        self.env_cfg = EnvConfig()

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
            self.cube_thrown = torch.zeros(b, dtype=torch.bool, device=self.device)
            self.max_cube_height = torch.zeros(b, device=self.device)
            self.throw_detected = torch.zeros(b, dtype=torch.bool, device=self.device)


    
    def evaluate(self) -> dict:
        # get cube and goalposition
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_region.pose.p

        # check if cube is in goal region
        distance_to_goal = torch.linalg.norm(cube_pos[..., :2] - goal_pos[..., :2], axis=1)
        in_goal_region = distance_to_goal < self.env_cfg.goal_radius

        # get cube height and velocity
        cube_height = cube_pos[..., 2]
        cube_velocity = torch.linalg.norm(self.cube.linear_velocity, axis=1)

        # update max height
        self.max_cube_height = torch.maximum(self.max_cube_height, cube_height)

        # check if cube is thrown
        table_height = self.env_cfg.table_height
        airborne = table_height + self.env_cfg.throw_air_threshold
        is_current_airborne = (cube_height > airborne) & (cube_velocity > self.env_cfg.throw_velocity_threshold)
        self.throw_detected = self.throw_detected | is_current_airborne

        # check if cube has landed
        has_landed = cube_height <= table_height + self.env_cfg.cube_halfsize

        success = in_goal_region & self.throw_detected & has_landed

        # additonal information
        tcp_pos = self.agent.robot.links_map["panda_hand_tcp"].pose.p
        tcp_to_cube_dist = torch.linalg.norm(tcp_pos - cube_pos, axis=1)
        
        # check if grasping
        is_grasping = self.agent.is_grasping(self.cube)

        return {
            "success": success,
            "distance_to_goal": distance_to_goal,
            "in_goal_region": in_goal_region,
            "throw_detected": self.throw_detected,
            "is_currently_airborne": is_current_airborne,
            "has_landed": has_landed,
            "max_cube_height": self.max_cube_height,
            "tcp_to_cube_dist": tcp_to_cube_dist,
            "is_grasping": is_grasping,
            "cube_height": cube_height,
        }