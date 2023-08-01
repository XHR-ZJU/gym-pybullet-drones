import os
import numpy as np
import torch
import torch.nn as nn
from gym import spaces
import cv2 as cv
import pybullet as p
import time
import math
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    def forward(self, input):
        return input.view(input.size(0), -1, 2, 2)
    
class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.LazyConv2d(32, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.LazyConv2d(32, kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(h_dim),
            nn.LazyConvTranspose2d(32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.LazyConvTranspose2d(16, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.LazyConvTranspose2d(image_channels, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )            
        
    def reparameterize(self, mu, logvar):
        with torch.no_grad():
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(device)
            z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        with torch.no_grad():
            mu, logvar = self.fc1(h), self.fc2(h)
            z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        with torch.no_grad():
            h = self.encoder(x)
            z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        with torch.no_grad():
            z = self.fc3(z)
            z = self.decoder(z)
        return z

    def forward(self, x):
        with torch.no_grad():
            z, mu, logvar = self.encode(x)
            z = self.decode(z)
        return z, mu, logvar
    
class RewardNet(nn.Module):
    def __init__(self):
        super(RewardNet, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(64)
        )
        self.mlp2 = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

    def forward(self, depth_encode, rel_pos, goal_vel, cur_vel, cur_q, cur_omg):
        with torch.no_grad():
            x = torch.concat((depth_encode, rel_pos, goal_vel, cur_vel, cur_q, cur_omg), dim=-1)
            x = self.mlp2(self.mlp1(x))
        return x


class IRL_env(BaseAviary):
    """Multi-drone environment class for control applications using vision."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=True,
                 user_debug_gui=True,
                 output_folder='results',
                 max_vel=2.0, max_acc=4.0
                 ):
        """Initialization of an aviary environment for control applications using vision.

        Attribute `vision_attributes` is automatically set to True when calling
        the superclass `__init__()` method.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         vision_attributes=True,
                         output_folder=output_folder
                         )
        self.EPISODE_LEN_SEC = 5
        print("--------------------------------")
        print("init net")
        self.vae_net = VAE(image_channels=1, h_dim=128).to(device)
        self.vae_net.load_state_dict(torch.load('/home/xhr/rl_motion_ws/depth_vae/model/vae128_0723.torch'))
        self.vae_net.eval()
        self.reward_net = RewardNet().to(device)
        self.reward_net.load_state_dict(torch.load('/home/xhr/rl_motion_ws/depth_vae/model/reward128_0723.torch'))
        self.reward_net.eval()
        self.set_final_goal = False
        self.final_goal = np.array([0, 0, 0])
        self.start_pt = np.array([0, 0, 0])
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.leave_path = False
        self.collision = False
        self.arrive = False
        self.obs = []
    
    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([0.,           0.,           0.,           0.])
        # act_upper_bound = np.array([self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        act_upper_bound = np.array([1.0, 1.0, 1.0, 1.0])
        # return spaces.Dict({str(i): spaces.Box(low=act_lower_bound,
        #                                        high=act_upper_bound,
        #                                        dtype=np.float32
        #                                        ) for i in range(self.NUM_DRONES)})
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################
    
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

        """
        #### Observation vector ### X        Y        Z                                Q1   Q2   Q3   Q4     VX       VY       VZ       WX       WY       WZ  
        # odom_lower_bound = np.array([-np.inf, -np.inf, 0.,  -np.inf, -np.inf, -np.inf, -1., -1., -1., -1.,-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        # odom_upper_bound = np.array([np.inf, np.inf, np.inf, np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1., np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])
        odom_lower_bound = np.array([-1., -1., 0.,  -1., -1., -1., -1., -1., -1., -1.,-1., -1., -1., -1., -1., -1.])
        odom_upper_bound = np.array([1.,  1.,  1., 1.,  1.,  1., 1.,  1.,  1.,  1., 1.,  1.,  1.,  1.,  1.,  1.])
        depth_code_lower_bound = np.zeros(128)
        depth_code_upper_bound = np.ones(128)
        state_lower_bound = np.concatenate((odom_lower_bound, depth_code_lower_bound))
        state_upper_bound = np.concatenate((odom_upper_bound, depth_code_upper_bound))
        # return spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=state_lower_bound,
        #                                                              high=state_upper_bound,
        #                                                              dtype=np.float32
        #                                                              )
        #                                          }) for i in range(self.NUM_DRONES)})
        return spaces.Box(low=state_lower_bound, high=state_upper_bound, dtype=np.float32)

    
    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix,
        "rgb", "dep", and "seg" are matrices containing POV camera captures.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

        """
        for i in range(self.NUM_DRONES):
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                # print(self.dep[i])
                #### Printing observation to PNG frames example ############
                # if self.RECORD:
                #     self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                #                       img_input=self.rgb[i],
                #                       path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                #                       frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                #                       )
            # obs[str(i)] = {"state": self._getDroneStateVector(i), \
            #                "dep": self.dep[i]
            #                }
            # cv.imshow("...", self.dep[i])
            # cv.waitKey(0)
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            q = state[3:7]
            vel = state[10:13]
            omg = state[13:16]
            if self.set_final_goal:
                mid_goal, mid_goal_vel = self.calc_mid_goal(pos, self.start_pt, self.final_goal)
            else:
                print("goal has not been set.")
                mid_goal = np.array([0.0, 0.0, 1.0])
                mid_goal_vel = np.array([0.0, 0.0, 0.0])

            # print("mid_goal = ", mid_goal)
            # print("pos = ", pos)
            rel_pos = mid_goal - pos
            if np.linalg.norm(rel_pos) < 0.1:
                self.arrive = True
            
            depth = cv.resize(self.dep[i], (64, 64))
            depth = np.asarray(depth, dtype=np.float32)
            depth = torch.unsqueeze(torch.tensor(depth, dtype=torch.float32), 0).to(device)
            depth = torch.unsqueeze(torch.tensor(depth, dtype=torch.float32), 0).to(device)
            # print("depth.shape = ", depth.shape)
            with torch.no_grad():
                depth_encode = self.vae_net.encoder(depth)
                depth_encode = depth_encode.squeeze(0)
            depth_encode = depth_encode.cpu().numpy()
            # print(depth_encode)
            odom = np.concatenate((rel_pos, mid_goal_vel, q, vel, omg, depth_encode))
            # print(odom)
            # obs[str(i)] = {"state": odom}
        self.detect_collision()
        odom = self._clipAndNormalizeState(odom)
        return odom
    
    def calc_mid_goal(self, p, s, g):
        l = np.dot(p-s, g-s) / np.linalg.norm(g-s)
        p2s = np.linalg.norm(p-s)
        vec = 1.0 / np.linalg.norm(g-s) * (g-s)
        # 如果当前位置离引导路径已经很远了，就直接结束
        if math.sqrt(p2s**2 - l**2) > 9.9:
            self.leave_path = True
            return s+l*vec, np.array([0, 0, 0])
        else:
            self.leave_path = False
        # 如果离起始点很近，又离目标点很远，直接置为10m以外朝向目标点
        if l < 0.5 and np.linalg.norm(g-p) > 12:
            goal_pos = 10.0*vec + s
            goal_vel = self.max_vel * vec
            return goal_pos, goal_vel
        # 如果离目标点很近，或者当前位置已经超过目标点了，就将目标点设为local_goal
        if np.linalg.norm(g-p) < 2 or l > np.linalg.norm(g-s):
            goal_pos = g
            goal_vel = np.array([0.0, 0.0, 0.0])
            return goal_pos, goal_vel
        # 中间的点，找一个点来引导
        while np.linalg.norm(l*vec + s - p) < 10 and l < np.linalg.norm(g-s):
            l = l + 0.1
        goal_pos = l*vec + s
        # 如果引导点距离目标点近到一定阈值，就应该减速
        if np.linalg.norm(l*vec + s - g) < self.max_vel**2 / (2*0.3*self.max_acc):
            goal_vel = math.sqrt(2 * 0.3*self.max_acc * np.linalg.norm(l*vec + s - g)) * vec
            # 很近的时候置零
            if np.linalg.norm(l*vec + s - g) < 0.5:
                goal_vel = np.array([0, 0, 0])
        else:
            # 比较远的时候还是最大速度
            goal_vel = self.max_vel * vec
        # print(goal_pos, goal_vel)
        return goal_pos, goal_vel
    ################################################################################
    
    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : dict[str, ndarray]
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        clipped_action = np.zeros((1, 4))
        clipped_action[0, :] = self.MAX_RPM * np.clip(action, 0, 1.0)
        return clipped_action

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        # 计算到达目标点需要的jerk cost
        state = self._unnormlizeState(self._computeObs())
        # print(state)
        rel_pos = state[0:3]
        goal_vel = state[3:6]
        cur_q = state[6:10]
        cur_vel = state[10:13]
        cur_omg = state[13:16]
        depth_encode = state[16:]
        rel_pos = torch.tensor(rel_pos, dtype=torch.float32).to(device)
        goal_vel = torch.tensor(goal_vel, dtype=torch.float32).to(device)
        cur_q = torch.tensor(cur_q, dtype=torch.float32).to(device)
        cur_vel = torch.tensor(cur_vel, dtype=torch.float32).to(device)
        cur_omg = torch.tensor(cur_omg, dtype=torch.float32).to(device)
        depth_encode = torch.tensor(depth_encode, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = self.reward_net(depth_encode, rel_pos, goal_vel, cur_vel, cur_q, cur_omg)
        jerk_cost = out.item()
        if jerk_cost < 0:
            jerk_cost = 1e9
        # 计算yaw角cost
        q = cur_q.cpu().numpy()
        cur_vel = cur_vel.cpu().numpy()
        rel_pos = rel_pos.cpu().numpy()
        cur_yaw = math.atan2(2*(q[0]*q[1] + q[3]*q[2]), q[3]**2 + q[0]**2 - q[1]**2 - q[2]**2)
        vel_yaw = math.atan2(cur_vel[1], cur_vel[0])
        delta_yaw = cur_yaw - vel_yaw
        while delta_yaw > np.pi:
            delta_yaw -= 2*np.pi
        while delta_yaw < -np.pi:
            delta_yaw += 2*np.pi
        yaw_cost = math.exp(delta_yaw**2)
        # 计算距离cost
        # dist_cost = np.linalg.norm(rel_pos)
        vec = np.linalg.norm(self._getDroneStateVector(0)[0:3] - np.array([0.5, 0.0, 1.5]))
        w_jerk = 0.0
        w_yaw = 0.0
        w_dist = 1.0
        reward = -w_jerk * jerk_cost - w_yaw * yaw_cost - w_dist * vec
        pos = self._getDroneStateVector(0)[0:3]
        if abs(pos[0])>15 or abs(pos[1])>15 or abs(pos[2])>3:
            reward -= 1000
        if self.collision:
            reward -= 1000
        if self.arrive:
            reward += 1000
        # print(reward)
        return reward

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            if self.leave_path:
                return True
            if self.collision:
                return True
            if self.arrive:
                return True
            pos = self._getDroneStateVector(0)[0:3]
            if abs(pos[0])>15 or abs(pos[1])>15 or abs(pos[2])>3:
                return True
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = 15
        MAX_Z = 3

        MAX_ENCODE = 30

        clipped_rel_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_rel_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_goal_vel_xy = np.clip(state[3:5], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_goal_vel_z = np.clip(state[5], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        normalized_rel_pos_xy = clipped_rel_pos_xy / MAX_XY
        normalized_rel_pos_z = clipped_rel_pos_z / MAX_Z
        normalized_goal_vel_xy = clipped_goal_vel_xy / MAX_LIN_VEL_XY
        normalized_goal_vel_z = clipped_goal_vel_z / MAX_LIN_VEL_Z
        normalized_q = state[6:10]
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        normalized_encode = state[16:] / MAX_ENCODE

        norm_and_clipped = np.hstack([normalized_rel_pos_xy,
                                      normalized_rel_pos_z,
                                      normalized_goal_vel_xy,
                                      normalized_goal_vel_z,
                                      normalized_q,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      normalized_encode
                                      ]).reshape(144,)

        return norm_and_clipped
    
    def _unnormlizeState(self, state):
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1
        MAX_XY = 15
        MAX_Z = 3
        MAX_ENCODE = 30

        clipped_rel_pos_xy = state[0:2]
        clipped_rel_pos_z = state[2]
        clipped_goal_vel_xy = state[3:5]
        clipped_goal_vel_z = state[5]
        clipped_vel_xy = state[10:12]
        clipped_vel_z = state[12]

        normalized_rel_pos_xy = clipped_rel_pos_xy * MAX_XY
        normalized_rel_pos_z = clipped_rel_pos_z * MAX_Z
        normalized_goal_vel_xy = clipped_goal_vel_xy * MAX_LIN_VEL_XY
        normalized_goal_vel_z = clipped_goal_vel_z * MAX_LIN_VEL_XY
        normalized_q = state[6:10]
        normalized_vel_xy = clipped_vel_xy * MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z * MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]*np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        normalized_encode = state[16:] * MAX_ENCODE

        unnormlize = np.hstack([normalized_rel_pos_xy,
                                      normalized_rel_pos_z,
                                      normalized_goal_vel_xy,
                                      normalized_goal_vel_z,
                                      normalized_q,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      normalized_encode
                                      ]).reshape(144,)

        return unnormlize
    
    def Set_Goal(self, goal):
        self.set_final_goal = True
        state = self._getDroneStateVector(0)
        self.start_pt = state[0:3]
        self.final_goal = goal

    def detect_collision(self):
        quad_id = self.DRONE_IDS[0]
        pmin, pmax = p.getAABB(quad_id)
        collide_ids = p.getOverlappingObjects(pmin, pmax)
        if collide_ids is not None:
            for collide_id in collide_ids:
                if collide_id[0] != quad_id:
                    self.collision = True
                    return True
            self.collision = False
        return False
    
if __name__ == '__main__':
    env = IRL_env()
    env.Set_Goal(np.array([20.0, 0, 1.0]))
    print("--------------------------------")
    obs = env.reset()
    # while True:
    #     print(env.detect_collision())
    # obs = obs["0"]["dep"]
    # print(obs.shape)
    # obs=cv.applyColorMap(cv.convertScaleAbs(obs,alpha=15),cv.COLORMAP_JET)
    # cv.imshow('depth', obs)
    # cv.waitKey(0)
    print(obs)
    # time.sleep(100)
