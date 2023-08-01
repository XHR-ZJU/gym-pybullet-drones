import IRL_env
from HoverAviary import HoverAviary
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch
import gym
import os

AGGR_PHY_STEPS = 5
EPISODE_REWARD_THRESHOLD = -0
DEFAULT_STEPS = 100000
has_model = True
filename = os.path.join("/home/xhr/reinforcement_learning/gym-pybullet-drones/results", datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
obs = ObservationType('kin')
act = ActionType('one_d_rpm')

if __name__ == '__main__':
    if has_model:
        model = PPO.load(
            "/home/xhr/reinforcement_learning/gym-pybullet-drones/results/08.01.2023_14.18.53/success_model.zip"
            )
        env = gym.make("hover-aviary-v0",
                        aggregate_phy_steps=AGGR_PHY_STEPS,
                        obs=obs,
                        act=act
                        )
        obs = env.reset()
        total_reward = 0
        while True:
            # time.sleep(1 / 50.0)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # env.render()
            total_reward += reward
            if done:
                print(total_reward)
                obs = env.reset()
                total_reward = 0
    else:
        # env = IRL_env.IRL_env()
        # env.Set_Goal(np.array([0.5, 0.0, 1.5]))
        onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                                net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
                                ) # or None
        
        parallel = 1
        sa_env_kwargs = dict(aggregate_phy_steps=AGGR_PHY_STEPS, obs=obs, act=act)
        train_env = make_vec_env(HoverAviary,
                                env_kwargs=sa_env_kwargs,
                                n_envs=parallel,
                                seed=0
                                )
        eval_env = gym.make("hover-aviary-v0",
                            aggregate_phy_steps=AGGR_PHY_STEPS,
                            obs=obs,
                            act=act
                            )
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD,
                                                            verbose=1
                                                            )
        eval_callback = EvalCallback(eval_env,
                                        callback_on_new_best=callback_on_best,
                                        verbose=1,
                                        best_model_save_path=filename+'/',
                                        log_path=filename+'/',
                                        eval_freq=int(2000/parallel),
                                        deterministic=True,
                                        render=False
                                        )  
        model = PPO(a2cppoMlpPolicy,
                    train_env,
                    policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    )
        model.learn(total_timesteps=DEFAULT_STEPS, #int(1e12),
                    callback=eval_callback,
                    log_interval=100,
                    )
        model.save(filename+'/success_model.zip')
    
