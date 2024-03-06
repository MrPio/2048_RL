import os
from time import sleep

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from gym_env import Game2048Env

TOTAL_TIMESTEPS = 10_000
MODEL_DIR = 'PPO'
LOG_DIR = 'logs'
LOAD_MODEL = '10980000.zip'
NEXT_INDEX = 1099
for path in [MODEL_DIR, LOG_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)


def train():
    env = Monitor(Game2048Env(), LOG_DIR)
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR) if LOAD_MODEL == '' else PPO.load(
        f'{MODEL_DIR}/{LOAD_MODEL}', env=env)
    for i in range(NEXT_INDEX, NEXT_INDEX + 999):
        model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL_DIR)
        model.save(f'{MODEL_DIR}/{TOTAL_TIMESTEPS * i}')
        print(f'trained for {TOTAL_TIMESTEPS * i} times----------')


def play():
    env = Monitor(Game2048Env(), LOG_DIR)
    model = PPO.load(f'{MODEL_DIR}/{LOAD_MODEL}', env=env)
    for episode in range(1):
        obs, _ = env.reset()
        while 1:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            print("action", action, 'reward', reward)
            env.render()
            sleep(0.35)
            if terminated or truncated:
                sleep(5)
                print('tot score', np.sum(obs))
                break


if __name__ == '__main__':
    play()
