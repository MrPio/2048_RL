import os
from time import sleep

import cv2
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor

from game_2048 import Move
from game_2048_env import Game2048Env

TOTAL_TIMESTEPS = 100_000
MODEL_DIR = 'PPO_6'
LOG_DIR = 'logs'
LOAD_MODEL = ''
NEXT_INDEX = 0
for path in [MODEL_DIR, LOG_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)


def train():
    env = Monitor(Game2048Env(), LOG_DIR)
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR) if LOAD_MODEL == '' else \
        PPO.load(f'{MODEL_DIR}/{LOAD_MODEL}', env=env)
    for i in range(NEXT_INDEX, NEXT_INDEX + 999):
        model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL_DIR)
        model.save(f'{MODEL_DIR}/{TOTAL_TIMESTEPS * i}')
        print(f'trained for {TOTAL_TIMESTEPS * i} times----------')


def play(human=False):
    env = Monitor(Game2048Env(), LOG_DIR)
    model = PPO.load(f'{MODEL_DIR}/{LOAD_MODEL}', env=env) if not human else False
    moves = {'w': Move.UP.value, 'a': Move.LEFT.value, 's': Move.DOWN.value, 'd': Move.RIGHT.value}
    for episode in range(5):
        obs, _ = env.reset()
        env.render()
        while 1:
            if human:
                key = cv2.waitKey(0)
                if chr(key) in moves.keys():
                    action = moves[chr(key)]
                else:
                    continue
            else:
                action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            print("action", action, 'reward', reward)
            env.render()
            sleep(0.15)
            if terminated or truncated:
                sleep(4)
                print('tot score', np.sum(obs))
                break


if __name__ == '__main__':
    # train()
    play(human=True)
