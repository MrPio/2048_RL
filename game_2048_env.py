from abc import ABC

import cv2
import gymnasium as gym
import numpy as np

from constants import BACKGROUND_COLORS, CELL_COLORS, FONT_SIZES
from game_2048 import Game2048, Move


class Game2048Env(gym.Env, ABC):

    def __init__(self):
        self.game = Game2048()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=max(self.game.logs.values()),
                                                shape=(self.game.size ** 2,),
                                                dtype=np.uint8)
        self.window_size = 900
        self.cell_size = self.window_size // self.game.size
        self.img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)

    def step(self, action: int):
        move: Move = list(Move)[action]
        self.game.user_move(move)
        if self.game.last_move_valid:
            self.game.cpu_move()
        observation = self.game.get_observation()
        reward = self.game.get_reward()
        terminated = self.game.state.is_done() or not self.game.has_move()
        truncated = self.game.moves > self.game.max_moves
        return observation, reward, terminated, truncated, {}

    def reset(self, *, seed=None, _=None):
        super().reset(seed=seed)
        self.game = Game2048()
        return self.game.get_observation(), {}

    def render(self):
        self.img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        l, n = self.cell_size, self.game.size
        font, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 8

        # Draw the cell backgrounds
        for i in range(n):
            for j in range(n):
                tile_index = self.game.board[i, j]
                bg_col, fg_col = BACKGROUND_COLORS[tile_index], CELL_COLORS[tile_index]
                font_scale = FONT_SIZES[tile_index]
                cv2.rectangle(self.img, (j * l, i * l), ((j + 1) * l, (i + 1) * l),
                              tuple(int(bg_col[i:i + 2], 16) for i in (1, 3, 5)), -1)
                # Draw the text numbers
                if tile_index != 0:
                    text_size = cv2.getTextSize(str(tile_index), font, font_scale, font_thickness)[0]
                    text_x = j * l + (l - text_size[0]) // 2
                    text_y = i * l + (l + text_size[1]) // 2
                    cv2.putText(self.img, str(tile_index), (text_x, text_y), font, font_scale,
                                tuple(int(fg_col[i:i + 2], 16) for i in (1, 3, 5)),
                                font_thickness, cv2.LINE_AA)

        # Draw the grid
        for i in range(self.game.size + 1):
            cv2.line(self.img, (i * l, 0), (i * l, n * l), (230, 230, 230), 4)
            cv2.line(self.img, (0, i * l), (n * l, i * l), (230, 230, 230), 4)

        cv2.imshow('2048', self.img)
