from abc import ABC

import cv2
import gymnasium as gym
import numpy as np

from game_2048 import Game2048, Move


class Game2048Env(gym.Env, ABC):

    def __init__(self):
        self.game = Game2048()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=self.game.target,
                                                shape=(self.game.size ** 2,),
                                                dtype=np.int16)
        self.window_size = 900
        self.cell_size = self.window_size // self.game.size
        self.img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)

    def step(self, action: int):
        moves = self.game.get_moves()
        move: Move = list(Move)[action]
        if move in moves:
            self.game.user_move(move)
            self.game.cpu_move()
        observation = self.game.board.ravel()
        reward = np.sum(self.game.board) * (1 if move in moves else -1)  # np.count_nonzero(self.game.board == 0)
        terminated = self.game.state.is_done() or len(moves) == 0
        truncated = self.game.moves > self.game.max_moves
        return observation, reward, terminated, truncated, {}

    def reset(self, *, seed=None, _=None):
        super().reset(seed=seed)
        self.game = Game2048()
        return self.game.board.ravel(), {}

    def render(self):
        cv2.imshow('a', self.img)
        cv2.waitKey(1)
        self.img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)

        # Draw the grid
        for i in range(self.game.size + 1):
            cv2.line(self.img, (i * self.cell_size, 0), (i * self.cell_size, self.game.size * self.cell_size),
                     (255, 255, 255), 4)
            cv2.line(self.img, (0, i * self.cell_size), (self.game.size * self.cell_size, i * self.cell_size),
                     (255, 255, 255), 4)

            # Add numbers to the cells
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 4

        for i in range(self.game.size):
            for j in range(self.game.size):
                value = self.game.board[i][j]
                if value != 0:
                    text_size = cv2.getTextSize(str(value), font, font_scale, font_thickness)[0]
                    text_x = (j * self.cell_size) + (self.cell_size - text_size[0]) // 2
                    text_y = (i * self.cell_size) + (self.cell_size + text_size[1]) // 2
                    cv2.putText(self.img, str(value), (text_x, text_y), font, font_scale, (255, 255, 255),
                                font_thickness, cv2.LINE_AA)
