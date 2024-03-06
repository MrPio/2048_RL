import random
from enum import Enum

import numpy
import numpy as np


class Move(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def is_horizontal(self) -> bool:
        return self in [Move.LEFT, Move.RIGHT]

    def transpose(self) -> 'Move':
        return list(Move)[(self.value + 2) % len(Move)]


class Player(Enum):
    CPU = 0
    HUMAN = 1

    def other(self) -> 'Player':
        return Player.CPU if self is Player.HUMAN else Player.HUMAN


class State(Enum):
    PLAY = 0
    WIN = 1
    LOSE = 2

    def is_done(self) -> bool:
        return self is not State.PLAY


class Game2048:
    def __init__(self):
        self.target = 2048
        self.size = 3
        self.max_moves = 999
        self.moves = 0
        self.player = Player.CPU
        self.state = State.PLAY
        self.board = np.zeros((self.size, self.size), dtype=np.int16)
        self.cpu_move()

    def cpu_move(self) -> None:
        self.moves += 1
        zero_rows, zero_cols = np.where(self.board == 0)
        if len(zero_rows) == 0:
            self.state = State.LOSE
        else:
            cell = random.randrange(0, len(zero_rows))
            self.board[zero_rows[cell], zero_cols[cell]] = 2

    def user_move(self, move: Move) -> None:
        new_move = move if move.is_horizontal() else move.transpose()
        if not move.is_horizontal():
            self.board = np.transpose(self.board)
        for i in range(self.size):
            if new_move is Move.LEFT:
                j = 0
                while j < self.size - 1:
                    if self.board[i, 0] == 0 and len(self.board[i][self.board[i] == 0]) < self.size:
                        self.board[i][j:] = np.roll(self.board[i][j:], -1)
                        j -= 1
                    elif self.board[i, j] == self.board[i, j + 1]:
                        self.board[i, j] *= 2
                        self.board[i][j + 1:] = np.roll(self.board[i][j + 1:], -1)
                        self.board[i, self.size - 1] = 0
                    j += 1
            elif new_move is Move.RIGHT:
                j = self.size - 1
                while j > 0:
                    if self.board[i, self.size - 1] == 0 and len(self.board[i][self.board[i] == 0]) < self.size:
                        self.board[i][:j + 1] = np.roll(self.board[i][:j + 1], 1)
                        j += 1
                    elif self.board[i, j] == self.board[i, j - 1]:
                        self.board[i, j] *= 2
                        self.board[i][:j] = np.roll(self.board[i][:j], 1)
                        self.board[i, 0] = 0
                    j -= 1
        if not move.is_horizontal():
            self.board = np.transpose(self.board)
        if np.any(self.board == self.target):
            self.state = State.WIN

    def get_moves(self) -> set[Move]:
        moves = []
        # Detect merge-able tiles
        if (self.board[:-1, :] == self.board[1:, :]).any():
            moves += [Move.UP, Move.DOWN]
        if (self.board[:, :-1] == self.board[:, 1:]).any():
            moves += [Move.LEFT, Move.RIGHT]
        # Detect zeros tiles
        zero_rows, zero_cols = np.where(self.board == 0)
        for zero_row, zero_col in zip(zero_rows, zero_cols):
            moves += [Move.LEFT] * int(zero_col < self.size - 1) + [Move.RIGHT] * int(zero_col > 0)
            moves += [Move.UP] * int(zero_row < self.size - 1) + [Move.DOWN] * int(zero_row > 0)
        return set(moves)
