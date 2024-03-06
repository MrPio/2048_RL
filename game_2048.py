import random
from enum import Enum

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
        self.max_moves = 1024 + 1
        self.moves = 0
        self.player = Player.CPU
        self.state = State.PLAY
        self.board = np.zeros((self.size, self.size), dtype=np.int16)
        self.logs = {(2 ** i if i > 0 else 0): i for i in range(20)}
        self.cpu_move()

    def cpu_move(self) -> None:
        """Choose a random cpu move"""
        self.moves += 1
        zero_rows, zero_cols = np.where(self.board == 0)
        if len(zero_rows) == 0:
            self.state = State.LOSE
        else:
            cell = random.randrange(0, len(zero_rows))
            self.board[zero_rows[cell], zero_cols[cell]] = 2

    def user_move(self, move: Move) -> bool:
        """Commits a chosen move on the current game board.
        Note: this checks for win condition
        :param move: The chosen move to commit
        :return: Whether the move resulted in any changes to the board
        """
        changes = False
        new_move = move if move.is_horizontal() else move.transpose()
        if not move.is_horizontal():
            self.board = np.transpose(self.board)

        for i in range(self.size):
            new_row = self._merge_row_left(i, new_move is Move.RIGHT)
            if not np.array_equal(new_row, self.board[i]):
                changes = True
            self.board[i] = new_row
        if not move.is_horizontal():
            self.board = np.transpose(self.board)
        if np.any(self.board == self.target):
            self.state = State.WIN
        return changes

    def _merge_row_left(self, row: int, reverse: bool = False) -> np.array:
        non_zeros = self.board[row][self.board[row] != 0]
        if reverse:
            non_zeros = non_zeros[::-1]
        merged_row = np.zeros(self.size)
        i = 0
        pos = 0
        while i < len(non_zeros):
            if i + 1 < len(non_zeros) and non_zeros[i] == non_zeros[i + 1]:
                merged_row[pos] = 2 * non_zeros[i]
                i += 2
            else:
                merged_row[pos] = non_zeros[i]
                i += 1
            pos += 1
        return merged_row if not reverse else merged_row[::-1]

    def get_reward(self) -> float:
        """Calculates the reward associated to the current board
        Note: should the reward be relative to the game state or to the chosen move?
              The former is here assumed
        :return: The reward associated with the current game board
        """
        # return np.sum(np.array([[1, 2, 3], [4, 5, 6], [9, 8, 7]]) * self.board)
        return np.sum(self.get_observation() ** 1.2)

    def get_observation(self) -> np.array:
        """Calculates the observation associated to the current board
        Note: observation = log2(board)
        :return: The observation associated with the current game board
        """
        return np.vectorize(lambda x: self.logs[x])(self.board).astype(np.uint8).ravel()

    def has_move(self) -> bool:
        return 0 in self.board or \
               (self.board[:-1, :] == self.board[1:, :]).any() or \
               (self.board[:, :-1] == self.board[:, 1:]).any()
