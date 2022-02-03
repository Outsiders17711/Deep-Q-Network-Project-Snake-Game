# [IMPORT MODULES]_____________________________________________________________
import random
from collections import deque

import torch
import numpy as np

from _dqnSnakeEnvironment import Direction, Point
from _dqnAINetwork import Linear_QNET, QTRainer


# [AGENT MEMORY AND TRAINING PARAMETERS]________________________________________
MAX_MEMORY = 100000
BATCH_SIZE = 100
LR = 0.001


# [INIT AI AGENT]_______________________________________________________________
class AIAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNET(11, 256, 3)
        self.trainer = QTRainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """Abstract the 11 game state variables from the game environment module."""

        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game._is_collision(point_r))
            or (dir_l and game._is_collision(point_l))
            or (dir_u and game._is_collision(point_u))
            or (dir_d and game._is_collision(point_d)),
            # Danger right
            (dir_u and game._is_collision(point_r))
            or (dir_d and game._is_collision(point_l))
            or (dir_l and game._is_collision(point_u))
            or (dir_r and game._is_collision(point_d)),
            # Danger left
            (dir_d and game._is_collision(point_r))
            or (dir_u and game._is_collision(point_l))
            or (dir_r and game._is_collision(point_u))
            or (dir_l and game._is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Add observed game information to memory for exprienced replay training at the end of each game."""

        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Carry out exprienced replay training of the agent at the end of each game."""

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Carry per frame training of the agent."""

        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, playMode=False, modelPath=None):
        """
        Get the agent's next action from the network model.
        At the beginning training, some actions are probabilistically chosen randomly to carry out exploration of the playing area.
        During testing, all actions are inferred from the trained model.
        """

        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if playMode:
            self.model.load_state_dict(torch.load(modelPath))

        if random.randint(0, 200) < self.epsilon and not playMode:
            move = random.randint(0, 2)
            final_move[move] = 1

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


# [_end]____________________________________________________________
