# [IMPORT MODULES]_____________________________________________________________
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# [DEFINE NEURAL NETWORK ARCHITECTURE]_________________________________________
class Linear_QNET(nn.Module):
    """
    Defines a neural network architecture based on the self-explanatory `input_size`, `hidden_size`, `output_size` parameters.
    Two architectures can be created from the same set of parameters by commenting/uncommenting the relevant blocks of code:
    --- simple: [input_size x hidden_size x output_size]
    --- complex: [input_size x hidden_size x 2*hidden_size x output_size]
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        #
        # simple model [comment/uncomment block]
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # simple model [comment/uncomment block]
        #
        # complex model [comment/uncomment block]
        # self.hidden1 = nn.Linear(input_size, hidden_size)
        # self.hidden2 = nn.Linear(hidden_size, hidden_size * 2)
        # self.outlayer = nn.Linear(hidden_size * 2, output_size)
        # complex model [comment/uncomment block]
        #

    def forward(self, x):
        #
        # simple model [comment/uncomment block]
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        # simple model [comment/uncomment block]
        #
        # complex model [comment/uncomment block]
        # x = F.relu(self.hidden1(x))
        # x = F.relu(self.hidden2(x))
        # x = self.outlayer(x)
        # complex model [comment/uncomment block]
        #
        return x

    def save(self, file_name="_aiTrainModel.pth"):
        """
        Saves PyTorch model dictionary to disk. Save folder is the containing folder of this module. An optional `file_name` can be provided.
        """

        model_folder_path = os.path.dirname(os.path.abspath(__file__))
        model_folder_path = os.path.join(model_folder_path, "models")
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


# [INIT Q-NETWORK TRAINER]____________________________________________________________
class QTRainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Trains the neural network based on the Q-learning algorithm. It can take a single set of state variables or a batch of state varisbles.
        """
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            # reshape to (1, x); if len(state.shape) > 1, the shape is already (n, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)  # convert to tuple with only one value

        # 1. predicted Q values with current state
        pred = self.model(state)

        # 2. Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        # 3. apply loss function
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    @staticmethod
    def _plotHistory(scores, mean_scores, lenPause=0.5):
        """
        Plots the historical individual and mean scores for a number of games.
        """

        try:
            plt.xlabel("Number of Games")
            plt.ylabel("Score")
            plt.plot(scores)
            plt.plot(mean_scores)
            plt.ylim(ymin=0)
            plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
            plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

            plt.show(block=False)
            plt.pause(lenPause)

        except:
            pass


# [_end]____________________________________________________________
