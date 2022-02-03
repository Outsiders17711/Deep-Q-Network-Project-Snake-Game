<p align="right">
<img src="https://badges.pufler.dev/visits/Outsiders17711/Deep-Q-Network-Project-Snake-Game?style=for-the-badge&logo=github" alt="https://github.com/Outsiders17711" />&nbsp;
<img src="https://badges.pufler.dev/updated/Outsiders17711/Deep-Q-Network-Project-Snake-Game?style=for-the-badge&logo=github" alt="https://github.com/Outsiders17711" />&nbsp;
<img src="https://badges.pufler.dev/created/Outsiders17711/Deep-Q-Network-Project-Snake-Game?style=for-the-badge&logo=github" alt="https://github.com/Outsiders17711" />&nbsp;
</p>

## Deep Q-Network Agent For Snake Game

This repository holds code and reports for an applied reinforcement learning research project I carried out in Fall 2021. The objective of the project was the Python implementation of a simple Deep Q-Network (DQN) agent for the classic game of Snake using the ***PyTorch*** machine learning framework and the ***[Pygame]( https://www.pygame.org/wiki/about. )*** library. 

Several DQN agents were trained on different combinations of environment and incentives &ndash; borders, obstacles, rewards, and penalties. Furthermore, agents trained on one combination of environment and incentives were tested on other combinations of environment and incentives to investigate how much the agent's training generalizes when faced with novel situations. 

The agents were trained for 500 games, and each test also lasted for 500 games. The results of the training and testing ablation studies conducted are presented and discussed in the [RL Project_ Deep Q-Network Agent Report](./RL%20Project_%20Deep%20Q-Network%20Agent%20Report.pdf) PDF file, and summarized in the [RL Project_ Deep Q-Network Agent Presentation](./RL%20Project_%20Deep%20Q-Network%20Agent%20Presentation.pdf) PDF file.

The code implementation was inspired by a similar open-source project [^1] [^2] by ***[Patrick Loeber](https://www.python-engineer.com/about/)***. Do check out his *[YouTube channel](https://www.youtube.com/c/PythonEngineer)* and *[website](https://www.python-engineer.com/)* for more informative Machine Learning, Computer Vision, and Data Science content. 

<!-- add link to future bog post(s) -->

<!-- -->
[^1]: L. Patrick, *[Teach AI To Play Snake! Reinforcement Learning With PyTorch and Pygame.](https://github.com/python-engineer/snake-ai-pytorch)*, 2021. 
[^2]: L. Patrick, *[Teach AI To Play Snake - Practical Reinforcement Learning With PyTorch And Pygame, Python Engineer](https://python-engineer.com/posts/teach-ai-snake-reinforcement-learning)*. [Accessed: 12-Oct-2021].
<!-- -->

### Code Implementation in Python, PyTorch & Pygame

The Python code implementation is split into four modules. The details, summarized below, can be found in the [RL Project_ Deep Q-Network Agent Report](./RL%20Project_%20Deep%20Q-Network%20Agent%20Report.pdf) PDF file. 

1. **`_dqnAINetwork.py`**: <br>
This helper module contains the PyTorch implementation of the neural network (NN) models and architectures. This module also contains the Q-Learning algorithm for calculating the loss across game states and training the agent.

2. **`_dqnSnakeEnvironment.py`**: <br>
This helper module contains the Pygame implementation of the game environment (borders, obstacles, food and the snake) within the playing area. This module outputs information needed to abstract the game state and inputs information to determine the next move of the agent and updates the user interface.

3. **`_dqnAIAgent.py`**: <br>
This helper module implements the snake game agent by building upon and connecting the `_dqnAINetwork.py` and `_dqnSnakeEnvironment.py` helper modules. It abstracts the game state variables from `_dqnSnakeEnvironment.py` and passes the state variables to `_dqnAINetwork.py` to determine the agent's next move.

4. **`dqnSnakeMain.py`**: <br>
This is the main module for running the Snake Game Deep Q-Network code implementation. It contains three functions – which depend on the `_dqnSnakeEnvironment.py` and `_dqnAIAgent.py` helper modules – viz:

  - *`AITrainer()`*: Trains an agent model for a number of games within a specified environment .
  - *`AITester()`*: Loads a trained agent model for testing in a specified environment.
  - *`userMode()`*: Sets up a game with a specified environment with the user's keyboard input determining the snake's movement.

  The `./models` folder contains some trained agent models. The suffixes `Mxxx` represents the maximum score (in one game) achieved by the agent during training across 500 games.

<hr>

### Training Demo

```python
AITrainer(max_games=500, borders="closed", obstacles=0, file_name="demo_border=closed_obstacles=0.pth")
```

<!-- <p align="center"><img src="https://github.com/Outsiders17711/Deep-Q-Network-Project-Snake-Game/blob/main/demo/Deep_Q-Network_Training_Demo_2_Trim.gif?raw=true" alt="Training Demo" style="width:640px;max-height:286px;"></p> -->
<p align="center"><img src="./demo/Deep_Q-Network_Training_Demo_2_Trim.gif?raw=true" alt="Training Demo" style="width:640px;max-height:286px;"></p>

### Testing Demo

```python
AITester(max_games=500, borders="closed", obstacles=0, model_file="demo_border=closed_obstacles=0.pth")
```

<!-- <p align="center"><img src="https://github.com/Outsiders17711/Deep-Q-Network-Project-Snake-Game/blob/main/demo/Deep_Q-Network_Testing_Demo_2_Trim.gif?raw=true" alt="Testing Demo" style="width:640px;max-height:286px;"></p> -->
<p align="center"><img src="./demo/Deep_Q-Network_Testing_Demo_2_Trim.gif?raw=true" alt="Testing Demo" style="width:640px;max-height:286px;"></p>
