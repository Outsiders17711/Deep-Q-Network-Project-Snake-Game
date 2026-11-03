## Deep Q-Network Agent for Snake Game

This repository holds code and reports for an applied reinforcement learning research project I carried out in Fall 2021. The objective of the project was the Python implementation of a simple Deep Q-Network (DQN) agent for the classic game of Snake using the ***PyTorch*** machine learning framework and the ***[Pygame]( https://www.pygame.org/wiki/about. )*** library. 

Several DQN agents were trained on different combinations of environment and incentives &ndash; borders, obstacles, rewards, and penalties. Furthermore, agents trained on one combination of environment and incentives were tested on other combinations of environment and incentives to investigate how much the agent's training generalizes when faced with novel situations. 

The agents were trained for 500 games, and each test also lasted for 500 games. The results of the training and testing ablation studies conducted are presented and discussed in the [report](./docs/RL-Project-DeepQ-Network-Agent-Report.pdf) PDF file, and summarized in the [presentation](./docs/RL-Project-DeepQ-Network-Agent-Presentation.pdf) PDF file.

The code implementation was inspired by a similar open-source project [^1] [^2] by ***[Patrick Loeber](https://www.python-engineer.com/about/)***. Do check out his *[YouTube channel](https://www.youtube.com/c/PythonEngineer)* and *[website](https://www.python-engineer.com/)* for more informative Machine Learning, Computer Vision, and Data Science content. 

<!-- -->
[^1]: L. Patrick, *[Teach AI To Play Snake! Reinforcement Learning With PyTorch and Pygame.](https://github.com/python-engineer/snake-ai-pytorch)*, 2021. 
[^2]: L. Patrick, *[Teach AI To Play Snake - Practical Reinforcement Learning With PyTorch And Pygame, Python Engineer](https://python-engineer.com/posts/teach-ai-snake-reinforcement-learning)*. [Accessed: 12-Oct-2021].
<!-- -->

### Environment Setup

The project uses Python 3.13 and [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run python DQNSnakeMain.py
```

`uv sync` creates the `.venv` environment and installs the locked dependencies, including the CPU build of PyTorch. The uncommented call at the bottom of `DQNSnakeMain.py` selects training, testing, or human play. The entry point is `DQNSnakeMain.py`, the game modules live in `src/`, trained models in `models/`, and the demo GIFs, report, and presentation in `docs/`.

### Code Implementation in Python, PyTorch & Pygame

The Python code implementation is split into four modules. The details, summarized below, can be found in the [report](./docs/RL-Project-DeepQ-Network-Agent-Report.pdf) PDF file.

1. **`src/AINetwork.py`**
This helper module contains the PyTorch implementation of the neural network (NN) models and architectures. This module also contains the Q-Learning algorithm for calculating the loss across game states and training the agent.

2. **`src/SnakeEnvironment.py`**
This helper module contains the Pygame implementation of the game environment (borders, obstacles, food and the snake) within the playing area. This module outputs information needed to abstract the game state and inputs information to determine the next move of the agent and updates the user interface.

3. **`src/AIAgent.py`**
This helper module implements the snake game agent by building upon and connecting the `src/AINetwork.py` and `src/SnakeEnvironment.py` helper modules. It abstracts the game state variables from `src/SnakeEnvironment.py` and passes the state variables to `src/AINetwork.py` to determine the agent's next move.

4. **`DQNSnakeMain.py`**
This is the main module for running the Snake Game Deep Q-Network code implementation. It contains three functions, which depend on the `src/SnakeEnvironment.py` and `src/AIAgent.py` helper modules, viz:
    - `AITrainer()`: Trains an agent model for a number of games within a specified environment .
    - `AITester()`: Loads a trained agent model for testing in a specified environment.
    - `userMode()`: Sets up a game with a specified environment with the user's keyboard input determining the snake's movement.

  The `./models` folder contains some trained agent models. The suffixes `Mxxx` represents the maximum score (in one game) achieved by the agent during training across 500 games. `complexNetworkM059.pth` requires the commented complex architecture block in `src/AINetwork.py` to be enabled; the default simple architecture cannot load it.

<hr>

### Training Demo

```python
AITrainer(max_games=500, borders="closed", obstacles=0)
```

<p align="center"><img src="./docs/DeepQ-Network-Training.gif?raw=true" alt="Training Demo" style="width:640px;max-height:286px;"></p>

### Testing Demo

```python
AITester(max_games=500, borders="closed", obstacles=0, model_file="closedBorderM064.pth")
```

<p align="center"><img src="./docs/DeepQ-Network-Testing.gif?raw=true" alt="Testing Demo" style="width:640px;max-height:286px;"></p>
