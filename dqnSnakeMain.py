# [IMPORT MODULES]_____________________________________________________________
import os
import math
import statistics as stats

from _dqnSnakeEnvironment import dqnSnakeGame
from _dqnAIAgent import AIAgent


# [>>>>>]______________________________________________________________________
def AITrainer(
    max_games=None, training_speed=None, obstacles=None, file_name=None, borders="hybrid"
):
    """
    Start a training session at a UI frame refresh rate `training_speed`, maximum number of games `max_games`.
    The game environment is created based on the specified number of `obstacles` and  `borders` (open/hybrid/closed).
    An optional `file_name` overrides the default name of the trained network model when saved to disk.
    """

    histScores = []
    histMeanScores = []
    histHighScore = 0
    max_games = math.inf if not max_games else max_games
    training_speed = math.inf if not training_speed else training_speed
    agent = AIAgent()
    game = dqnSnakeGame(speed=training_speed, mode=0, obstacles=obstacles, border_type=borders)

    while True:
        try:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > histHighScore:
                    histHighScore = score
                    agent.model.save()

                histScores.append(score)
                histMeanScores.append(round(stats.mean(histScores), ndigits=2))

            if agent.n_games == max_games:
                print("Maximum number of games reached!")
                raise Exception

        except Exception:
            print(
                f"AI training terminated! ",
                f"Number of Games: {agent.n_games}  |  High Score: {histHighScore}",
            )
            if not file_name:
                file_name = file_name = f"_aiTestModel{histHighScore:0=3d}.pth"
            agent.model.save(file_name=file_name)
            agent.trainer._plotHistory(histScores, histMeanScores, lenPause=60)
            break


# [>>>>>]______________________________________________________________________
def AITester(model_file=None, testing_speed=None, obstacles=None, max_games=None, borders="hybrid"):
    """
    Start a testing session at a UI frame refresh rate `testing_speed`, maximum number of games `max_games` and agent from trained model `model_file`.
    The game environment is created based on the specified number of `obstacles` and  `borders` (open/hybrid/closed).
    """

    model_folder_path = os.path.dirname(os.path.abspath(__file__))
    model_folder_path = os.path.join(model_folder_path, "models")

    if not model_file:
        models = [f for f in os.listdir(model_folder_path) if "_aiTestModel" in f]
        model_file = sorted(models, reverse=True)[0]
        print(f"Available Test Models: {models}")

    model_path = os.path.join(model_folder_path, model_file)
    print(f"Playing game in AI mode using best trained model <{model_file}>")

    histScores = []
    histMeanScores = []
    agent = AIAgent()
    testing_speed = math.inf if not testing_speed else testing_speed
    max_games = math.inf if not max_games else max_games
    game = dqnSnakeGame(speed=testing_speed, mode=0, obstacles=obstacles, border_type=borders)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old, playMode=True, modelPath=model_path)
        reward, done, score = game.play_step(final_move)
        agent.get_state(game)

        if done:
            histScores.append(score)
            histMeanScores.append(round(stats.mean(histScores), ndigits=2))
            print(
                f"- game {len(histScores)}\t- score {score}\t- high score {max(histScores)}\t  - mean score {stats.mean(histScores):.2f}"
            )
            game.reset()
            agent.n_games += 1

        if agent.n_games == max_games:
            print("Maximum number of games reached!")
            agent.trainer._plotHistory(histScores, histMeanScores, lenPause=60)
            break


# [>>>>>]______________________________________________________________________
def userMode(playing_speed=10, xxspeed=1, mode=1, obstacles=None, borders="open"):
    """
    Start a human playing session at a UI frame refresh rate `playing_speed`, maximum number of games `max_games` and game speed increment/food of `xxspeed`.
    The game environment is created based on the specified number of `obstacles` and  `borders` (open/hybrid/closed).
    """

    game = dqnSnakeGame(
        speed=playing_speed, xxspeed=xxspeed, mode=mode, obstacles=obstacles, border_type=borders
    )

    while True:
        reward, game_over, score = game.play_step()
        if game_over:
            print("Final Score:", score)
            break


if __name__ == "__main__":
    """
    Carry out agent training from scratch or testing with previously trained agents by uncommenting the relevant line of code.
    Each section and line of code is self-explanatory, with comments to aid comprehension.
    """

    # ---
    # 1.1 - CLOSED BORDERS & 0 OBSTACLES
    # AITrainer(max_games=500, obstacles=0, borders="closed")
    # AITester(max_games=500, model_file="closedBorderM064.pth", borders="closed", obstacles=0)
    # AITester(max_games=500, model_file="closedBorderM064.pth", borders="closed", obstacles=10)
    # AITester(max_games=500, model_file="closedBorderM064.pth", borders="closed", obstacles=20)
    # ---
    # 1.2 - CLOSED BORDERS & 10 OBSTACLES
    # AITrainer(max_games=500, obstacles=10, borders="closed")
    # AITester(max_games=500, model_file="closedBorderM050.pth", borders="closed", obstacles=0)
    # AITester(max_games=500, model_file="closedBorderM050.pth", borders="closed", obstacles=10)
    # AITester(max_games=500, model_file="closedBorderM050.pth", borders="closed", obstacles=20)
    # 1.3 - CLOSED BORDERS & 20 OBSTACLES
    # AITrainer(max_games=500, obstacles=20, borders="closed")
    # AITester(max_games=500, model_file="closedBorderM041.pth", borders="closed", obstacles=0)
    # AITester(max_games=500, model_file="closedBorderM041.pth", borders="closed", obstacles=10)
    # AITester(max_games=500, model_file="closedBorderM041.pth", borders="closed", obstacles=20)
    # ---
    #
    # ---
    # 2.1 - HYBRID BORDERS & 0 OBSTACLES
    # AITrainer(max_games=500, obstacles=0, borders="hybrid")
    # AITester(max_games=500, model_file="hybridBorderM074.pth", borders="hybrid", obstacles=0)
    # AITester(max_games=500, model_file="hybridBorderM074.pth", borders="hybrid", obstacles=10)
    # AITester(max_games=500, model_file="hybridBorderM074.pth", borders="hybrid", obstacles=20)
    # ---
    # 2.2 - HYBRID BORDERS & 0 OBSTACLES
    # AITrainer(max_games=500, obstacles=10, borders="hybrid")
    # AITester(max_games=500, model_file="hybridBorderM048.pth", borders="hybrid", obstacles=0)
    # AITester(max_games=500, model_file="hybridBorderM048.pth", borders="hybrid", obstacles=10)
    # AITester(max_games=500, model_file="hybridBorderM048.pth", borders="hybrid", obstacles=20)
    # ---
    # 2.3 - HYBRID BORDERS & 0 OBSTACLES
    # AITrainer(max_games=500, obstacles=20, borders="hybrid")
    # AITester(max_games=500, model_file="hybridBorderM035.pth", borders="hybrid", obstacles=0)
    # AITester(max_games=500, model_file="hybridBorderM035.pth", borders="hybrid", obstacles=10)
    # AITester(max_games=500, model_file="hybridBorderM035.pth", borders="hybrid", obstacles=20)
    # ---
    #
    # ---
    # 3 - TRAINING AND TESTING WITH DIFFERENT INCENTIVES [collision=-10, food=+30, no progress=-10]
    # AITrainer(max_games=500, obstacles=20, borders="hybrid")
    # AITester(max_games=500, model_file="fixLoopingM030.pth", borders="hybrid", obstacles=20)
    # ---
    #
    # ---
    # 4.1. TESTING CLOSED BORDER - NO OBSTACLES MODEL ON HYBRID BORDER
    # AITester(max_games=500, model_file="closedBorderM064.pth", borders="hybrid", obstacles=0)
    # AITester(max_games=500, model_file="closedBorderM064.pth", borders="hybrid", obstacles=20)
    # ---
    # 4.2. TESTING HYBRID BORDER - NO OBSTACLES MODEL ON CLOSED BORDER
    # AITester(max_games=500, model_file="hybridBorderM074.pth", borders="closed", obstacles=0)
    # AITester(max_games=500, model_file="hybridBorderM074.pth", borders="closed", obstacles=20)
    # ---
    #
    # ---
    # 5. CLOSED BORDERS - COMPLEX NN
    # AITrainer(max_games=500, obstacles=0, borders="closed")
    # AITester(max_games=500, model_file="complexNetworkM059.pth", borders="closed", obstacles=0)
    # AITester(max_games=500, model_file="complexNetworkM059.pth", borders="closed", obstacles=10)
    # AITester(max_games=500, model_file="complexNetworkM059.pth", borders="closed", obstacles=20)
    # ---
    #
    # --- DEMO TRAINING & TESTING
    # AITrainer(max_games=100, obstacles=None, borders="closed", training_speed=75)
    # AITester(max_games=100, model_file="closedBorderM064.pth", borders="closed", testing_speed=75)
    # --- DEMO TRAINING & TESTING
    #
    # ---
    userMode(playing_speed=10, xxspeed=1, obstacles=None, borders="hybrid")
    # ---
    #
    pass

# [_end]____________________________________________________________
