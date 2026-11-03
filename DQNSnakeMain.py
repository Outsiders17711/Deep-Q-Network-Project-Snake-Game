# [IMPORT MODULES]_____________________________________________________________
import argparse
import math
import statistics as stats
from pathlib import Path

from src.AIAgent import AIAgent
from src.SnakeEnvironment import dqnSnakeGame


# [>>>>>]______________________________________________________________________
def AITrainer(
    max_games=None,
    training_speed=None,
    obstacles=None,
    file_name=None,
    borders="hybrid",
    architecture="simple",
):
    """
    Start a training session at a UI frame refresh rate `training_speed`, maximum number of games `max_games`.
    The game environment is created based on the specified number of `obstacles` and  `borders` (open/hybrid/closed).
    An optional `file_name` overrides the default name of the trained network model when saved to disk.
    The network architecture can be `simple` or `complex`.
    """

    histScores = []
    histMeanScores = []
    histHighScore = 0
    max_games = math.inf if not max_games else max_games
    training_speed = math.inf if not training_speed else training_speed
    agent = AIAgent(architecture=architecture)
    game = dqnSnakeGame(speed=training_speed, mode=0, obstacles=obstacles, border_type=borders)

    while True:
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
            break

    print(f"AI training terminated! Number of Games: {agent.n_games}  |  High Score: {histHighScore}")
    if not file_name:
        file_name = f"_aiTestModel{histHighScore:0=3d}.pth"
    agent.model.save(file_name=file_name)
    agent.trainer._plotHistory(histScores, histMeanScores, lenPause=60)


# [>>>>>]______________________________________________________________________
def AITester(
    model_file=None,
    testing_speed=None,
    obstacles=None,
    max_games=None,
    borders="hybrid",
    architecture="simple",
):
    """
    Start a testing session at a UI frame refresh rate `testing_speed`, maximum number of games `max_games` and agent from trained model `model_file`.
    The game environment is created based on the specified number of `obstacles` and  `borders` (open/hybrid/closed).
    The network architecture can be `simple` or `complex`.
    """

    model_folder_path = Path(__file__).resolve().parent / "models"

    if not model_file:
        models = [f.name for f in model_folder_path.iterdir() if "_aiTestModel" in f.name]
        model_file = max(models)
        print(f"Available Test Models: {models}")

    model_path = model_folder_path / model_file
    print(f"Playing game in AI mode using best trained model <{model_file}>")

    histScores = []
    histMeanScores = []
    agent = AIAgent(architecture=architecture)
    testing_speed = math.inf if not testing_speed else testing_speed
    max_games = math.inf if not max_games else max_games
    game = dqnSnakeGame(speed=testing_speed, mode=0, obstacles=obstacles, border_type=borders)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old, playMode=True, modelPath=model_path)
        _reward, done, score = game.play_step(final_move)
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
        _reward, game_over, score = game.play_step()
        if game_over:
            print("Final Score:", score)
            break


def parseArgs():
    parser = argparse.ArgumentParser(description="Run the Snake game Deep Q-Network agent.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a new agent")
    train.add_argument("--games", type=int)
    train.add_argument("--speed", type=float)
    train.add_argument("--obstacles", type=int)
    train.add_argument("--model-name")
    train.add_argument("--borders", choices=("open", "hybrid", "closed"), default="hybrid")
    train.add_argument("--architecture", choices=("simple", "complex"), default="simple")

    test = commands.add_parser("test", help="test a trained agent")
    test.add_argument("--model")
    test.add_argument("--games", type=int)
    test.add_argument("--speed", type=float)
    test.add_argument("--obstacles", type=int)
    test.add_argument("--borders", choices=("open", "hybrid", "closed"), default="hybrid")
    test.add_argument("--architecture", choices=("simple", "complex"), default="simple")

    play = commands.add_parser("play", help="play the game manually")
    play.add_argument("--speed", type=float, default=10)
    play.add_argument("--xxspeed", type=float, default=1)
    play.add_argument("--obstacles", type=int)
    play.add_argument("--borders", choices=("open", "hybrid", "closed"), default="hybrid")

    return parser.parse_args()


def main():
    args = parseArgs()

    if args.command == "train":
        AITrainer(
            max_games=args.games,
            training_speed=args.speed,
            obstacles=args.obstacles,
            file_name=args.model_name,
            borders=args.borders,
            architecture=args.architecture,
        )
    elif args.command == "test":
        AITester(
            model_file=args.model,
            testing_speed=args.speed,
            obstacles=args.obstacles,
            max_games=args.games,
            borders=args.borders,
            architecture=args.architecture,
        )
    else:
        userMode(
            playing_speed=args.speed,
            xxspeed=args.xxspeed,
            obstacles=args.obstacles,
            borders=args.borders,
        )


if __name__ == "__main__":
    main()

# [_end]____________________________________________________________
