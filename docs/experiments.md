# Experiment Matrix

These commands document the original training and testing runs. Run them from the project root after `uv sync`.

Model suffixes `Mxxx` record the maximum score reached during training. Training without `--model-name` saves `_aiTestModel{score}.pth`; the checkpoint listed here is the one retained in `models/`. `_aiTestModel063.pth` is the default `test` fallback when `--model` is omitted.

## 1.1 Closed Borders, No Obstacles

```bash
# train
uv run python DQNSnakeMain.py train --games 500 --borders closed --obstacles 0

# test on closed borders with 0, 10, and 20 obstacles
uv run python DQNSnakeMain.py test --games 500 --model closedBorderM064.pth --borders closed --obstacles 0
uv run python DQNSnakeMain.py test --games 500 --model closedBorderM064.pth --borders closed --obstacles 10
uv run python DQNSnakeMain.py test --games 500 --model closedBorderM064.pth --borders closed --obstacles 20
```

## 1.2 Closed Borders, 10 Obstacles

```bash
# train
uv run python DQNSnakeMain.py train --games 500 --borders closed --obstacles 10

# test on closed borders with 0, 10, and 20 obstacles
uv run python DQNSnakeMain.py test --games 500 --model closedBorderM050.pth --borders closed --obstacles 0
uv run python DQNSnakeMain.py test --games 500 --model closedBorderM050.pth --borders closed --obstacles 10
uv run python DQNSnakeMain.py test --games 500 --model closedBorderM050.pth --borders closed --obstacles 20
```

## 1.3 Closed Borders, 20 Obstacles

```bash
# train
uv run python DQNSnakeMain.py train --games 500 --borders closed --obstacles 20

# test on closed borders with 0, 10, and 20 obstacles
uv run python DQNSnakeMain.py test --games 500 --model closedBorderM041.pth --borders closed --obstacles 0
uv run python DQNSnakeMain.py test --games 500 --model closedBorderM041.pth --borders closed --obstacles 10
uv run python DQNSnakeMain.py test --games 500 --model closedBorderM041.pth --borders closed --obstacles 20
```

## 2.1 Hybrid Borders, No Obstacles

```bash
# train
uv run python DQNSnakeMain.py train --games 500 --borders hybrid --obstacles 0

# test on hybrid borders with 0, 10, and 20 obstacles
uv run python DQNSnakeMain.py test --games 500 --model hybridBorderM074.pth --borders hybrid --obstacles 0
uv run python DQNSnakeMain.py test --games 500 --model hybridBorderM074.pth --borders hybrid --obstacles 10
uv run python DQNSnakeMain.py test --games 500 --model hybridBorderM074.pth --borders hybrid --obstacles 20
```

## 2.2 Hybrid Borders, 10 Obstacles

```bash
# train
uv run python DQNSnakeMain.py train --games 500 --borders hybrid --obstacles 10

# test on hybrid borders with 0, 10, and 20 obstacles
uv run python DQNSnakeMain.py test --games 500 --model hybridBorderM048.pth --borders hybrid --obstacles 0
uv run python DQNSnakeMain.py test --games 500 --model hybridBorderM048.pth --borders hybrid --obstacles 10
uv run python DQNSnakeMain.py test --games 500 --model hybridBorderM048.pth --borders hybrid --obstacles 20
```

## 2.3 Hybrid Borders, 20 Obstacles

```bash
# train
uv run python DQNSnakeMain.py train --games 500 --borders hybrid --obstacles 20

# test on hybrid borders with 0, 10, and 20 obstacles
uv run python DQNSnakeMain.py test --games 500 --model hybridBorderM035.pth --borders hybrid --obstacles 0
uv run python DQNSnakeMain.py test --games 500 --model hybridBorderM035.pth --borders hybrid --obstacles 10
uv run python DQNSnakeMain.py test --games 500 --model hybridBorderM035.pth --borders hybrid --obstacles 20
```

## 3 Training and Testing with Different Incentives

The historical run used collision=-10, food=+30, and no progress=-10. The current code uses food=+10, so the training command below does not reproduce `fixLoopingM030.pth` exactly. Test the trained model to see how it performs.

```bash
# historical training run
uv run python DQNSnakeMain.py train --games 500 --borders hybrid --obstacles 20

# test
uv run python DQNSnakeMain.py test --games 500 --model fixLoopingM030.pth --borders hybrid --obstacles 20
```

## 4.1 Closed-Border Model Tested on Hybrid Borders

```bash
uv run python DQNSnakeMain.py test --games 500 --model closedBorderM064.pth --borders hybrid --obstacles 0
uv run python DQNSnakeMain.py test --games 500 --model closedBorderM064.pth --borders hybrid --obstacles 20
```

## 4.2 Hybrid-Border Model Tested on Closed Borders

```bash
uv run python DQNSnakeMain.py test --games 500 --model hybridBorderM074.pth --borders closed --obstacles 0
uv run python DQNSnakeMain.py test --games 500 --model hybridBorderM074.pth --borders closed --obstacles 20
```

## 5 Complex Network on Closed Borders

```bash
# train with the complex architecture
uv run python DQNSnakeMain.py train --games 500 --borders closed --obstacles 0 --architecture complex

# test the complex model on closed borders with 0, 10, and 20 obstacles
uv run python DQNSnakeMain.py test --games 500 --model complexNetworkM059.pth --borders closed --obstacles 0 --architecture complex
uv run python DQNSnakeMain.py test --games 500 --model complexNetworkM059.pth --borders closed --obstacles 10 --architecture complex
uv run python DQNSnakeMain.py test --games 500 --model complexNetworkM059.pth --borders closed --obstacles 20 --architecture complex
```

## Demo Training and Testing

```bash
# train for 100 games with a display speed of 75
uv run python DQNSnakeMain.py train --games 100 --borders closed --speed 75

# test for 100 games with a display speed of 75
uv run python DQNSnakeMain.py test --games 100 --model closedBorderM064.pth --borders closed --speed 75
```
