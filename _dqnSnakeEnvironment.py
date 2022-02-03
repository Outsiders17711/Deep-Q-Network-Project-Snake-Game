# [IMPORT MODULES]_____________________________________________________________
import os
import sys
import random
from collections import namedtuple
from enum import Enum


os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame as pyg
import numpy as np


# [UI PARAMETERS]___________________________________________________________
WIDTH, HEIGHT = 720, 320
BLOCK = 20
PADDING = 2
BORDER_SPACING = {"closed": 1, "hybrid": 4, "open": None}
INNER_HEAD = BLOCK - PADDING * 2
INNER_BODY = BLOCK - PADDING * 4
FONT_SIZE = 20

# [UI COLORS]________________________________________________________________
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BORDER_COLOR = (235, 26, 26)
OUTER_COLORS = np.random.randint(0, 100, (100, 3)).tolist()
INNER_COLORS = np.random.randint(150, 256, (100, 3)).tolist()

# [GAME COORDINATES REPRESENTATION]__________________________________________
Point = namedtuple("Point", "x, y")


# [GAME DIRECTION REPRESENTATION]____________________________________________
class Direction(Enum):
    RIGHT, LEFT, UP, DOWN = 1, 2, 3, 4


# [INIT GAME INSTANCE AND ENVIRONMENT AND FRAMES]____________________________
class dqnSnakeGame:
    """
    Creates a game environment based on the specified number of `obstacles` and  `border_type` (open/hybrid/closed).
    Using the `mode` parameter, the game created can be toggled between AI (mode=0) and Human (mode=1).
    The `xxspeed` parameter controls how much to increase the game speed when the food is eaten (for Human mode only).
    """

    def __init__(self, speed=10, xxspeed=None, mode=1, obstacles=None, border_type="hybrid"):
        # init ui
        pyg.init()
        self.speed = speed
        self.xxspeed = xxspeed
        self.paused_speed = None
        self.human_mode = bool(mode)
        self.obstacles = obstacles
        self.border_type = border_type

        # init display
        self.display = pyg.display.set_mode((WIDTH, HEIGHT))
        pyg.display.set_caption("Deep Q-Network Agent for Snake Game")
        self.clock = pyg.time.Clock()
        self.font = pyg.font.SysFont("3ds", FONT_SIZE, bold=False)
        self.reset()

    def reset(self):
        """
        Initialize tne game state.
        -- 1. Create a snake with the head (placed in the middle of the playing area) and two body segments.
        -- 2. Create borders and obstacles based on specfications.
        """

        self.direction = Direction.RIGHT
        self.head = Point(WIDTH // 2, HEIGHT // 2)
        self.snake = [
            self.head,
            Point(self.head.x - (BLOCK), self.head.y),
            Point(self.head.x - (2 * BLOCK), self.head.y),
        ]
        self.score = 0
        self.nframes = 0
        self.borders = []
        self._draw_borders()
        self._place_food()

    def _draw_borders(self):
        """Create borders and obstacles based on specfications."""

        if BORDER_SPACING[self.border_type]:
            b_vertical = range(0, HEIGHT, BLOCK * BORDER_SPACING[self.border_type])
            for pt in b_vertical:
                self.borders.append(Point(0, pt))
                self.borders.append(Point(0, pt + BLOCK))
                self.borders.append(Point(WIDTH - BLOCK, pt))
                self.borders.append(Point(WIDTH - BLOCK, pt + BLOCK))

            b_horizontal = range(0, WIDTH, BLOCK * BORDER_SPACING[self.border_type])
            for pt in b_horizontal:
                self.borders.append(Point(pt, 0))
                self.borders.append(Point(pt + BLOCK, 0))
                self.borders.append(Point(pt, HEIGHT - BLOCK))
                self.borders.append(Point(pt + BLOCK, HEIGHT - BLOCK))

        if self.obstacles:
            for i in range(self.obstacles):
                self.borders.append(self._create_obstacles())

    def _create_obstacles(self):
        """Create a single obstacles, checking that there is no overlap with existing borders/obstacles and the snake."""

        x = random.randint(0, (WIDTH - 3 * BLOCK) // BLOCK) * BLOCK
        y = random.randint(0, (HEIGHT - 3 * BLOCK) // BLOCK) * BLOCK
        obstacle = Point(x, y)

        if obstacle in self.borders or obstacle in self.snake:
            obstacle = self._create_obstacles()

        return obstacle

    def _place_food(self):
        """Create the off, checking that there is no overlap with existing borders/obstacles and the snake."""

        x = random.randint(0, (WIDTH - BLOCK) // BLOCK) * BLOCK
        y = random.randint(0, (HEIGHT - BLOCK) // BLOCK) * BLOCK
        self.food = Point(x, y)

        if self.food in self.snake or self.food in self.borders:
            self._place_food()

    def play_step(self, action=None):
        """Create a single frame of the game."""

        self.nframes += 1

        # 1. collect user/ai input
        for event in pyg.event.get():
            if event.type == pyg.QUIT:
                self._quit_game()

            if event.type == pyg.KEYDOWN:
                if event.key == pyg.K_SPACE:
                    if not self.paused_speed:
                        self.paused_speed = self.speed
                        self.speed = 1
                    else:
                        self.speed = self.paused_speed
                        self.paused_speed = None
                elif event.key == pyg.K_ESCAPE:
                    self._quit_game()

            if event.type == pyg.KEYDOWN and self.human_mode:
                if event.key == pyg.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pyg.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pyg.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pyg.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN

        # 2. move snake
        self._move(self.direction) if self.human_mode else self._move(action)
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False

        if self._is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.nframes > 50 * len(self.snake) and not self.human_mode:
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            if self.xxspeed and self.human_mode:
                self.speed += self.xxspeed
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update pygame ui and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # 6. return game over and score
        return reward, game_over, self.score

    def _quit_game(self):
        """Quit the program gracefully."""

        print("Quitting game...")
        pyg.display.quit()
        pyg.quit()
        sys.exit()

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # snake hits the borders
        if pt in self.borders:
            return True

        # snake hits its body
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """Draw the UI i.e., the environment components and the snake."""

        # render the display
        self.display.fill(WHITE)

        # draw borders
        for pt in self.borders:
            pyg.draw.rect(
                self.display,
                BLACK,
                pyg.Rect(pt.x, pt.y, BLOCK, BLOCK),
                border_radius=3,
            )

        # render score
        if self.human_mode:
            text = self.font.render("Score: " + str(self.score), True, [0, 168, 243])
            self.display.blit(text, [2, 2])

        # draw head
        pyg.draw.rect(
            self.display,
            OUTER_COLORS[0],
            pyg.Rect(
                self.snake[0].x - PADDING,
                self.snake[0].y - PADDING,
                BLOCK + PADDING * 2,
                BLOCK + PADDING * 2,
            ),
            border_radius=7,
        )
        pyg.draw.rect(
            self.display,
            OUTER_COLORS[0],
            pyg.Rect(
                self.snake[0].x + PADDING,
                self.snake[0].y + PADDING,
                INNER_HEAD,
                INNER_HEAD,
            ),
            border_radius=7,
        )

        # draw body
        for idx, pt in enumerate(self.snake[1:-1]):
            pyg.draw.rect(
                self.display,
                OUTER_COLORS[idx],
                pyg.Rect(pt.x, pt.y, BLOCK, BLOCK),
                border_radius=3,
            )
            pyg.draw.rect(
                self.display,
                INNER_COLORS[idx],
                pyg.Rect(pt.x + PADDING * 2, pt.y + PADDING * 2, INNER_BODY, INNER_BODY),
            )

        # draw tail
        pyg.draw.rect(
            self.display,
            OUTER_COLORS[-1],
            pyg.Rect(
                self.snake[-1].x + PADDING,
                self.snake[-1].y + PADDING,
                INNER_HEAD,
                INNER_HEAD,
            ),
        )

        # draw food
        pyg.draw.rect(
            self.display,
            BLACK,
            pyg.Rect(self.food.x, self.food.y, BLOCK, BLOCK),
            border_radius=1,
        )
        pyg.draw.rect(
            self.display,
            RED,
            pyg.Rect(self.food.x + PADDING, self.food.y + PADDING, INNER_HEAD, INNER_HEAD),
            border_radius=3,
        )

        # update ui
        pyg.display.flip()

    def _move(self, action=None):
        """Update the position of the snake taking direction and porous borders into account."""

        if not self.human_mode:
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)

            if np.array_equal(action, [1, 0, 0]):
                new_direction = clock_wise[idx]  # no change
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4  # looping continuously through the list
                new_direction = clock_wise[next_idx]  # right turn r -> d -> l -> u
            else:  # [0,0,1]
                next_idx = (idx - 1) % 4  # looping continuously through the list
                new_direction = clock_wise[next_idx]  # right turn r -> u -> l -> d

            self.direction = new_direction

        x = self.head.x
        y = self.head.y

        # outside playing area; porous border; come out from opposite side
        if x > WIDTH - BLOCK or x < 0 or y > HEIGHT - BLOCK or y < 0:
            if x > WIDTH - BLOCK:
                x -= WIDTH
            elif x < 0:
                x += WIDTH
            elif y > HEIGHT - BLOCK:
                y -= HEIGHT
            elif y < 0:
                y += HEIGHT

        # within playing area
        else:
            if self.direction == Direction.RIGHT:
                x += BLOCK
            elif self.direction == Direction.LEFT:
                x -= BLOCK
            elif self.direction == Direction.DOWN:
                y += BLOCK
            elif self.direction == Direction.UP:
                y -= BLOCK

        self.head = Point(x, y)


# [_end]____________________________________________________________
