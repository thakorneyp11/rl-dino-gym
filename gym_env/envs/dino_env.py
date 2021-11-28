import os
import cv2
import numpy as np
import pandas as pd
from gym import Env
from gym import spaces
from gym.utils import seeding

from gym_env.spaces.person import Person
from gym_env.spaces.items import RockItem, BirdItem, EnergyItem


class DinoEnv(Env):
    """A Dino Chrome Game environment for OpenAI gym"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, image_obs=False, verbose=None):
        super(DinoEnv, self).__init__()

        # TODO: create option to use an element-coordinate observation space
        self.image_obs = image_obs

        self.observation_shape = (350, 900, 3)
        self.observation_space = spaces.Box(low=np.full(self.observation_shape, 0),
                                            high=np.full(self.observation_shape, 255),
                                            dtype=np.uint8)
        self.action_space = spaces.Discrete(3, )

        self.canvas = np.ones(self.observation_shape)
        self.elements = []
        self.x_min = 0
        self.x_max = 900
        self.y_min = 75
        self.y_max = 300

        self.ep_reward = 0
        self.step_count = 0
        self.max_reward = -1000

        self.current_energy = 1000
        self.max_energy = 1000

        self.episode_idx = -1
        self.verbose = verbose

        self.seed()
        self.dino_setup()
        self.canvas_position_setup()

    def dino_setup(self):
        self.dino_base_x = 150
        self.dino_base_y = 300
        self.dino = Person("dino person", self.dino_base_x, self.dino_base_x, 50, 300)
        self.dino.set_position(self.dino_base_x, self.dino_base_y)

    def canvas_position_setup(self):
        self.y_row_3 = 200
        self.y_row_2 = 250
        self.y_row_1 = 300

    def seed(self, seed=None):
        """Set seed to reproduce the training"""
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self):
        self.episode_idx += 1
        self.ep_reward = 0
        self.step_count = 0

        self.current_energy = self.max_energy
        self.bird_count = 0
        self.rock_count = 0
        self.energy_count = 0

        self.dino_setup()

        self.elements = [self.dino]

        self.canvas = np.ones(self.observation_shape)
        self.draw_on_canvas()

        return self.canvas

    def step(self, action):
        self.step_count += 1
        self.current_energy -= 1
        done = False
        step_reward = 1

        # update dino icon_width, icon_height, position
        dino_jump_type = len(self.dino.jump_heights)
        if self.dino.jump_idx == dino_jump_type-1:
            self.dino.update_icon_idx(0)
            self.dino.update_jump_idx(0)
        elif 0 < self.dino.jump_idx < dino_jump_type-1:
            if action == 2:
                self.dino.update_icon_idx(4)
            else:
                self.dino.update_icon_idx(2)
            self.dino.update_jump_idx(self.dino.jump_idx+1)
        elif self.dino.jump_idx == 0:
            if action == 0:
                self.dino.switch_walk_icon()
            elif action == 1:
                self.dino.update_icon_idx(2)
                self.dino.update_jump_idx(self.dino.jump_idx + 1)
            elif action == 2:
                self.dino.update_icon_idx(3)
                self.dino.update_jump_idx(0)
        current_dino_y = self.dino_base_y - self.dino.jump_height
        self.dino.set_position(self.dino_base_x, current_dino_y)

        # spawn object to the environment
        if self.step_count % 10 == 0:
            # spawn obstacle items
            if self.np_random.random() < 0.9:
                # spawn rock item
                if self.np_random.random() < 0.7:
                    rock_type = self.np_random.randint(2)  # select between rock 0 (1x1) and 2 (2x1)
                    if rock_type == 1:
                        rock_type = 2
                    spawned_rock = RockItem("rock_{}".format(self.rock_count), rock_type, self.x_min, self.x_max, self.y_min, self.y_max)
                    spawned_rock.set_position(self.x_max, self.y_row_1)
                    self.elements.append(spawned_rock)
                    self.rock_count += 1

                # spawn bird item
                else:
                    # row 3 bird (60%)
                    if self.np_random.random() <= 0.6:
                        spawned_bird = BirdItem("bird_{}".format(self.bird_count), self.x_min, self.x_max, self.y_min, self.y_max)
                        y_bird = self.y_row_3
                    # row 2 bird (40%)
                    else:
                        spawned_bird = BirdItem("bird_{}".format(self.bird_count), self.x_min, self.x_max, self.y_min, self.y_max)
                        y_bird = self.y_row_2
                    # row 1 bird (0%)
                    # else:
                    #     spawned_bird = BirdItem("bird_{}".format(self.bird_count), self.x_min, self.x_max, self.y_min, self.y_max)
                    #     y_bird = self.y_row_1
                    spawned_bird.set_position(self.x_max, y_bird)
                    self.elements.append(spawned_bird)
                    self.bird_count += 1

            # spawn energy item
            else:
                energy_row = self.np_random.randint(1, 2)  # select only row 2
                # row 3
                if energy_row == 0:
                    spawned_energy = EnergyItem("energy_{}".format(self.energy_count), self.x_min, self.x_max, self.y_min, self.y_max)
                    y_energy = self.y_row_3
                # row 2
                elif energy_row == 1:
                    spawned_energy = EnergyItem("energy_{}".format(self.energy_count), self.x_min, self.x_max, self.y_min, self.y_max)
                    y_energy = self.y_row_2
                # row 1
                elif energy_row == 2:
                    spawned_energy = EnergyItem("energy_{}".format(self.energy_count), self.x_min, self.x_max, self.y_min, self.y_max)
                    y_energy = self.y_row_1
                spawned_energy.set_position(self.x_max, y_energy)
                self.elements.append(spawned_energy)
                self.energy_count += 1

            # move elements: check for collision
        for element in self.elements:
            if isinstance(element, BirdItem):
                if element.get_position()[0] <= self.x_min:
                    self.elements.remove(element)
                else:
                    element.move(-50, 0)

                if self.has_collided(self.dino, element):
                    done = True
                    # step_reward -= 50
                    self.elements.remove(self.dino)

            if isinstance(element, RockItem):
                if element.get_position()[0] <= self.x_min:
                    self.elements.remove(element)
                else:
                    element.move(-50, 0)

                if self.has_collided(self.dino, element):
                    done = True
                    # step_reward -= 50
                    self.elements.remove(self.dino)

            if isinstance(element, EnergyItem):
                if element.get_position()[0] <= self.x_min:
                    self.elements.remove(element)
                else:
                    element.move(-50, 0)

                if self.has_collided(self.dino, element):
                    step_reward += 20
                    self.current_energy += 500
                    self.current_energy = element.clamp(self.current_energy, 0, self.max_energy)
                    self.elements.remove(element)

        if self.current_energy <= 0:
            done = True

        if done:
            step_reward -= 15
            print(f'finish episode {self.episode_idx} with reward={self.ep_reward+step_reward} (max: {self.max_reward})')

        self.ep_reward += step_reward

        if self.ep_reward > self.max_reward:
            self.max_reward = self.ep_reward

        self.draw_on_canvas()
        # self.render()

        return self.canvas, step_reward, done, {}

    def render(self, mode="human"):
        if mode == "human":
            cv2.imshow("Dino Game", self.canvas)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()
        print('close gym environment')

    def draw_on_canvas(self):
        self.canvas = np.ones(self.observation_shape)

        for element in self.elements:
            height, width, _ = element.icon_image.shape
            x, y = element.x, element.y
            right_x = x+width
            upper_y = y-height
            if right_x <= self.x_max:
                self.canvas[int(upper_y):int(y), int(x):int(right_x)] = element.icon_image
            elif x == self.x_max:
                pass
            # elif right_x > self.x_max:
            #     x_diff = right_x - self.x_max
            #     self.canvas[upper_y:y, x:x_diff] = element.icon_image[:, 0:x_diff]

        self.canvas[self.y_max:, :] = [0, 0, 0]
        text = 'Energy Left: {} | Rewards: {} | Max Reward: {}'.format(self.current_energy, self.ep_reward, self.max_reward)
        self.canvas = cv2.putText(self.canvas, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 1)

    def has_collided(self, element1, element2):
        x_element1, y_element1 = self.get_center_position(element1)
        x_element2, y_element2 = self.get_center_position(element2)

        x_col = 2 * abs(x_element1 - x_element2) <= (element1.icon_width + element2.icon_width)
        y_col = 2 * abs(y_element1 - y_element2) <= (element1.icon_height + element2.icon_height)

        collided = x_col and y_col

        # if collided:
        #     print(f'{element1.name} collided with {element2.name}')

        return collided

    @staticmethod
    def get_center_position(element):
        x, y = element.get_position()
        new_x = x + int(element.icon_width / 2)
        new_y = y - int(element.icon_height / 2)
        return new_x, new_y


ACTION_MEANING = {
    0: "do nothing",
    1: "jump",
    2: "duck"
}
