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

        self.canvas_shape = (350, 900, 3)
        self.canvas_position_setup()
        self.dino_setup()


        self.setup_spaces()

        self.elements = []

        self.ep_reward = 0
        self.step_count = 0
        self.max_reward = -1000

        self.current_energy = 1000
        self.max_energy = 1000

        self.episode_idx = -1
        self.verbose = verbose

        self.video_count = 0

        self.seed()

    def canvas_position_setup(self):
        self.y_row_3 = 200
        self.y_row_2 = 250
        self.y_row_1 = 300
        self.x_min = 0
        self.x_max = 900
        self.y_min = 75
        self.y_max = 300
        self.canvas = np.ones(self.canvas_shape)

    def dino_setup(self):
        self.dino_base_x = 150
        self.dino_base_y = self.y_max
        self.dino = Person("dino person", self.dino_base_x, self.dino_base_x, self.y_min, self.y_max)
        self.dino.set_position(self.dino_base_x, self.dino_base_y)

    def setup_spaces(self):
        """
        format: [dino_x, dino_y, dino_w, dino_h, obj1_x, obj1_y, obj1_w, obj1_h, obj1_type, obj2_x, obj2_y, obj2_w, obj2_h, obj2_type]
        obj_type = {0: 'rock/bird', 1:'energy'}
        """
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                            high=np.array([self.dino_base_x, self.y_max, 50, 100, self.x_max, self.y_max, 150, 100, 1, self.x_min, self.y_min, 150, 100, 1]),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(3, )

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

        self.frames = []

        self.draw_on_canvas()
        new_obs = self.get_observation()

        return new_obs

    def step(self, action):
        # TODO: penalty if jump for nothing1111
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
            if self.np_random.random() < 0.8:
                # spawn rock item
                if self.np_random.random() < 0.6:
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
                    # self.elements.remove(self.dino)

                # reward when miss the BirdItem
                if self.dino.x == element.x and not self.has_collided(self.dino, element):
                    step_reward += 5

            if isinstance(element, RockItem):
                if element.get_position()[0] <= self.x_min:
                    self.elements.remove(element)
                else:
                    element.move(-50, 0)

                if self.has_collided(self.dino, element):
                    done = True
                    # step_reward -= 50
                    # self.elements.remove(self.dino)

                # reward when miss the RockItem
                if element.icon_idx == 0:  # rock 1x1
                    if self.dino.x == element.x and not self.has_collided(self.dino, element):
                        step_reward += 5
                elif element.icon_idx == 2:  # rock 2x1
                    if (self.dino.x == element.x or self.dino.x == element.x+50) and not self.has_collided(self.dino, element):
                        step_reward += 5

            if isinstance(element, EnergyItem):
                if element.get_position()[0] <= self.x_min:
                    self.elements.remove(element)
                else:
                    element.move(-50, 0)

                if self.has_collided(self.dino, element):
                    step_reward += 20
                    self.current_energy += 100
                    self.current_energy = element.clamp(self.current_energy, 0, self.max_energy)
                    self.elements.remove(element)

                # penalty when miss the EnergyItem
                if self.dino.x == element.x and not self.has_collided(self.dino, element):
                    step_reward -= 20

        if self.current_energy <= 0:
            done = True

        change_max_reward = False
        if done:
            step_reward -= 30
            self.ep_reward += step_reward
            if self.ep_reward > self.max_reward:
                self.max_reward = self.ep_reward
                change_max_reward = True
            print(f'finish episode {self.episode_idx} with reward={self.ep_reward} (max: {self.max_reward})')
        else:
            self.ep_reward += step_reward

        self.draw_on_canvas()
        # self.render()
        self.frames.append(self.canvas)

        if done:
            for i in range(3):
                self.frames.append(self.canvas)

        if done and change_max_reward == True and self.max_reward >= 30:
            if self.video_count <= 100:
                self.set_video_writer(self.max_reward)
                for frame in self.frames:
                    self.writer.write(np.uint8(frame * 255.0))
                self.writer.release()
                self.video_count += 1

        new_obs = self.get_observation(done)

        return new_obs, step_reward, done, {}

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
        self.canvas = np.ones(self.canvas_shape)

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

    def get_observation(self, done=False):
        """
        format: [dino_x, dino_y, dino_w, dino_h, obj1_x, obj1_y, obj1_w, obj1_h, obj1_type, obj2_x, obj2_y, obj2_w, obj2_h, obj2_type]
        """
        if len(self.elements) <= 0:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if len(list(filter(lambda x: isinstance(x, Person), self.elements))) <= 0:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if done:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        new_obs = []
        for element in self.elements:
            if isinstance(element, Person):
                new_obs.append(element.x)
                new_obs.append(element.y)
                new_obs.append(element.icon_width)
                new_obs.append(element.icon_height)
            elif isinstance(element, RockItem) or isinstance(element, BirdItem):
                new_obs.append(element.x)
                new_obs.append(element.y)
                new_obs.append(element.icon_width)
                new_obs.append(element.icon_height)
                new_obs.append(0)
            elif isinstance(element, EnergyItem):
                new_obs.append(element.x)
                new_obs.append(element.y)
                new_obs.append(element.icon_width)
                new_obs.append(element.icon_height)
                new_obs.append(1)

        length_obs = len(new_obs)
        if length_obs < 14:
            for i in range(14-length_obs):
                new_obs.append(0)

        return new_obs

    def has_collided(self, element1, element2):
        x_element1, y_element1 = self.get_center_position(element1)
        x_element2, y_element2 = self.get_center_position(element2)

        x_col = 2 * abs(x_element1 - x_element2) < (element1.icon_width + element2.icon_width)
        y_col = 2 * abs(y_element1 - y_element2) < (element1.icon_height + element2.icon_height)

        collided = x_col and y_col

        # if collided:
        #     print(f'{element1.name} collided with {element2.name}')
        #     print(f'{element1.name}: x={element1.x} y={element1.y} w={element1.icon_width} h={element1.icon_height}')
        #     print(f'{element2.name}: x={element2.x} y={element2.y} w={element2.icon_width} h={element2.icon_height}')

        return collided

    @staticmethod
    def get_center_position(element):
        x, y = element.get_position()
        new_x = x + int(element.icon_width / 2)
        new_y = y - int(element.icon_height / 2)
        return new_x, new_y

    def set_video_writer(self, max_reward):
        vid_path = f'/Users/thakorns/Desktop/Eyp/codebases/rl-dino-gym/training_pipeline/videos/video_{self.episode_idx}_{max_reward}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 10
        self.writer = cv2.VideoWriter(vid_path, fourcc, fps, (self.canvas_shape[1], self.canvas_shape[0]), True)

ACTION_MEANING = {
    0: "do nothing",
    1: "jump",
    2: "duck"
}
