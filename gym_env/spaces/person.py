import cv2
from gym_env.spaces.point import Point


class Person(Point):
    icon_names = ['left', 'right', 'jump', 'duck', 'jump-duck']
    abs_path = '/Users/thakorns/Desktop/Eyp/codebases/rl-dino-gym/gym_env/spaces/images'
    icon_info = {
        'left': {'image': cv2.imread(f'{abs_path}/dino_left.png'), 'width': 50, 'height': 100},
        'right': {'image': cv2.imread(f'{abs_path}/dino_right.png'), 'width': 50, 'height': 100},
        'jump': {'image': cv2.imread(f'{abs_path}/dino_jump.png'), 'width': 50, 'height': 100},
        'duck': {'image': cv2.imread(f'{abs_path}/dino_duck.png'), 'width': 50, 'height': 50},
        'jump-duck': {'image': cv2.imread(f'{abs_path}/dino_duck.png'), 'width': 50, 'height': 50}
    }
    for key in icon_info.keys():
        image = icon_info[key]['image']
        width = icon_info[key]['width']
        height = icon_info[key]['height']
        icon_info[key]['image'] = cv2.resize(image, (width, height))

    jump_heights = [0, 1, 2.5, 2.5, 2.5, 1]

    def __init__(self, name, x_min, x_max, y_min, y_max):
        super(Person, self).__init__(name, x_min, x_max, y_min, y_max)

        self.icon_idx = 0
        self.select_icon = self.icon_info[self.icon_names[self.icon_idx]]
        self.icon_image = self.select_icon['image']
        self.icon_width = self.select_icon['width']
        self.icon_height = self.select_icon['height']

        self.jump_idx = 0
        self.jump_height = self.jump_heights[self.jump_idx] * 50

    def update_icon_idx(self, icon_idx):
        self.icon_idx = icon_idx
        self.select_icon = self.icon_info[self.icon_names[self.icon_idx]]
        self.icon_image = self.select_icon['image']
        self.icon_width = self.select_icon['width']
        self.icon_height = self.select_icon['height']

    def update_jump_idx(self, jump_idx):
        self.jump_idx = jump_idx
        self.jump_height = self.jump_heights[self.jump_idx] * 50

    def switch_walk_icon(self):
        if self.icon_idx == 0:
            self.icon_idx = 1
        elif self.icon_idx == 1:
            self.icon_idx = 0
        self.update_icon_idx(self.icon_idx)
