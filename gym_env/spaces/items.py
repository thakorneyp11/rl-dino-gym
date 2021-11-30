import cv2
from gym_env.spaces.point import Point


class EnergyItem(Point):
    def __init__(self, name, x_min, x_max, y_min, y_max):
        super(EnergyItem, self).__init__(name, x_min, x_max, y_min, y_max)

        abs_path = '/Users/thakorns/Desktop/Eyp/codebases/rl-dino-gym/gym_env/spaces/images'
        image = cv2.imread(f'{abs_path}/energy_2.png')
        self.icon_image = cv2.resize(image, (50, 50))
        self.icon_width = 50
        self.icon_height = 50

        self.energy_score = 500

    def set_energy_score(self, energy):
        self.energy_score = energy

    def get_energy_score(self):
        return self.energy_score


class RockItem(Point):
    icon_names = ['rock_1_1', 'rock_1_2', 'rock_2_1', 'rock_2_2', 'rock_3_1', 'rock_3_2']
    abs_path = '/Users/thakorns/Desktop/Eyp/codebases/rl-dino-gym/gym_env/spaces/images'
    icon_info = {
        'rock_1_1': {'image': cv2.imread(f'{abs_path}/rock_1_1.png'), 'width': 50, 'height': 50},
        'rock_1_2': {'image': cv2.imread(f'{abs_path}/rock_1_2.png'), 'width': 50, 'height': 100},
        'rock_2_1': {'image': cv2.imread(f'{abs_path}/rock_2_1.png'), 'width': 100, 'height': 50},
        'rock_2_2': {'image': cv2.imread(f'{abs_path}/rock_2_2.png'), 'width': 100, 'height': 100},
        'rock_3_1': {'image': cv2.imread(f'{abs_path}/rock_3_1.png'), 'width': 150, 'height': 50},
        'rock_3_2': {'image': cv2.imread(f'{abs_path}/rock_3_2.png'), 'width': 150, 'height': 100}
    }
    for key in icon_info.keys():
        image = icon_info[key]['image']
        width = icon_info[key]['width']
        height = icon_info[key]['height']
        icon_info[key]['image'] = cv2.resize(image, (width, height))

    def __init__(self, name, icon_idx, x_min, x_max, y_min, y_max):
        super(RockItem, self).__init__(name, x_min, x_max, y_min, y_max)

        icon_name = self.icon_names[icon_idx]
        self.icon_image = self.icon_info[icon_name]['image']
        self.icon_width = self.icon_info[icon_name]['width']
        self.icon_height = self.icon_info[icon_name]['height']

        self.penalty_score = 100  # should be positive number

        self.image_setup()

    def set_penalty_score(self, penalty):
        self.penalty_score = penalty

    def get_penalty_score(self):
        return self.penalty_score

    def image_setup(self):
        pass


class BirdItem(Point):
    def __init__(self, name, x_min, x_max, y_min, y_max):
        super(BirdItem, self).__init__(name, x_min, x_max, y_min, y_max)

        abs_path = '/Users/thakorns/Desktop/Eyp/codebases/rl-dino-gym/gym_env/spaces/images'
        image = cv2.imread(f'{abs_path}/bird.png')
        self.icon_image = cv2.resize(image, (50, 50))
        self.icon_width = 50
        self.icon_height = 50

        self.penalty_score = 50  # should be positive number

    def set_penalty_score(self, penalty):
        self.penalty_score = penalty

    def get_penalty_score(self):
        return self.penalty_score

