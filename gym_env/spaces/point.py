class Point(object):
    def __init__(self, name, x_min, x_max, y_min, y_max):
        self.name = name
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x = 0
        self.y = 0
        self.icon_width = 0
        self.icon_height = 0

    def get_position(self):
        return self.x, self.y

    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.apply_clamp()

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        self.apply_clamp()

    def apply_clamp(self):
        # object coordinate is located at the bottom-left of the object
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_width)
        self.y = self.clamp(self.y, self.y_min + self.icon_height, self.y_max)

    @staticmethod
    def clamp(val, min_val, max_val):
        return max(min(val, max_val), min_val)
