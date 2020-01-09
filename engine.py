import math
import cv2


class Car:
    def __init__(self, initial_pos, width_of_car):
        self.position = initial_pos
        self.width_of_car = width_of_car

        # Tha main program calls `image_to_speed()` once a second.
        # And it is calculated by n steps. Within every single step,
        # we suppose that the car's two wheels run separately for 1000/n ms.
        self.freq_of_call = 1   # 1 call per second
        self.sub_steps_per_call = 10
        self.sub_step_duration = 1 / self.sub_steps_per_call  # unit: second

    def move(self, left, right):
        d_left = left * self.sub_step_duration
        d_right = right * self.sub_step_duration
        d_center = (d_left + d_right) / 2
        phi = (d_right - d_left) / self.width_of_car

        x = self.position[0] + d_center * math.cos(self.position[2] + phi / 2)
        y = self.position[1] + d_center * math.sin(self.position[2] + phi / 2)
        theta = self.position[2] + phi
        self.position = (x, y, theta)


def run(f, seconds, position, log, unity_view1, unity_view2):
    # Suppose the width of our car is 6 pixel
    w = 6
    # We simply ignore the rest part of the car except its head.

    car = Car(position, w)
    pos = None

    left, right = f(unity_view1, unity_view2)
    # cv2.imwrite("/Users/zhouyuan/hi.jpg", unity_view2)
    for i in range(seconds * car.freq_of_call):
        for j in range(car.sub_steps_per_call):
            car.move(right, left)
            pos = car.position

    return {
        "position": pos,
        "log": log
    }

