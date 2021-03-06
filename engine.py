import math
import cv2

from states import states


class SceneState:
    def __init__(self, identifier):
        self.id = identifier

    def set(self, value):
        states[self.id] = value

    def get(self):
        return states.get(self.id)


class Car:
    def __init__(self, initial_pos, width_of_car = 6):
        self.position = initial_pos
        self.width_of_car = width_of_car

        # Tha main program calls `image_to_speed()` once per second.
        # And it is calculated by n sub-steps. Within every single sub-step,
        # we suppose that the car's two wheels run for 1000/n ms.
        self.freq_of_call = 1   # 1 call per second
        self.sub_steps_per_call = 10
        self.sub_step_duration = 1 / self.sub_steps_per_call  # unit: second

    def move(self, left, right):
        """
        :param left, right: car wheels' speed
        """
        d_left = left * self.sub_step_duration
        d_right = right * self.sub_step_duration
        d_center = (d_left + d_right) / 2
        phi = (d_right - d_left) / self.width_of_car

        x = self.position[0] + d_center * math.cos(self.position[2] + phi / 2)
        y = self.position[1] + d_center * math.sin(self.position[2] + phi / 2)
        theta = self.position[2] + phi
        self.position = (x, y, theta)


def run(f, seconds, position, log, from_ip, unity_view1, unity_view2):
    """
    Engine entry.
    :param f: students' image_to_speed function
    :param seconds: time to run our car
    :param position: the car is stateless, so every time when we need to calculate its new position, current position
                     should be provided firstly.
    :param log: a helper variable for students' debugging
    :param from_ip: client's ip address
    :param unity_view1: bird view
    :param unity_view2: in-car view
    :return:
    """
    car = Car(position)
    pos = [0, 0, 0]

    state = SceneState(from_ip)
    left, right = f(unity_view1, unity_view2, state)
    # cv2.imwrite("./in-car-view.jpg", unity_view2)
    for i in range(seconds * car.freq_of_call):
        for j in range(car.sub_steps_per_call):
            car.move(right, left)
            pos = car.position

    return {
        "position": pos,
        "log": log
    }

