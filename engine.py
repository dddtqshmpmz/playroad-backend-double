import math
import time
import numpy as np
import cv2

sign_straight = cv2.imread("static/signs/straight.jpg")
sign_left = cv2.imread("./static/signs/left.jpg")
sign_right = cv2.imread("./static/signs/right.jpg")
sign_imgs = {"straight": sign_straight, "left": sign_left, "right": sign_right}


class Car:
    def __init__(self, initial_pos, width_of_car, img):
        self.position = initial_pos
        self.width_of_car = width_of_car
        self.img = img

        self.view_w = 30
        self.view_h = 15
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

    def view(self, signs=[]):
        tmp = np.zeros(shape=(self.view_w, road_w))
        img = np.concatenate((tmp, self.img))
        tmp = np.zeros(shape=(self.view_w+road_h, self.view_h))
        img = np.concatenate((img, tmp), axis=1)

        x, y, theta = self.position
        y += self.view_w

        m = cv2.getRotationMatrix2D((x, y), theta / math.pi * 180, 1)
        rotated = cv2.warpAffine(img, m, (road_w*2, road_h*2))
        cropped = rotated[math.ceil(y - self.view_w / 2):math.ceil(y + self.view_w / 2),
                            math.ceil(x + 0.1):math.ceil(x + self.view_h + 0.1)]
        # cv2.imwrite("./rotated.jpg", rotated)
        filename = str(time.time())
        y -= self.view_w

        pi = math.pi
        twopi = 2 * pi
        sign = "none"
        for sg in signs:
            pos = sg["position"]
            dist = np.linalg.norm([x - pos[0], y - pos[1]])
            if dist < 5:
                if 0.2*pi < (theta - pos[2])%twopi < 0.8*pi:
                    sign = sg["sign"]

        # cropped = 255 - cropped
        sign_img = sign_imgs.get(sign)
        cv2.imwrite("./static/imgs/{0}.jpg".format(filename), cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE))
        return "/static/imgs/{0}.jpg".format(filename), cropped, "/static/signs/{0}.jpg".format(sign), sign_img


road_w = 100
road_h = 200


def run(f, seconds, position, map, log):
    # Suppose the width of our car is 6 cm
    w = 6
    # We simply ignore the rest part of the car except its head.
    img = np.zeros(shape=(road_h, road_w))
    for line in map['lines']:
        cv2.polylines(img, pts=np.int32([line]), isClosed=False, color=(255, 0, 0))

    car = Car(position, w, img)
    coors = []

    for i in range(seconds * car.freq_of_call):
        view = car.view(map["signs"])
        left, right = f(view[1], view[3])
        for j in range(car.sub_steps_per_call):
            car.move(right, left)
            coors.append(car.position)

    view = car.view(map["signs"])
    return {
        "positions": coors,
        "view": view[0],
        "sign": view[2],
        "log": log
    }

