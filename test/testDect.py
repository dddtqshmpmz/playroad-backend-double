import cv2
import numpy as np


class detection():
    def __init__(self):
        self.H_low_thresh = 110
        self.H_high_thresh = 125
        self.S_low_thresh = 120
        self.V_low_thresh = 90
        self.aspect_ratio_thresh = 1.5
        self.use_h_only = 0
        self.img_width = 640
        self.img_height = 480

    def show_circles(self, im, circles):
        """
        show the circles detected by hough transform
        """
        if len(circles) != 0:
            for i in range(0, len(circles)):
                cv2.circle(im, (circles[i][0], circles[i][1]), circles[i][2], (0, 0, 255), 2)
        cv2.imshow("circles", im)
        cv2.waitKey(0)

    def show_rects(self, im, rects):
        """
        show one or more than one rectangles
        """
        if len(rects) != 0:
            if type(rects[0]) == list:
                for i in range(0, len(rects)):
                    cv2.rectangle(im, (rects[i][0], rects[i][1]), (rects[i][2], rects[i][3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(im, (rects[0], rects[1]), (rects[2], rects[3]), (0, 0, 255), 2)
        cv2.imshow("rects", im)
        cv2.waitKey(10)

    def show_rects_id(self, im, rects):
        """
        show one or more than one rectangles
        """
        if len(rects) != 0:
            if type(rects[0]) == list:
                for i in range(0, len(rects)):
                    cv2.rectangle(im, (rects[i][0], rects[i][1]), (rects[i][2], rects[i][3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(im, (rects[0], rects[1]), (rects[2], rects[3]), (0, 0, 255), 2)
        cv2.imshow("rects", im)
        cv2.waitKey(10)

    def draw_rects(self, im, rects):
        """
        draw rects on im
        """
        if len(rects) != 0:
            if type(rects[0]) == list:
                for i in range(0, len(rects)):
                    cv2.rectangle(im, (rects[i][0], rects[i][1]), (rects[i][2], rects[i][3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(im, (rects[0], rects[1]), (rects[2], rects[3]), (0, 0, 255), 2)

    def change_threshold(self, h_low, h_high):

        """
        change the h thresholds for color segmentation.
        adopt the h for segmentation only, only can be used
        for Signal lamp detection
        """
        self.H_low_thresh = h_low
        self.H_high_thresh = h_high
        self.use_h_only = 1

    def change_threshold_all(self, h_low, h_high, s_low, v_low):
        self.H_low_thresh = h_low
        self.H_high_thresh = h_high
        self.S_low_thresh = s_low
        self.V_low_thresh = v_low
        self.use_h_only = 0

    def seg_color(self, im):
        """
        input: image
        output: the position of the sign
                rects:[[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax]....]
        """
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        blue = im[:, :, 0]
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        if self.use_h_only == 0:
            label = (h > self.H_low_thresh) * (h < self.H_high_thresh) * (s > self.S_low_thresh) * (
                        v > self.V_low_thresh)
        else:
            label = (h > self.H_low_thresh) * (h < self.H_high_thresh)
        gray = gray * label
        gray[label] = 255
        # cv2.imshow("gray",gray)
        # cv2.waitKey(0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        eroded = cv2.erode(gray, element)
        dilated = cv2.dilate(eroded, element)
        ret, thresh = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
        # cv2.imshow("thresh",thresh)
        # cv2.waitKey(0)
        # for opencv3
        # binary,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # for opencv2
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects = []

        for i in range(0, len(contours)):
            rect = cv2.boundingRect(contours[i])
            w = rect[2]
            h = rect[3]
            aspect_ratio = max(w, h) / (min(w, h) * 1.0)
            if w > 10 and h > 10 and aspect_ratio < self.aspect_ratio_thresh:
                rects.append([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])
        return rects

    def det_circle(self, img):
        """
        input: image
        output: the position of the circle
                circles:[[c_x-r,c_y-r,c_x+r,c_y+r],[c_x-r,c_y-r,c_x+r,c_y+r]....]
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # circles= cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,100,param1=120,param2=40,minRadius=5,maxRadius=40)
        # circles=cv2.HoughCircles(gray,cv2.cv.CV_HOUGH_GRADIENT,1,100,param1=120,param2=40,minRadius=1,maxRadius=40)
        # circles=cv2.HoughCircles(gray,cv2.cv.CV_HOUGH_GRADIENT,1,100,param1=120,param2=20,minRadius=1,maxRadius=40)
        # for opencv2
        # circles=cv2.HoughCircles(gray,cv2.CV_HOUGH_GRADIENT,1,100,param1=120,param2=40,minRadius=1,maxRadius=40)
        # for opencv3
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=120, param2=20, minRadius=1, maxRadius=160)

        # print(circles)

        res_circles = []
        if type(circles) == np.ndarray:
            for circle in circles[0]:
                x = int(circle[0])
                y = int(circle[1])
                r = int(circle[2])
                res_circles.append([x, y, r])
        return res_circles

    def enlarge_rects(self, rects):
        for i in range(0, len(rects)):
            w = rects[i][2] - rects[i][0]
            h = rects[i][3] - rects[i][1]
            w = w * 1.4
            h = h * 1.4

            c_x = (rects[i][0] + rects[i][2]) / 2
            c_y = (rects[i][1] + rects[i][3]) / 2

            rects[i][0] = int(round(c_x - w / 2))
            if rects[i][0] < 0:
                rects[i][0] = 0
            rects[i][1] = int(round(c_y - h / 2))
            if rects[i][1] < 0:
                rects[i][1] = 0
            rects[i][2] = int(round(c_x + w / 2))
            if rects[i][2] >= 640:
                rects[i][2] = 639
            rects[i][3] = int(round(c_y + h / 2))
            if rects[i][3] >= 480:
                rects[i][3] = 479

            # print rects[i]
        return rects

    def ensemble(self, img):
        """
        input:image
        output:
        """
        res_rects = []
        max_index = []
        orig_img = img.copy()
        rects = self.seg_color(img)

        if len(rects) == 0:
            return res_rects
        else:
            rects = self.enlarge_rects(rects)
            for i in range(0, len(rects)):
                rect = rects[i]
                xmin = rect[0]
                ymin = rect[1]
                xmax = rect[2]
                ymax = rect[3]

                if ymax - ymin < 300 and xmax - xmin < 300:
                    sign_roi = orig_img[ymin:ymax, xmin:xmax, :]
                    # print(xmin,ymin,xmax,ymax)
                    # cv2.imshow("roi",sign_roi)
                    # cv2.waitKey(0)
                    circles = self.det_circle(sign_roi)

                    # print(len(circles))

                    if len(circles) != 0:
                        res_rects.append(rect)
                        max_index.append((xmax - xmin) * (ymax - ymin))
        if len(max_index) != 0:
            index = max_index.index(max(max_index))
            dst_rect = res_rects[index]

            return dst_rect
        else:
            return res_rects

    def ensemble2(self, img):
        """
        input:image
        output:the position of the object
              rects:[xmin,ymin,xmax,ymax]
              only one sign, and if detect more than one sign,
              the sign with the largest area would be returned

        """
        res_rects = []
        max_index = []
        circles = self.det_circle(img)
        rects = self.seg_color(img)
        if len(circles) != 0 and len(rects) != 0:
            for i in range(0, len(circles)):
                for j in range(0, len(rects)):
                    r = circles[i][2]
                    inter_xmin = max(circles[i][0] - r, rects[j][0])
                    inter_ymin = max(circles[i][1] - r, rects[j][1])
                    inter_xmax = min(circles[i][0] + r, rects[j][2])
                    inter_ymax = min(circles[i][1] + r, rects[j][3])

                    rect_w = rects[j][2] - rects[j][0]
                    rect_h = rects[j][3] - rects[j][1]

                    inter_w = inter_xmax - inter_xmin
                    inter_h = inter_ymax - inter_ymin
                    if inter_w > 0 and inter_h > 0 and abs(
                            (2 * inter_w * inter_h * 1.0) / (4 * r * r + rect_w * rect_h) - 1) < 0.3:
                        res_rects.append(rects[j])
                        max_index.append(rect_w * rect_h)
        if len(max_index) != 0:
            index = max_index.index(max(max_index))
            return res_rects[index]
        else:
            return res_rects


detector = detection()

# im = cv2.imread("../static/signs/testu.jpg")
im = cv2.imread("/Users/zhouyuan/hi.jpg")

roi = im

rects = detector.ensemble(roi)
print(rects)
detector.show_rects(roi,rects)