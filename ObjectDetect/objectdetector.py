#!/usr/bin/env python
# -*-coding:utf-8-*-
import numpy as np
import time
import cv2
import os


class ObjectDetector:
    def __init__(self):
        # 이미지를 불러오는 경로에 관한 변수
        self.__path = None

        # 영상에서 찾고자하는 물체이미지를 저장한 변수
        self.__key = list()
        self.__cmp_image = dict()

        # 물체이미지에 관한 정보를 담는 변수
        self.__kp = dict()
        self.__des = dict()

        # 물체이미지의 특징점을 미리 계산해
        self.__sift = cv2.xfeatures2d.SIFT_create()
        self.__flann = None

    # 물체이미지가 있는 디렉토리 경로를 설정하는 메소드
    def set_path(self, path):
        self.__path = path

    # 물체이미지의 데이터와 이름을 설정하는 메소드
    def set_cmp_image(self, key):
        if len(key) == 0:
            raise ValueError

        self.__key = key
        for k in key:
            img = cv2.imread(self.__path + "/" + k[0] + ".jpg")
            if img is None:
                print("[WARNING] " + k[0] + ".png not found!")
                continue
            self.__cmp_image.update({k[0]: img})

    # 물체인식에 필요한 비교이미지의 특징점을 구하는 메소드
    def set_initialize(self):
        for key in self.__key:
            self.__kp[key[0]], self.__des[key[0]] = self.__sift.detectAndCompute(self.__cmp_image[key[0]], None)
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.__flann = cv2.FlannBasedMatcher(index_params, search_params)

    def __call__(self, src, fps):
        src_kp, src_des = self.__sift.detectAndCompute(src, None)

        matches = dict()
        for key in self.__key:
            matches.update({key[0]: self.__flann.knnMatch(src_des, self.__des[key[0]], k=2)})

        good = dict()
        for key in self.__key:
            good.update({key[0]: []})
            good[key[0]] = [m for m, n in matches[key[0]] if m.distance < 0.6 * n.distance]

        result = src
        for key in self.__key:
            if len(good[key[0]]) > key[1]:
                src_pts = np.float32([src_kp[m.queryIdx].pt for m in good[key[0]]]).reshape(-1, 1, 2)
                dst_pts = np.float32([self.__kp[key[0]][m.trainIdx].pt for m in good[key[0]]]).reshape(-1, 1, 2)

                sum = np.sum((src_pts - dst_pts) ** 2)
                num_all = src_pts.shape[0] * src_pts.shape[1]
                mse = sum / num_all
                if mse < key[2]:
                    print(key[0][0].upper() + key[0][1:] + " Sign Detect")
                    result = cv2.drawMatches(src, src_kp, self.__cmp_image[key[0]], self.__kp[key[0]],
                                             good[key[0]], None, flags=2)
        fps = "FPS : %0.1f" % fps
        result = cv2.putText(result, fps, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return result


if __name__ == '__main__':
    obj = ObjectDetector()
    obj.set_path(os.getcwd() + "/data")

    # k[0] : Image Name
    k = [['parking', 4, 20000]]

    obj.set_cmp_image(k)
    obj.set_initialize()

    cap = cv2.VideoCapture(0)
    prev_time = 0
    while cap.isOpened():
        ret, fm = cap.read()
        fm = cv2.resize(fm, (640, 240))

        # FPS 를 구하는 과정
        cur_time = time.time()
        f = 1 / (cur_time - prev_time)
        prev_time = cur_time

        pre_t = time.time()
        rst = obj(fm, f)
        print(str(round(time.time() - pre_t, 6)) + "(초)")

        if rst is not None:
            cv2.imshow("Result", rst)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

