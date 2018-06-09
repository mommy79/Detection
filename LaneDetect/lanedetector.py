#!/usr/bin/env python
# -*-coding:utf-8-*-
import numpy as np
import cv2
import time


class LaneDetector:
    # 기본변수 셋팅 후 전처리를 진행하는 생성자
    def __init__(self, img):
        # 이미지 처리과정을 보여주는 변수
        self.__process = []

        # 원본 이미지 관련 변수
        self.__src_image = img
        self.__src_height, self.__src_width = self.__src_image.shape[:2]

        # Perspective 이미지 관련 변수
        self.__ratio = (1, 1)
        self.__pers_height = int(self.__src_height * self.__ratio[0])
        self.__pers_width = int(self.__src_width * self.__ratio[1])
        self.__init_position = np.float32([[40, 0], [0, 99], [430, 0], [480, 99]])
        self.__trans_position = np.float32([[50, 0], [40, 99], [420, 0], [440, 99]])

        # 전처리에 필요한 도구를 만드는 변수와 메소드(가로선, 네비게이터)
        self.__num_of_section = 10
        self.__horizon_image, self.__navigator_image = self._set_tools()

        # 전처리 과정을 진행하는 메소드
        self.__rst_image, self.__error = self._run()

    # 처리과정을 출력하는 메소드
    def fn_show_process(self):
        for i in range(0, len(self.__process)):
            cv2.imshow(self.__process[i][0], self.__process[i][1])

    # 결과이미지를 반환하는 메소드
    def fn_get_result(self):
        return self.__rst_image, self.__error

    # 결과이미지를 만드는 메소드
    def _set_tools_navigation(self, value):
        mask = cv2.line(self.__navigator_image, (value, self.__src_height // 2 - 5),
                        (value, self.__src_height // 2 + 5), (255, 0, 255), 1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        __, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1 = cv2.bitwise_and(self.__src_image, self.__src_image, mask=mask_inv)
        img2 = cv2.bitwise_and(self.__navigator_image, self.__navigator_image, mask=mask)
        img = cv2.add(img1, img2)
        return img

    # 차선인식에 필요한 도구를 만드는 메소드
    def _set_tools(self):
        # 가로선 이미지를 생성
        horizon = np.zeros((self.__src_height, self.__src_width), np.uint8)
        offset = int(round(self.__src_height / self.__num_of_section))
        pre_pos = 0
        for i in range(0, self.__num_of_section):
            horizon = cv2.line(horizon, (0, pre_pos + offset), (self.__src_width, pre_pos + offset), 255, 1)
            pre_pos += offset

        # 기준선 이미지를 생성
        navigator = np.zeros((self.__src_height, self.__src_width, 3), np.uint8)
        cnt_point = (self.__src_width // 2, self.__src_height // 2)
        cv2.line(navigator, (cnt_point[0], cnt_point[1] - 10),
                 (cnt_point[0], cnt_point[1] + 10), (255, 0, 0), 2)
        cv2.line(navigator, (cnt_point[0] - 100, cnt_point[1]),
                 (cnt_point[0] + 100, cnt_point[1]), (255, 0, 0), 2)
        cv2.line(navigator, (cnt_point[0] - 100, cnt_point[1] - 5),
                 (cnt_point[0] - 100, cnt_point[1] + 5), (255, 0, 0), 2)
        cv2.line(navigator, (cnt_point[0] + 100, cnt_point[1] - 5),
                 (cnt_point[0] + 100, cnt_point[1] + 5), (255, 0, 0), 2)
        return horizon, navigator

    # 전치리 과정을 진행하는 메소드
    def _run(self):
        # 원본 이미지 복사
        frame = self.__src_image.copy()
        self.__process.append(["Source Image", frame])

        # 그레이 이미지 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.__process.append(["Gray Image", frame])

        # 이진화 이미지 변환
        __, frame = cv2.threshold(frame, 240, 255, cv2.THRESH_BINARY)
        self.__process.append(["Threshold Image", frame])

        # Canny 이미지 변환
        frame = cv2.Canny(frame, 150, 200, apertureSize=7)
        self.__process.append(["Canny Image", frame])

        # Perspective 이미지 변환
        tmp = cv2.getPerspectiveTransform(self.__init_position, self.__trans_position)
        frame = cv2.warpPerspective(frame, tmp, (self.__pers_width, self.__pers_height))
        self.__process.append(["Perspective Image", frame])

        # Canny 이미지와 Horizon 이미지의 교차점 탐색
        frame = cv2.bitwise_and(frame, self.__horizon_image)
        self.__process.append(["Dot Image", frame])

        # frame[0]: 좌, frame[1]: 중앙선, frame[2]: 우 좌우의
        # 좌우에서 각 행별 중앙선에 가까운 픽셀을 탐색해 사전 형태로 저장
        frame = np.hsplit(frame, [self.__src_width // 2, self.__src_width // 2 + 1])
        pxl_info = [dict({i: np.argwhere(frame[0][i]).transpose().reshape(-1) for i in range(0, self.__src_height)}),
                    dict({i: np.argwhere(frame[2][i]).transpose().reshape(-1) for i in range(0, self.__src_height)})]
        frame = np.hstack((frame[0], frame[1], frame[2]))

        for i, info in enumerate(pxl_info):
            tmp = info.copy()
            for key in info:
                if len(info[key]) == 0:
                    tmp.pop(key)
                else:
                    if i == 0:
                        tmp[key] = max(info[key])
                    else:
                        tmp[key] = min(info[key])
            pxl_info[i] = tmp

        # 좌측 또는 우측의 차선이 검출되지 않을 경우의 예외처리
        center = dict()
        # TODO: 이 if문 때문에 None이 Return되서 에러뜸
        if len(pxl_info[0]) < 3 or len(pxl_info[1]) < 3:
            if len(pxl_info[1]) < 3 <= len(pxl_info[0]):
                # TODO: 수정해야함 여기서 None이 Return되서 에러뜸
                for key in pxl_info[0]:
                    pxl_info[1].update({key: pxl_info[0][key]})
            elif len(pxl_info[0]) < 3 <= len(pxl_info[1]):
                # TODO: 수정해야함
                for key in pxl_info[1]:
                    pxl_info[0].update({key: self.__src_width // 2 - pxl_info[1][key]})
            else:
                print("[WARNING]Lane not found")
                err = 0
                frame = cv2.add(self.__src_image, self.__navigator_image)
                self.__process.append(["Result Image", frame])
                return frame, err
        # 좌우측의 차선이 모두 검출되는 경우 중앙값 구하기
        for key in pxl_info[0]:
            if pxl_info[1].get(key) is not None:
                center.update({key: (pxl_info[0][key] + (self.__src_width // 2 + pxl_info[1][key])) // 2})
        tmp = np.array(list(center.values()))
        try:
            val = int(np.mean(tmp))
            err = val - self.__src_width // 2
            frame = self._set_tools_navigation(val)
        except ZeroDivisionError:
            print("[WARNING]ZeroDivisionError")
            err = 0
            frame = cv2.add(self.__src_image, self.__navigator_image)
            self.__process.append(["Result Image", frame])
            return frame, err
        self.__process.append(["Result Image", frame])
        return frame, err


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        pre_t = time.time()

        __, fm = cap.read()
        fm = cv2.resize(fm, (480, 100))
        fm = fm[60:, :, :]

        node = LaneDetector(fm)
        # 처리과정 이미지 출력
        node.fn_show_process()

        # 결과 이미지 출력 및 Error 출력
        result, error = node.fn_get_result()
        print(error)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

        print(str(round(time.time() - pre_t, 6)) + "(초)")
    cap.release()
    cv2.destroyAllWindows()

