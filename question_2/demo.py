import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import os
import cv2

def draw_bbx(data_path):
    img = cv2.imread(data_path)
    img_draw = cv2.imread(data_path)
    size = img.shape
    w = size[1]
    h = size[0]
    img_area = w * h

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 120, 220) 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))


    dilated = cv2.dilate(canny_image, kernel)

    eroded = cv2.erode(dilated, kernel)

    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):
        max_area = -1
        x, y, w, h = cv2.boundingRect(contours[i])  # 得出轮廓最小外接矩形左上角坐标及长、宽信息
        if w * h < 0.9 * img_area and w * h >= 0.3 * img_area:
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 0, 255), 2)


    return img_draw


if __name__ == "__main__":
    data_path = '/media/data1/xyx/dataset/medical_test_images/'
    filenames = os.listdir(data_path)
    for i, img in enumerate(filenames):
        imgs = cv2.imread(os.path.join(data_path, img))
        img_path = os.path.join(data_path, filenames[i])
        res = draw_bbx(img_path)
        cv2.imwrite('./outputs/res%d.jpg' % (i), res)




