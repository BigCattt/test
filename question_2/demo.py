import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import os

# watershedSegment
def watershedSegment(img_path):
    # '/media/data1/xyx/dataset/medical_test_images/0b82d520572111ec877d305a3a77b88e.jpg'
    img = cv.imread(img_path)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    #将图像转化为灰度图像
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #阈值化处理
    ret,thresh=cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # plt.imshow(thresh,cmap='gray')
    # plt.show()

    #noise removal
    #opening operator是先腐蚀后膨胀，可以消除一些细小的边界，消除噪声
    kernel=np.ones((3,3),np.uint8)
    opening=cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=3)
    # plt.imshow(opening,cmap='gray')
    # plt.show()


    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=1)
    # plt.imshow(sure_bg,cmap='gray')
    # plt.show()

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    # plt.imshow(img,cmap='gray')
    # plt.show()
    return img

if __name__ == "__main__":
    data_path = '/media/data1/xyx/dataset/medical_test_images/'
    filenames = os.listdir(data_path)
    print('./inputs/test%d.jpg')
    for i, img in enumerate(filenames):
        imgs = cv.imread(os.path.join(data_path, img))
        img_path = os.path.join(data_path, filenames[i])
        res = watershedSegment(img_path)
        # plt.imshow(res, cmap='gray')
        # plt.show()
        cv.imwrite('./inputs/test%d.jpg' % (i), imgs)
        cv.imwrite('./outputs/res%d.jpg' % (i), res)




