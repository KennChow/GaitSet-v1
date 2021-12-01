import numpy as np
import cv2
from collections import Counter
import os


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False



def MOG(path):  # data_video/125-nm-04-270.mp4
    root_path = path.split('/')[0]  # data_video
    video_name = path.split('/')[1]  # 125-nm-04-270.mp4
    cap = cv2.VideoCapture(path)
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg = cv2.createBackgroundSubtractorKNN()
    cnt = 1
    while(1):
        # ret1, frame1 = cap1.read()
        ret, frame = cap.read()
        
        if frame is None:
            break
        fgmask = fgbg.apply(frame)
        cnt = cnt + 1
        if cnt <= 60:
            continue

        # 去噪点
        img = cv2.medianBlur(fgmask, 3)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # video_name:125-nm-04-270.mp4
        v_lable = video_name[0:3]  # 125
        v_seq_type = video_name[4:9]  # nm-01
        v_view = video_name[10:13]  # 072
        # v_data = 'data/' + v_lable + '/' + v_seq_type + '/' + v_view + '/'  # data/125/nm-01/072/
        out_path = 'data_frame/' + v_lable + '/' + v_seq_type + '/' + v_view + '/'  # out/125/nm-01/072/
        mkdir(out_path)
        cv2.imwrite(out_path + str(cnt) + '.jpg', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


# if __name__ == '__main__':
#     f = open("source.txt", "r")
#     lines = f.readlines()
#     # print(lines)
#     for line in lines:
#         line = line.strip('\n')  # 125-nm-01-072.mp4
#         print(line)
#         MOG(line)
