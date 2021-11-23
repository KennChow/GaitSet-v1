import os
import MOG_video_batch as mog

# 轮廓提取
INPUT_PATH = "data_video"
id_list = os.listdir(INPUT_PATH)
id_list.sort()
# print(id_list)
for i in id_list:
    # print(os.path.join(INPUT_PATH, i))  # data_video/125-nm-04-270.mp4
    path = os.path.join(INPUT_PATH, i)
    print(path)
    mog.MOG(path)

# 裁剪
#
