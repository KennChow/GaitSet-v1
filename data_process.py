import os
import MOG_video_batch as mog
import pretreatment
from multiprocessing import Pool

# 轮廓提取
INPUT_PATH = "data_video"
id_list = os.listdir(INPUT_PATH)
id_list.sort()
# print(id_list)
pool = Pool(processes=3)
# names = ["the great " + name for name in names]
id_list = [INPUT_PATH + '/' + i for i in id_list]
# print(id_list)
for i in id_list:
    # print(os.path.join(INPUT_PATH, i))  # data_video/125-nm-04-270.mp4

    # path = os.path.join(INPUT_PATH, i)
    # print(path)
    # mog.MOG(path)
    pool.apply_async(mog.MOG, args=(i,))
pool.close()
# # 裁剪
# pretreatment.pretreatment()
