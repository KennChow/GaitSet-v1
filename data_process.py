import os
import MOG_video_batch as mog
import pretreatment
from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError
from time import sleep
import multiprocessing
import time
from model.utils.remove_dirs import remove_dirs
# 轮廓提取
remove_dirs('/workspace/projects/GaitSet/data_frame/')
INPUT_PATH = "data_video"
id_list = os.listdir(INPUT_PATH)
print(len(id_list))
id_list.sort()
# print(id_list)
# num_workers = multiprocessing.cpu_count()
# pool = Pool(processes=1)
# print(num_workers)
# names = ["the great " + name for name in names]
id_list = [INPUT_PATH + '/' + i for i in id_list]
print(id_list)
# print(id_list)
results = list()

t1 = time.time()
tt1 = time.perf_counter()

# 不使用多进程
for i in id_list:
    mog.MOG(i)

# 使用多进程
# for i in id_list:
#     pool.apply_async(mog.MOG, args=(i,))
#     results.append(pool.apply_async(mog.MOG, args=(i,)))
#     sleep(0.02)
# pool.close()
# unfinish = 1
# while unfinish > 0:
#     unfinish = 0
#     for i, res in enumerate(results):
#         try:
#             res.get(timeout=0.1)
#         except Exception as e:
#             if type(e) == MP_TimeoutError:
#                 unfinish += 1
#                 continue
#             else:
#                 print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
#                       i, type(e))
#                 raise e
# pool.join()


# 裁剪
remove_dirs('/workspace/projects/GaitSet/data_pretreated/')
pretreatment.pretreatment()

t2 = time.time()
tt2 = time.perf_counter()
print("total_time:" + str(t2 - t1))
print("cpu_time:" + str(tt2 - tt1))
