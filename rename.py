import os
path = '/home/zc/GaitSet/data_probe/xxx/nm-01/072'
for file in os.listdir(path):
    # 130-nm-01-072-040.png
    os.rename(os.path.join(path, file), os.path.join(path, 'XXX'+file[3:]))
    print(file[3:])
