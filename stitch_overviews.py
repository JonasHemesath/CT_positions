import tifffile
import os
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt

path = 'D:\ESRF\ZF13\zf13_slices\z-971720'

voxel_size = 0.72
downsample = 5
div = voxel_size * downsample

for p in os.listdir(path):
    img = tifffile.imread(path + '\\' + p)
    img = measure.block_reduce(img, downsample)
    tifffile.imwrite(path + '_n\\' + p, img)



names =os.listdir(path + '_n\\')
min_x = 10000000
min_y = 10000000
pos = []
count = 0
for name in names:
    if name[0:3] != 'set':
        count+=1
        spl = name.split('_')
        #print(spl)
        x = int(spl[2][1:-1])
        if x < min_x:
            min_x = x
        y = int(spl[3][1:-1])
        if y < min_y:
            min_y = y
        #print(count)
        pos.append([x, y])


norm_pos = [[int((p[0] - min_x)/div),int((p[1] - min_y)/div)] for p in pos]
max_x = 0
max_y = 0
for p in norm_pos:
    if p[0] > max_x:
        max_x = p[0]
    if p[1] > max_y:
        max_y = p[1]

max_x += 1000
max_y += 1000

mask = np.zeros(img.shape)
for x in range(mask.shape[0]//2):
    y1= mask.shape[0]//2 - int(np.sqrt((mask.shape[0]//2)**2-x**2))
    y2= mask.shape[0]//2 -1 + int(np.sqrt((mask.shape[0]//2)**2-x**2))
    mask[(mask.shape[0]//2-1)+x, y1:y2] = 1
    mask[(mask.shape[0]//2)-x, y1:y2] = 1

final = np.zeros((max_x, max_y))


for i, p in enumerate(os.listdir(path)):
    img = tifffile.TiffFile(path + '_n\\' + p).asarray()
    #print(img.shape)

    mask = np.zeros(img.shape)
    for x in range(mask.shape[0]//2):
        y1= mask.shape[0]//2 - int(np.sqrt((mask.shape[0]//2)**2-x**2))
        y2= mask.shape[0]//2 -1 + int(np.sqrt((mask.shape[0]//2)**2-x**2))
        mask[(mask.shape[0]//2-1)+x, y1:y2] = 1
        mask[(mask.shape[0]//2)-x, y1:y2] = 1

    x1 = max_x - norm_pos[i][0]
    x2 = x1 - img.shape[0]
    y1 = max_y - norm_pos[i][1]
    y2 = y1 - img.shape[1]

    final[x2:x1, y2:y1][mask>0] = img[mask>0]

tifffile.imwrite(path + '_n\\' + 'overview.tif', final)

plt.imshow(final)
plt.show()

    


