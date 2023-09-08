import bpy
import numpy as np
import json

beamline = 'ANATOMIX_SOLEIL'
voxel_size = '650nm'

name = beamline + '_' + voxel_size
p = 'J:\\Jonas Hemesath\\Synchrotron_time_calculations\\220112\\' + name + '\\'

volume = np.load('C:\\Users\\hemesath\\Desktop\\' + name + '.npy')

with open(p + name + '.json', 'r') as f:
    data = json.load(f)

volume = np.load(p + name + '.npy')

#for x in range(1):
for x in range(volume.shape[1]):
    x1 = np.mean([x * data[0] * data[2], (x+1) * data[0] * data[2]])

    for y in range(volume.shape[2]):
        y1 = np.mean([y * data[0] * data[2], (y+1) * data[0] * data[2]])

        for z in range(volume.shape[0]):
            z1 = np.mean([z * data[1] * data[2], (z+1) * data[1] * data[2]])

            if volume[z, x, y] > 0:
                bpy.ops.mesh.primitive_cylinder_add()
                cylinder = bpy.context.selected_objects[0]
                cylinder.location = (x1, y1, z1)
                cylinder.scale = (data[3]/2, data[3]/2, data[4]/2)