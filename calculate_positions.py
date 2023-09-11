# Script to calculate positions for tiled CT scan. Jonas Hemesath, Max-Planck-Institute for biological Intelligence, 2023

import tifffile
from pathlib import Path
import numpy as np
import math
import trimesh
from skimage import measure
import json


# Parameters
fp_overview = 'J:\\Jonas Hemesath\\Synchrotron_time_calculations\\220112\\bw_images'        # images have to be binary
overview_pos = (0, 0, 0)            # x, y, z in micrometer
overview_voxel_size = 24.41         # in micrometer

fp_output = 'C:\\Users\\hemesath\\python_experiments\\CT_positions\\'

detector_px_x = 2560
detector_px_y = 2560

half_aquisition = True
half_acquistion_overlap = 0.9

hres_voxel_size = 0.7               # in micrometer

time_per_tomogram = 12              # in minutes

overlap_between_tomograms = 0.31    # fraction
overlap_column = 100                # tile voxel

tomogram_overlap = 'offset'         # equals to either 'linear' or 'offset'

export_brain_mesh = True
downsample = 2

origin_offset = 5                   # mm from the lower left corner of the overview


def calculate_postions(fp_overview, overview_pos, overview_voxel_size, fp_output, detector_px_x, detector_px_y, half_aquisition, half_acquistion_overlap,
                       hres_voxel_size, time_per_tomogram, overlap_between_tomograms, overlap_column, tomogram_overlap, origin_offset):
    
    # Overview
    overview_paths = list(Path(fp_overview).iterdir())
    overview = tifffile.TiffSequence(overview_paths).asarray()

    overview = np.rot90(overview, 1, (0,1))
    overview = np.rot90(overview, 1, (1,2))
    overview = np.rot90(overview, 2, (0,1))
    
    overview_size_xy_um = overview_voxel_size * overview.shape[0]
    overview_size_z_um = overview_voxel_size * overview.shape[2]

    top_left = (overview_pos[0]+overview_size_xy_um/2, overview_pos[1]-overview_size_xy_um/2, overview_pos[2]+overview_size_z_um/2)
    lower_left = ((top_left[0]-overview_size_xy_um)*0.001, top_left[1]*0.001, (top_left[2]-overview_size_z_um)*0.001)

    # Tile 
    if half_aquisition:
        tile_size_xy_um = (detector_px_x + detector_px_x * half_acquistion_overlap) * hres_voxel_size
        tile_size_z_um = detector_px_y * hres_voxel_size
    else:
        tile_size_xy_um = detector_px_x * hres_voxel_size
        tile_size_z_um = detector_px_y * hres_voxel_size
    
    ts_overview_xy = overview.shape[0]/overview_size_xy_um * tile_size_xy_um
    ts_overview_z = overview.shape[2]/overview_size_z_um * tile_size_z_um

    ts_xy = math.floor(ts_overview_xy * (1-overlap_between_tomograms))
    ts_z = math.floor(ts_overview_z - overlap_column * (hres_voxel_size/overview_voxel_size))

    # Tiling
    i_x = math.ceil(overview.shape[0]/ts_xy)
    i_y = math.ceil(overview.shape[1]/ts_xy)
    i_z = math.ceil(overview.shape[2]/ts_z)

    positions = []

    if tomogram_overlap == 'linear':
        for x in range(i_x):
            x1 = x * ts_xy
            x2 = (x+1) * ts_xy
            x3 = x2
            if x2 > overview.shape[0]:
                x2 = -1
                
            for y in range(i_y):
                y1 = y * ts_xy
                y2 = (y+1) * ts_xy 
                y3 = y2
                if y2 > overview.shape[0]:
                    y2 = -1
                
                for z in range(i_z):
                    z1 = z * ts_z
                    z2 = (z+1) * ts_z
                    z3 = z2
                    if z2 > overview.shape[0]:
                        z2 = -1

                    if np.any(overview[x1:x2, y1:y2, z1:z2] > 0):
                        positions.append((top_left[0]*0.001 - np.mean([x3,x1])*overview_voxel_size*0.001, 
                                        top_left[1]*0.001 + np.mean([y3,y1])*overview_voxel_size*0.001,
                                        top_left[2]*0.001 - np.mean([z3,z1])*overview_voxel_size*0.001))
                        
    elif tomogram_overlap == 'offset':
        for x in range(i_x):
            x1 = x * ts_xy
            x2 = (x+1) * ts_xy
            x3 = x1
            x4 = x2
            if x2 > overview.shape[0]:
                x2 = -1

            if x%2 == 1:
                i_y += 1
                y_offset = ts_xy//2
            else:
                i_y -= 1
                y_offset = 0
                
            for y in range(i_y):
                y1 = y * ts_xy - y_offset
                y2 = (y+1) * ts_xy - y_offset
                y3 = y1
                y4 = y2
                if y1 < 0:
                    y1 = 0
                if y2 > overview.shape[0]:
                    y2 = -1
                
                for z in range(i_z):
                    z1 = z * ts_z
                    z2 = (z+1) * ts_z
                    z3 = z1
                    z4 = z2
                    if z2 > overview.shape[0]:
                        z2 = -1

                    if np.any(overview[x1:x2, y1:y2, z1:z2] > 0):
                        positions.append((top_left[0]*0.001 - np.mean([x4,x3])*overview_voxel_size*0.001, 
                                        top_left[1]*0.001 + np.mean([y4,y3])*overview_voxel_size*0.001,
                                        top_left[2]*0.001 - np.mean([z4,z3])*overview_voxel_size*0.001))

    # Determine origin and normalize to origin
    origin = [lower_left[i]-origin_offset for i in range(3)]

    mesh_origin = [lower_left[i] - origin[i] for i in range(3)]

    norm_positions = [[p[i] - origin[i] for i in range(3)] for p in positions]

    # Exports

    print('Number of tomograms:', len(positions))
    print('Total scan time [hrs]:', len(positions)*time_per_tomogram/60)

    with open(fp_output + 'postions.txt', 'w') as f:
        for p in positions:
            line = 'uvm(sx, ' + str(p[0]-origin[0]) + ', sy, ' + str(p[1]-origin[1]) + ', sz, ' + str(p[2]-origin[2]) + ')\n'
            f.write(line)


    blender_data = {'origin': origin, 'mesh_origin': mesh_origin, 'positions': norm_positions, 'tilesize': (tile_size_xy_um*0.001, tile_size_z_um*0.001)}

    with open('blender_data.json', 'w') as f:
        json.dump(blender_data, f)

                    
    return positions


def output_brain_mesh(fp_overview, fp_output, overview_voxel_size, downsample=None):
    overview_paths = list(Path(fp_overview).iterdir())
    overview = tifffile.TiffSequence(overview_paths).asarray()

    overview = np.rot90(overview, 1, (0,1))
    overview = np.rot90(overview, 1, (1,2))
    overview = np.rot90(overview, 1, (0,1))
    overview = np.flip(overview, 2)
    print('Overview loaded')
    vx = overview_voxel_size * 0.001
    if downsample:
        overview = measure.block_reduce(overview, downsample)
        vx = vx * downsample
        print('Image downsampled by factor of', downsample)

    verts, faces, normals, values = measure.marching_cubes(overview, 0, spacing=(vx, vx, vx))
    print('Marching cubes finished')
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    print('Mesh calculated')
    surf_mesh.export(fp_output + 'overview.stl')
    




    


if __name__ == '__main__':
    positions = calculate_postions(fp_overview, overview_pos, overview_voxel_size, fp_output, detector_px_x, detector_px_y, half_aquisition, half_acquistion_overlap,
                       hres_voxel_size, time_per_tomogram, overlap_between_tomograms, overlap_column, tomogram_overlap, origin_offset)
    
    if export_brain_mesh:
        output_brain_mesh(fp_overview, fp_output, overview_voxel_size, downsample)
    
    





