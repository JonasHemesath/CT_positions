# Script to calculate positions for tiled CT scan. Jonas Hemesath, Max-Planck-Institute for biological Intelligence, 2023

import tifffile
from pathlib import Path
import numpy as np
import math
import trimesh
from skimage import measure
import json
import matplotlib.pyplot as plt
import copy
from scipy.ndimage import convolve


# Parameters
fp_overview = 'D:\\ESRF\\ZF13\\zf13_bin'        # images have to be binary
fp_resin_overview = None
overview_pos = (-0.575*1000, -1.3*1000, -95*1000)            # x, y, z in micrometer
overview_voxel_size = 4*6        # in micrometer

fp_output = 'C:\\Users\\jonas\\python_experiments\\CT_positions\\'

detector_px_x = 2560
detector_px_y = 1990

half_aquisition = True
half_acquistion_overlap = 0.9

hres_voxel_size = 0.7               # in micrometer

time_per_tomogram = 6              # in minutes

overlap_between_tomograms = 0.31    # fraction
overlap_column = 100                # tile voxel

tomogram_overlap = 'offset'         # equals to either 'linear' or 'offset'

export_brain_mesh = True
export_resin_mesh = False
downsample = 2

origin_offset = 5                   # mm from the lower left corner of the overview

show_snippets = True

invert = True                       # Invert the bw image


def get_columns(pos, tile_z_um, overlap_z, hres_voxel_size):
    """Generate equal sized columns fit to the xy plane of the object

    Args:
        pos (list): List of tomogram positions that fit the object in all dimensions
        tile_z_um (float): Size in z of one tile in micrometer
        overlap_z (int): Number of voxels overlap in z-direction
        hres_voxel_size (float): voxel size of the high resolution scan

    Returns:
        list: list of [x, y, z] positions
    """

    overlap_z_um = overlap_z * hres_voxel_size
    xy = []
    for p in pos:
        if [p[0], p[1]] not in xy:
            xy.append([p[0], p[1]])
    z = list(set([p[2] for p in pos]))
    z_max = np.max(z)
    z_min = np.min(z)
    n_z = round((z_max - z_min)/((tile_z_um-overlap_z_um) * 0.001)) + 1
    blender_positions = []
    for p in xy:
        for i in range(n_z):
            blender_positions.append([p[0], p[1], z_max - i * (tile_z_um-overlap_z_um) * 0.001])
    with open('column_positions.txt', 'w') as f:
        line = 'z_max = ' + str(z_max) + ', number of tomograms in one column = ' + str(n_z) + '\n\n'
        f.write(line)
        for p in xy:
            line = 'uvm(sx, ' + str(p[0]) + ', sy, ' + str(p[1]) + ')\n'
            f.write(line)

    return blender_positions


def get_grid(pos , tile_z_um, tomogram_overlap, overlap_z, hres_voxel_size):
    """Generate a grid of tomograms fit to the xy plane of the object

    Args:
        pos (list): List of tomogram positions that fit the object in all dimensions
        tile_z_um (float): Size in z of one tile in micrometer
        tomogram_overlap (str): Type of overlap (either 'linear' or 'grid')
        overlap_z (int): Number of voxels overlap in z-direction
        hres_voxel_size (float): voxel size of the high resolution scan

    Returns:
        list: list of [x, y, z] positions
    """
    overlap_z_um = overlap_z * hres_voxel_size
    xs = list(set([p[0] for p in pos]))
    ys = list(set([p[1] for p in pos]))
    ys1 = [ys[i] for i in range(0, len(ys), 2)]
    ys2 = [ys[i] for i in range(1, len(ys), 2)]
    ys.sort()

    zs = list(set([p[2] for p in pos]))

    xy = []

    if tomogram_overlap == 'linear':
        for x in xs:
            for y in ys:
                xy.append([x, y])

    elif tomogram_overlap == 'offset':
        for i, x in enumerate(xs):
            if i%2 == 0:
                for y in ys1:
                     xy.append([x, y])
            else:
                for y in ys2:
                     xy.append([x, y])


    z_min = np.min(zs)
    z_max = np.max(zs)

    n_z = round((z_max - z_min)/((tile_z_um-overlap_z_um) * 0.001)) + 1

    blender_positions = []
    for p in xy:
        for i in range(n_z):
            blender_positions.append([p[0], p[1], z_max - i * (tile_z_um-overlap_z_um) * 0.001])
    with open('grid_positions.txt', 'w') as f:
        line = 'z_max = ' + str(z_max) + ', number of tomograms in one column = ' + str(n_z) + '\n\n'
        f.write(line)
        for p in xy:
            line = 'uvm(sx, ' + str(p[0]) + ', sy, ' + str(p[1]) + ')\n'
            f.write(line)

    return blender_positions

    


def calculate_postions(fp_overview, overview_pos, overview_voxel_size, fp_output, detector_px_x, detector_px_y, half_aquisition, half_acquistion_overlap,
                       hres_voxel_size, time_per_tomogram, overlap_between_tomograms, overlap_column, tomogram_overlap, origin_offset, show_snippets, invert):
    """_summary_

    Args:
        fp_overview (str): filepath to an overview scan of the object
        overview_pos (_type_): _description_
        overview_voxel_size (_type_): _description_
        fp_output (_type_): _description_
        detector_px_x (_type_): _description_
        detector_px_y (_type_): _description_
        half_aquisition (_type_): _description_
        half_acquistion_overlap (_type_): _description_
        hres_voxel_size (_type_): _description_
        time_per_tomogram (_type_): _description_
        overlap_between_tomograms (_type_): _description_
        overlap_column (_type_): _description_
        tomogram_overlap (_type_): _description_
        origin_offset (_type_): _description_
        show_snippets (_type_): _description_
        invert (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Overview
    overview_paths = list(Path(fp_overview).iterdir())
    overview = tifffile.TiffSequence(overview_paths).asarray()

    overview = np.rot90(overview, 1, (0,1))
    overview = np.rot90(overview, 1, (1,2))
    overview = np.rot90(overview, 2, (0,1))
    overview = np.fliplr(overview)


    if invert:
        overview_new = copy.deepcopy(overview)

        overview = np.zeros(overview_new.shape)
        overview[overview_new==0] = 255


    if show_snippets:
        snippets = [overview.shape[2]//4, overview.shape[2]//2, (3*overview.shape[2])//4]
        for s in snippets:
            plt.imshow(overview[:,:,s])
            plt.show()
    
    overview_size_xy_um = overview_voxel_size * overview.shape[0]
    overview_size_z_um = overview_voxel_size * overview.shape[2]

    top_left = (overview_pos[0]+overview_size_xy_um/2, overview_pos[1]-overview_size_xy_um/2, overview_pos[2]-overview_size_z_um/2)
    lower_left = ((top_left[0]-overview_size_xy_um)*0.001, top_left[1]*0.001, top_left[2]*0.001)

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
                                        top_left[2]*0.001 + np.mean([z3,z1])*overview_voxel_size*0.001))
                        
    
    elif tomogram_overlap == 'offset':
        x2 = 0
        y2 = 0
        z2 = 0
        x = 0
        y = 0
        z = 0
        while x2 != -1:
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
            x += 1
                
            while y2 != -1:
                y1 = y * ts_xy - y_offset
                y2 = (y+1) * ts_xy - y_offset
                y3 = y1
                y4 = y2
                if y1 < 0:
                    y1 = 0
                if y2 > overview.shape[0]:
                    y2 = -1
                y += 1
                
                while z2 != -1:
                    z1 = z * ts_z
                    z2 = (z+1) * ts_z
                    z3 = z1
                    z4 = z2
                    if z2 > overview.shape[0]:
                        z2 = -1
                    z += 1

                    if np.any(overview[x1:x2, y1:y2, z1:z2] > 0):
                        positions.append((top_left[0]*0.001 - np.mean([x4,x3])*overview_voxel_size*0.001, 
                                        top_left[1]*0.001 + np.mean([y4,y3])*overview_voxel_size*0.001,
                                        top_left[2]*0.001 + np.mean([z4,z3])*overview_voxel_size*0.001))
                z=0
                z2=0
            y=0
            y2=0


    # Exports
    with open(fp_output + 'postions.txt', 'w') as f:
        f.write('[')
        for p in positions:
            line = '(' + str(p[0]) + ', ' + str(p[1]) + ', ' + str(p[2]) + '),\n'
            f.write(line)
        f.write(']')

    column_positions = get_columns(positions, tile_size_z_um, overlap_column, hres_voxel_size)
    grid_positions = get_grid(positions, tile_size_z_um, tomogram_overlap, overlap_column, hres_voxel_size)

    blender_data = {'mesh_origin': lower_left, 'positions': positions, 'tilesize': (tile_size_xy_um*0.001, tile_size_z_um*0.001), 'column_positions': column_positions, 'grid_positions': grid_positions}

    with open('blender_data.json', 'w') as f:
        json.dump(blender_data, f)


    # Print time estimates
    print('Optimal fit:')
    print('Number of tomograms:', len(positions))
    print('Total scan time [hrs]:', len(positions)*time_per_tomogram/60)
    print('\n')
    print('Columns:')
    print('Number of tomograms:', len(column_positions))
    print('Total scan time [hrs]:', len(column_positions)*time_per_tomogram/60)
    print('\n')
    print('Grid:')
    print('Number of tomograms:', len(grid_positions))
    print('Total scan time [hrs]:', len(grid_positions)*time_per_tomogram/60)

                    
    return positions


def output_resin_mesh(fp_resin_overview, fp_output, overview_voxel_size, downsample=None):
    overview_paths = list(Path(fp_resin_overview).iterdir())
    overview = tifffile.TiffSequence(overview_paths).asarray()

    overview = np.rot90(overview, 1, (0,1))
    overview = np.rot90(overview, 1, (1,2))
    overview = np.rot90(overview, 3, (0,1))
    overview = np.fliplr(overview)
    #overview = np.flip(overview, 2)
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
    surf_mesh.export(fp_output + 'resin_overview.stl')

def output_brain_mesh(fp_overview, fp_output, overview_voxel_size, downsample=None):
    overview_paths = list(Path(fp_overview).iterdir())
    overview = tifffile.TiffSequence(overview_paths).asarray()

    overview = np.rot90(overview, 1, (0,1))
    overview = np.rot90(overview, 1, (1,2))
    overview = np.rot90(overview, 3, (0,1))
    overview = np.fliplr(overview)
    #overview = np.flip(overview, 2)
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
                       hres_voxel_size, time_per_tomogram, overlap_between_tomograms, overlap_column, tomogram_overlap, origin_offset, show_snippets, invert)
    
    if export_brain_mesh:
        output_brain_mesh(fp_overview, fp_output, overview_voxel_size, downsample)

    if export_resin_mesh:
        output_resin_mesh(fp_resin_overview, fp_output, overview_voxel_size, downsample)

    
    





