import bpy
import json

path = 'C:\\Users\\jonas\\python_experiments\\CT_positions\\'



with open(path + 'blender_data.json', 'r') as f:
    data = json.load(f)

positions = data['column_positions']

bpy.ops.import_mesh.stl(filepath=path + "overview.stl")
brain = bpy.context.selected_objects[0]
brain.location = (data['mesh_origin'][1], data['mesh_origin'][0], data['mesh_origin'][2])

for p in positions:

    bpy.ops.mesh.primitive_cylinder_add()
    cylinder = bpy.context.selected_objects[0]
    cylinder.location = (p[1], p[0], p[2])
    cylinder.scale = (data['tilesize'][0]/2, data['tilesize'][0]/2, data['tilesize'][1]/2)