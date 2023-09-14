import bpy

path = 'C:\\Users\\jonas\\python_experiments\\CT_positions\\'
fname = 'blender_positions.txt'

scene = bpy.context.scene
with open(path+fname, 'w') as f:
    f.write('[')
    for obj in scene.objects:
        line = '(' + str(obj.location[1]) + ', ' + str(obj.location[0]) + ', ' + str(obj.location[2]) + '),\n'
        f.writelines(str(obj.location))
    f.write(']')