import os
currentpath = os.path.join(os.getcwd(), 'image_path_v2')
print('currentpath', currentpath)
pathfull = os.path.join(currentpath, 'saved_model.pb')
print('pathfull', pathfull)
if os.path.exists(currentpath):
    print('!!!!!!!!! if the path exists')