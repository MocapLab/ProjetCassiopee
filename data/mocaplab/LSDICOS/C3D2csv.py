import glob
from ezc3d import c3d
import csv
import sys
import os
import numpy as np
glob_dir = os.path.dirname(os.path.realpath(__file__))

list_c3d = glob.glob(r"D:\Mcl_FTP\prod\MCL_LSDICO3D\40_*\00_*\*\*\*\*.c3d")
if not list_c3d:
    print('no c3d found')
other_c3d = glob.glob(r"D:\Mcl_FTP\prod\MCL_LSDICOLSF_T02\40_*\00_*\*\*\*\*.c3d")
if not other_c3d:
    print('no c3d found')
else:
    list_c3d.extend(other_c3d)

for c3d_path in list_c3d:
    filename = os.path.join(glob_dir, os.path.basename(c3d_path).replace('.c3d','.csv'))
    if os.path.exists(filename):
        print(f"{filename} already exists")
        continue
    print(c3d_path)
    h = c3d(c3d_path)
    list_mrk = h['parameters']['POINT']['LABELS']['value']
    subsample = 4
    points = h['data']['points'][:3,:,::subsample]
    if len(list_mrk)>0:
        points_exp = points.transpose(2,1,0).reshape(points.shape[2],-1)
        
        with open(filename, 'w', encoding='UTF8', newline="") as csvfile:
        
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar=' ')
            rows = []
            writer.writerow([";".join([";".join((i,i,i)) for i in list_mrk])])
            writer.writerow([";".join([";".join(["X","Y","Z"]) for i in list_mrk])])
        
            np.savetxt(csvfile, points_exp, fmt='%.3f', delimiter=";")
    