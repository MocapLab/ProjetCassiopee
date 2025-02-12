import glob
from ezc3d import c3d
import csv
# import sys
from itertools import chain
import os
import numpy as np
glob_dir = os.path.dirname(os.path.realpath(__file__))

list_c3d = glob.glob(os.path.join(glob_dir,"*","*.c3d"))
if not list_c3d:
    print('no c3d found')

# other_c3d = glob.glob(r"\\Jiraya\prod\MCL_LSDICOLSF_T02\40_*\00_*\*\*\*\*.c3d")
# if not other_c3d:
#     print('no c3d found')
# else:
#     list_c3d.extend(other_c3d)
# other_c3d = glob.glob(r"\\Jiraya\prod\MCL_LSDICO3D\40_*\00_*\*\*\*\*.c3d")
# if not other_c3d:
#     print('no c3d found')
# else:
#     list_c3d.extend(other_c3d)
sub_sample_points = ['RUPA', 'LFWT', 'LELB', 'UPHD', 'RELBEXT', 'RCLAV', 'LBSHO', 'LELBEXT', 'RFWT', 'LCLAV', 'LBWT', 'LFSHO', 'CLAV', 'C7', 'RFHD', 'RBAC', 'RBSHO', 'RBWT', 'T10', 'LFRM', 'RELB', 'LBAC', 'LBHD', 'RBHD', 'LFHD', 'STRN', 'RFRM', 'RFSHO', 'LSHOULD', 'LUPA', 'RSHOULD']
all_missing = []
for c3d_path in list_c3d:
    filename = os.path.join(glob_dir, os.path.basename(c3d_path).replace('.c3d','.csv'))
    filename = filename.replace(".Noemie","")
    if os.path.exists(filename):
        print(f"{filename} already exists")
        continue
    print(c3d_path)
    h = c3d(c3d_path)
    list_mrk = h['parameters']['POINT']['LABELS']['value']
    missing = [i for i in sub_sample_points if i not in list_mrk]
    if len(missing)>0:
        print(f"missing {missing}")
        all_missing.append([filename, missing])
        continue
    ids_pts = [[i[0]*3,i[0]*3+1,i[0]*3+2] for i in enumerate(list_mrk) if i[1] in sub_sample_points]
    ids_mrk = [i[0]*3 for i in enumerate(list_mrk) if i[1] in sub_sample_points]
    sub_list_mrk = [list_mrk[int(i/3)] for i in ids_mrk]
    ids_pts = list(chain.from_iterable(ids_pts))
    subsample = 4
    points = h['data']['points'][:3,:,::subsample]
    if len(list_mrk)>0:
        points_exp = points.transpose(2,1,0).reshape(points.shape[2],-1)
        points_exp = points_exp[:,ids_pts]
        with open(filename, 'w', encoding='UTF8', newline="") as csvfile:
        
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar=' ')
            rows = []
            writer.writerow([";".join([";".join((i,i,i)) for i in sub_list_mrk])])
            writer.writerow([";".join([";".join(["X","Y","Z"]) for i in sub_list_mrk])])
        
            np.savetxt(csvfile, points_exp, fmt='%.3f', delimiter=";")
    
print(all_missing)