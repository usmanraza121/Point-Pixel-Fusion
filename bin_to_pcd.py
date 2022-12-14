import open3d as o3d
import numpy as np
import struct
import glob
size_float = 4


file_to_open = sorted(glob.glob("/home/usman/kitti_track/data_tracking_velodyne/training/velodyne/0000/*.bin"))


for fi in file_to_open:
	list_pcd = []
	with open (fi, "rb") as f:
	    byte = f.read(size_float*4)
	    while byte:
	        x,y,z,intensity = struct.unpack("ffff", byte)
	        list_pcd.append([x, y, z])
	        byte = f.read(size_float*4)
	np_pcd = np.asarray(list_pcd)
	pcd = o3d.geometry.PointCloud()
	v3d = o3d.utility.Vector3dVector
	pcd.points = v3d(np_pcd)
	o3d.io.write_point_cloud(fi[:-3]+"pcd", pcd)

