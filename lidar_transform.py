import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image
import glob
import open3d as o3d
import yolov4

import statistics
import random

from yolov4.tf import YOLOv4 
import tensorflow as tf
import time

yolo = YOLOv4(tiny=True)
yolo.classes = "/home/usman/visual_fusion/Yolov4/coco.names"
yolo.make_model()
yolo.load_weights("/home/usman/visual_fusion/Yolov4/yolov4-tiny.weights", weights_type="yolo")

def run_obstacle_detection(img):
    start_time=time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = yolo.resize_image(img)
   
    resized_image = resized_image / 255.
    
    input_data = resized_image[np.newaxis, ...].astype(np.float32)

    candidates = yolo.model.predict(input_data)

    _candidates = []
    result = img.copy()
    for candidate in candidates:
        batch_size = candidate.shape[0]
        grid_size = candidate.shape[1]
        _candidates.append(tf.reshape(candidate, shape=(1, grid_size * grid_size * 3, -1)))
      
        candidates = np.concatenate(_candidates, axis=1)
    
        pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0], iou_threshold=0.35, score_threshold=0.40)
        pred_bboxes = pred_bboxes[~(pred_bboxes==0).all(1)] 
        pred_bboxes = yolo.fit_pred_bboxes_to_original(pred_bboxes, img.shape)
        exec_time = time.time() - start_time
    
        result = yolo.draw_bboxes(img, pred_bboxes)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result, pred_bboxes

# result, pred_bboxes = run_obstacle_detection(image)

class LiDAR2Camera(object):
    def __init__(self, calib_file):
        calibs = self.read_calib_file(calib_file)
        P = calibs["P2"]
        self.P = np.reshape(P, [3, 4])

  
        V2C = calibs["Tr_velo_cam"]
        self.V2C = np.reshape(V2C, [3, 4])
      

        R0 = calibs["R_rect"]
        self.R0 = np.reshape(R0, [3, 3])

    def read_calib_file(self, filepath):
 
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
     
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

def cart2hom(self, pts_3d):
    """ Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

LiDAR2Camera.cart2hom = cart2hom

def project_velo_to_image(self, pts_3d_velo):

    R0_homo = np.vstack([self.R0, [0, 0, 0]])
    R0_homo_2 = np.column_stack([R0_homo, [0, 0, 0, 1]])
    p_r0 = np.dot(self.P, R0_homo_2) #PxR0
    p_r0_rt =  np.dot(p_r0, np.vstack((self.V2C, [0, 0, 0, 1]))) #PxROxRT
    pts_3d_homo = np.column_stack([pts_3d_velo, np.ones((pts_3d_velo.shape[0],1))])
    p_r0_rt_x = np.dot(p_r0_rt, np.transpose(pts_3d_homo))#PxROxRTxX
    pts_2d = np.transpose(p_r0_rt_x)
    
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

LiDAR2Camera.project_velo_to_image = project_velo_to_image

def get_lidar_in_image_fov(self,pc_velo, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0):
  
    pts_2d = self.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance) 
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo
    
LiDAR2Camera.get_lidar_in_image_fov = get_lidar_in_image_fov

def show_lidar_on_image(self, pc_velo, img, debug="False"):
   
    imgfov_pc_velo, pts_2d, fov_inds = self.get_lidar_in_image_fov(
        pc_velo, 0, 0, img.shape[1], img.shape[0], True
    )
    if (debug==True):
        print("3D PC Velo "+ str(imgfov_pc_velo))
        print("2D PIXEL: " + str(pts_2d)) 
        print("FOV : "+str(fov_inds)) 
    self.imgfov_pts_2d = pts_2d[fov_inds, :]
    '''
  
    homogeneous = self.cart2hom(imgfov_pc_velo)
    transposed_RT = np.dot(homogeneous, np.transpose(self.V2C))
    dotted_RO = np.transpose(np.dot(self.R0, np.transpose(transposed_RT)))
    self.imgfov_pc_rect = dotted_RO
    
    if debug==True:
        print("FOV PC Rect "+ str(self.imgfov_pc_rect))
    '''
    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    self.imgfov_pc_velo = imgfov_pc_velo
    
    for i in range(self.imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_velo[i,0]
        color = cmap[int(510.0 / depth), :]
        cv2.circle(
            img,(int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))),2,
            color=tuple(color),
            thickness=-1,)
    return img

LiDAR2Camera.show_lidar_on_image = show_lidar_on_image

def rectContains(rect,pt, w, h, shrink_factor = 0):       
    x1 = int(rect[0]*w - rect[2]*w*0.5*(1-shrink_factor)) # center_x - width /2 * shrink_factor
    y1 = int(rect[1]*h-rect[3]*h*0.5*(1-shrink_factor)) # center_y - height /2 * shrink_factor
    x2 = int(rect[0]*w + rect[2]*w*0.5*(1-shrink_factor)) # center_x + width/2 * shrink_factor
    y2 = int(rect[1]*h+rect[3]*h*0.5*(1-shrink_factor)) # center_y + height/2 * shrink_factor
    
    return x1 < pt[0]<x2 and y1 <pt[1]<y2 

def filter_outliers(distances):
    inliers = []
    mu  = statistics.mean(distances)
    std = statistics.stdev(distances)
    for x in distances:
        if abs(x-mu) < std:
            inliers.append(x)
    return inliers

def get_best_distance(distances, technique="closest"):
    if technique == "closest":
        return min(distances)
    elif technique =="average":
        return statistics.mean(distances)
    elif technique == "random":
        return random.choice(distances)
    else:
        return statistics.median(sorted(distances))

def lidar_camera_fusion(self, pred_bboxes, image):
    img_bis = image.copy()
    best_d = []
    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    distances = []
    for box in pred_bboxes:
        distances = []
        
        for i in range(self.imgfov_pts_2d.shape[0]):
      
            depth = self.imgfov_pc_velo[i,0]
            if (rectContains(box, self.imgfov_pts_2d[i], image.shape[1], image.shape[0], shrink_factor=0.2)==True):
                distances.append(depth)

                color = cmap[int(510.0 / depth), :]
                cv2.circle(img_bis,(int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))),2,color=tuple(color),thickness=-1,)
        h, w, _ = img_bis.shape
        if (len(distances)>2):
            distances = filter_outliers(distances)
            best_distance = get_best_distance(distances, technique="closest")
            cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (int(box[0]*w),int(box[1]*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3, cv2.LINE_AA)    
    
        best_d.append(best_distance)

        distances_to_keep = []
    
    return img_bis, best_d, w, h

LiDAR2Camera.lidar_camera_fusion = lidar_camera_fusion

def pipeline (self, image, point_cloud):
    "For a pair of 2 Calibrated Images"
    img = image.copy()
  
    lidar_img = self.show_lidar_on_image(point_cloud[:,:3], image)

    result, pred_bboxes = run_obstacle_detection(img)
   
    img_final, best_dist, w, h = self.lidar_camera_fusion(pred_bboxes, result)
    return img_final, best_dist, pred_bboxes, w, h

LiDAR2Camera.pipeline = pipeline

image_files = sorted(glob.glob("/home/usman/kitti_track/data_tracking_image_2/training/image_02/0000/*.png"))

point_files = sorted(glob.glob("/home/usman/kitti_track/data_tracking_velodyne/training/velodyne/0000/*.pcd"))
label_files = sorted(glob.glob("/home/usman/visual_fusion/data/label/*.txt"))
calib_files = sorted(glob.glob("/home/usman/visual_fusion/data/calib/*.txt"))
index=0

outt=cv2.VideoWriter('outputs/out_4.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10,(1242, 375))
p_dist= 5
while True:
    if index<152:
        print('index', index)
        index+=1
    else:
        print('Video has ended or failed, try a different video format!')
        break
    
    pcd_file = point_files[index]
    lidar2cam = LiDAR2Camera(calib_files[0])
    cloud = o3d.io.read_point_cloud(pcd_file)
    points= np.asarray(cloud.points)
    start_time = time.time()

    image = cv2.cvtColor(cv2.imread(image_files[index]), cv2.COLOR_BGR2RGB)
    img = cv2.imread(image_files[index])
 
    final_result, distance, pred_bb, w, h = lidar2cam.pipeline(image.copy(), points)
    print(distance)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    final_re= yolo.draw_bboxes(image, pred_bb)
    i=0
    for box in pred_bb:
        best_distance =distance[i]
        cv2.putText(final_re, '{0:.2f} m'.format(best_distance), (int(box[0]*w),int(box[1]*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3, cv2.LINE_AA) 
        i +=1

    t_time = time.time()- start_time
    speed = (abs(best_distance-p_dist))/t_time
    
    fps = 1.0 / (time.time() - start_time)
    print("FPS: %.2f" % fps)
    print("process: %.2f" % t_time)
    p_dist = best_distance.copy()
    print('speed', speed)
    outt.write(final_re)
    cv2.imshow("Output Video", final_re)
    cv2.waitKey(1)
 
cv2.destroyAllWindows()
