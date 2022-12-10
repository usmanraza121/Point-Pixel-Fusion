from lidar_transform import*
LiDAR2Camera.lidar_camera_fusion = lidar_camera_fusion


def pipeline (self, image, point_cloud):
    "For a pair of 2 Calibrated Images"
    img = image.copy()

    lidar_img = self.show_lidar_on_image(point_cloud[:,:3], image)

    result, pred_bboxes = run_obstacle_detection(img)

    img_final, best_dist = self.lidar_camera_fusion(pred_bboxes, result)
    return img_final, best_dist
LiDAR2Camera.pipeline = pipeline

image_files = sorted(glob.glob("/home/usman/visual_fusion_course-main/data2/img/*.png"))
point_files = sorted(glob.glob("/home/usman/visual_fusion_course-main/data2/velodyne/*.pcd"))
label_files = sorted(glob.glob("/home/usman/visual_fusion_course-main/data/label/*.txt"))
calib_files = sorted(glob.glob("/home/usman/visual_fusion_course-main/data/calib/*.txt"))
index=0
while True:
    if index<25:
        print('index', index)
        index+=1
    else:
        print('Video has ended or failed, try a different video format!')
        break
    index +=1
    pcd_file = point_files[index]
    lidar2cam = LiDAR2Camera(calib_files[0])
    cloud = o3d.io.read_point_cloud(pcd_file)
    points= np.asarray(cloud.points)

    #index = 1
    image = cv2.cvtColor(cv2.imread(image_files[index]), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14,7))
    final_result, distance = lidar2cam.pipeline(image.copy(), points)
    print(distance)
    plt.imshow(final_result)
    plt.show()
cv2.destroyAllWindows()