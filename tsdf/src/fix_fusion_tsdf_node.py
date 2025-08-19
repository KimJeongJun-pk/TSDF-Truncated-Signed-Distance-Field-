#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_geometry_msgs
import tf
import cv2

bridge = CvBridge()
depth_intrinsic = None
frame_count = 0

tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length = 0.005,
    sdf_trunc = 0.04,
    color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

#----------Extrinsic Parameter----------
T_lidar_to_cam = np.array([
[ 0.994465644344459, 0.096213952332983, -0.042201393283000, -0.085813267661546],
[-0.050760946414676, 0.088325869182148, -0.994797400053048,  0.053176436492591],
[-0.091985914887261, 0.991434020099010,  0.092720953687228, -0.090404115072207] 
])

#----------PointCloud Coordinate transform----------    
def transform_pointcloud(cloud_msg):
    points = []
    for pt in pc2.read_points(cloud_msg, skip_nans = True):
        p = np.array([pt[0], pt[1], pt[2], 1.0])
        p_cam = T_lidar_to_cam @ p
        points.append(p_cam[:3])
    return np.array(points)
    
def camera_info_callback(msg):
    global depth_intrinsic
    if depth_intrinsic is None:
        depth_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        depth_intrinsic.set_intrinsics(
            msg.width, msg.height,
            msg.K[0], msg.K[4],
            msg.K[2], msg.K[5]
        )
        rospy.loginfo("Camera intirnsics set.")

#----------Point Cloud -> Depth Map----------        
def generate_depth_image(points, width, height, fx, fy, cx, cy): 
    depth_image = np.zeros((height, width), dtype=np.uint16)
    for p in points:
        if p[2] <= 0: continue
        u = int(fx * p[0]/p[2] + cx)
        v = int(fy * p[1]/p[2] + cy)
        if 0 <= u < width and 0 <= v < height:
            d = int(p[2] * 1000.0)
            if depth_image[v, u] == 0 or d < depth_image[v, u]:
                depth_image[v, u] = d
    return depth_image

#----------TSDF----------
def fusion_callback(color_msg, depth_msg, lidar_msg):
    global frame_count, depth_intrinsic
    if depth_intrinsic is None:
        rospy.logwarn("Waiting for camera info")
        return
        
    try:
        color = bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        
        depth = depth.copy()
        depth[depth == 0] = 65535 
        
        if color.shape[:2] != depth.shape[:2]:
            rospy.logwarn("Resizing color image to match depth resolution")
            color = cv2.resize(color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        color_o3d = o3d.geometry.Image(np.asarray(color, dtype=np.uint8))
        depth_o3d = o3d.geometry.Image(np.asarray(depth, dtype=np.uint16))    
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale = 1000.0,
            depth_trunc = 2.0,
            convert_rgb_to_intensity=False
        )
        
        cam_pose = np.eye(4)
        
        tsdf_volume.integrate(rgbd, depth_intrinsic, cam_pose)
        
        points_lidar = transform_pointcloud(lidar_msg)
        width, height = depth.shape[1], depth.shape[0]
        fx, fy, cx, cy = (
            depth_intrinsic.get_focal_length()[0], depth_intrinsic.get_focal_length()[1],
            depth_intrinsic.get_principal_point()[0], depth_intrinsic.get_principal_point()[1]
        )
        lidar_depth = generate_depth_image(points_lidar, width, height, fx, fy, cx, cy)
        rgb_black = np.zeros_like(color, dtype=np.uint8)
        
        rgbd_lidar = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_black),
            o3d.geometry.Image(lidar_depth),
            depth_scale = 1000.0,
            depth_trunc = 2.0,
            convert_rgb_to_intensity = False
        )
        
        tsdf_volume.integrate(rgbd_lidar, depth_intrinsic, cam_pose)
        
        frame_count += 1
        rospy.loginfo(f"[TSDF] Frame {frame_count}/30 integrated")
        
        if frame_count == 30 :
            rospy.loginfo("Extracting TSDF mesh...")
            mesh = tsdf_volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh], window_name = "TSDF Mesh")
            
            rospy.loginfo("Extracting TSDF point cloud...")
            pcd = tsdf_volume.extract_point_cloud()
            o3d.visualization.draw_geometries([pcd], window_name="TSDF PointCloud")

            rospy.loginfo("Creating voxel grid from point cloud...")
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.01)
            o3d.visualization.draw_geometries([voxel_grid], window_name="Voxelized PointCloud")
        
            rospy.signal_shutdown("TSDF, Voxel Visualization Completed")
                         
    except Exception as e:
        rospy.logerr(f"Fusion error : {e}")
        
def main():
    rospy.init_node("fix_fusion_tsdf_node")
    
    image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
    lidar_sub = message_filters.Subscriber("/denoised/pointcloud", PointCloud2)
    
    depth_info_sub = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, camera_info_callback)

    sync = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub, lidar_sub], 10, 0.01)
    sync.registerCallback(fusion_callback)

    rospy.loginfo("TSDF mapper node started")
    rospy.spin()      

if __name__ =="__main__":
    main()
