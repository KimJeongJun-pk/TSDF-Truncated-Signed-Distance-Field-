#!/usr/bin/env python3
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import open3d as o3d
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D

tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.005,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

depth_intrinsic = None
bridge = CvBridge()
frame_count = 0

def depth_camera_info_callback(msg):
    global depth_intrinsic
    if depth_intrinsic is None:
        depth_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        depth_intrinsic.set_intrinsics(
            msg.width, msg.height,
            msg.K[0], msg.K[4],  # fx, fy
            msg.K[2], msg.K[5]   # cx, cy
        )
        rospy.loginfo(f"Depth camera intrinsics set: {msg.width}x{msg.height}")

def rgbd_callback(color_msg, depth_msg):
    global frame_count
    if depth_intrinsic is None:
        rospy.logwarn("Depth intrinsic not set yet.")
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
            depth_scale=1000.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False
        )

        cam_pose = np.eye(4)

        tsdf_volume.integrate(rgbd, depth_intrinsic, cam_pose)

        frame_count += 1
        rospy.loginfo(f"[TSDF] Frame {frame_count}/30 integrated")

        if frame_count == 30:
            rospy.loginfo("Extracting TSDF mesh...")
            mesh = tsdf_volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh], window_name="TSDF Mesh")
            
            rospy.loginfo("Extracting TSDF Point Cloud ...")
            pcd = tsdf_volume.extract_point_cloud()
            o3d.visualization.draw_geometries([pcd], window_name="TSDF PointCloud")
            
            rospy.loginfo("Extracting TSDF Voxel Grid ...")
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.01)
            o3d.visualization.draw_geometries([voxel_grid], window_name="Voxelized PointCloud")
                       
            rospy.signal_shutdown("TSDF, Voxel Visualization Completed")

    except Exception as e:
        rospy.logerr(f"RGBD callback error: {str(e)}")

def main():
    rospy.init_node("tsdf_mapper")

    image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
    depth_info_sub = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, depth_camera_info_callback)

    sync = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 0.1)
    sync.registerCallback(rgbd_callback)

    rospy.loginfo("TSDF mapper node started")
    rospy.spin()

if __name__ == "__main__":
    main()
