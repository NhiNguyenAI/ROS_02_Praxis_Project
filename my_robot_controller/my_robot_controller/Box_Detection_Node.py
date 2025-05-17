#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseArray, Pose, Vector3, TransformStamped
import numpy as np
import open3d as o3d
from rclpy.qos import qos_profile_sensor_data
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import CameraInfo
import math

class BoxDetectionSystem(Node):
    def __init__(self):
        super().__init__('box_detection_system')
        
        # Initialize sensor subscribers
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/depth_camera/points',
            self.pointcloud_callback,
            qos_profile_sensor_data)
        
        self.image_sub = self.create_subscription(
            Image,
            '/depth_camera/rgb',
            self.image_callback,
            qos_profile_sensor_data)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/depth_camera/camera_info',
            self.camera_info_callback,
            10)
        
        # Initialize publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization/boxes', 10)
        self.dimensions_pub = self.create_publisher(MarkerArray, '/visualization/dimensions', 10)
        self.overlay_pub = self.create_publisher(Image, '/visualization/overlay', 10)
        
        # TF broadcaster for box coordinate systems
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Image processing
        self.bridge = CvBridge()
        self.current_image = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Box detection parameters
        self.min_box_height = 0.02  # 2cm
        self.max_box_height = 1.0   # 1m
        self.min_box_volume = 0.0005  # 500cm³
        self.max_box_volume = 0.5     # 0.5m³
        self.aspect_ratio_threshold = 8.0
        
        # Visualization parameters
        self.box_colors = {
            'small': ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.7),   # Green
            'medium': ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.7), # Orange
            'large': ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7)   # Red
        }
        
        self.get_logger().info("Box Detection System initialized")

    def camera_info_callback(self, msg):
        """Store camera intrinsic parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Store current RGB image"""
        self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def pointcloud_callback(self, msg):
        try:
            # Convert ROS PointCloud2 to Open3D format
            pcd = self.pointcloud2_to_o3d(msg)
            
            # Process point cloud and detect boxes
            boxes, dimensions_list = self.detect_boxes(pcd)
            
            # Publish visualization
            self.publish_visualization(boxes, dimensions_list, msg.header)
            
            # Publish coordinate systems
            self.publish_coordinate_frames(boxes, dimensions_list, msg.header)
            
            # Create and publish overlay image
            if self.current_image is not None:
                overlay_img = self.create_overlay_image(boxes, dimensions_list)
                self.overlay_pub.publish(self.bridge.cv2_to_imgmsg(overlay_img, "bgr8"))
                
        except Exception as e:
            self.get_logger().error(f"Processing failed: {str(e)}")

    def pointcloud2_to_o3d(self, msg):
        """Convert ROS PointCloud2 to Open3D point cloud"""
        pcd = o3d.geometry.PointCloud()
        points = np.frombuffer(msg.data, dtype=np.float32)
        points = points.reshape(-1, 4)[:,:3]  # XYZ only
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def detect_boxes(self, pcd):
        """Detect boxes in point cloud with improved processing"""
        # Preprocessing
        pcd = self.preprocess_pointcloud(pcd)
        
        # Remove dominant plane (floor/table)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=100)
        objects_pcd = pcd.select_by_index(inliers, invert=True)
        
        # Cluster remaining points
        labels = np.array(objects_pcd.cluster_dbscan(
            eps=0.05, 
            min_points=20, 
            print_progress=False))
        
        boxes = []
        dimensions_list = []
        max_label = labels.max()
        
        for label in range(max_label + 1):
            cluster = objects_pcd.select_by_index(np.where(labels == label)[0])
            
            if len(cluster.points) < 50:  # Skip small clusters
                continue
                
            try:
                # Get oriented bounding box
                bbox = cluster.get_oriented_bounding_box()
                
                # Get sorted dimensions (length > width > height)
                dimensions = np.sort(np.abs(bbox.extent))[::-1]
                length, width, height = dimensions
                
                # Calculate metrics
                volume = length * width * height
                max_aspect = max(length/width, length/height, width/height)
                
                # Validate box shape
                is_valid_box = (
                    self.min_box_height < height < self.max_box_height and
                    self.min_box_volume < volume < self.max_box_volume and
                    max_aspect < self.aspect_ratio_threshold
                )
                
                if is_valid_box:
                    boxes.append(bbox)
                    
                    # Classify box size
                    size_category = self.classify_box_size(volume)
                    
                    # Store box info
                    box_info = {
                        'dimensions': (length, width, height),
                        'volume': volume,
                        'size_category': size_category,
                        'center': bbox.center,
                        'rotation': bbox.R,
                        'label': f"box_{label}"
                    }
                    dimensions_list.append(box_info)
                    
            except Exception as e:
                self.get_logger().warn(f"Box processing error: {str(e)}")
                continue
                
        return boxes, dimensions_list

    def preprocess_pointcloud(self, pcd):
        """Clean and filter point cloud data"""
        # Remove outliers
        cl, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # Downsample
        return cl.voxel_down_sample(voxel_size=0.01)

    def classify_box_size(self, volume):
        """Classify box into size category"""
        if volume < 0.01:    # < 10 liters
            return 'small'
        elif volume < 0.1:   # < 100 liters
            return 'medium'
        else:                # >= 100 liters
            return 'large'

    def publish_visualization(self, boxes, dimensions_list, header):
        """Create and publish visualization markers"""
        box_markers = MarkerArray()
        dimension_markers = MarkerArray()
        
        for i, (box, dims) in enumerate(zip(boxes, dimensions_list)):
            # Box marker
            marker = Marker()
            marker.header = header
            marker.ns = "boxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = box.center[0]
            marker.pose.position.y = box.center[1]
            marker.pose.position.z = box.center[2]
            marker.pose.orientation = self.rotation_matrix_to_quaternion(box.R)
            marker.scale.x = dims['dimensions'][0]  # length
            marker.scale.y = dims['dimensions'][1]  # width
            marker.scale.z = dims['dimensions'][2]  # height
            marker.color = self.box_colors[dims['size_category']]
            marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
            box_markers.markers.append(marker)
            
            # Dimension text marker
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "dimensions"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = box.center[0]
            text_marker.pose.position.y = box.center[1]
            text_marker.pose.position.z = box.center[2] + dims['dimensions'][2]/2 + 0.05
            text_marker.scale.z = 0.05  # Text size
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.text = (f"{dims['label']}\n"
                              f"L: {dims['dimensions'][0]:.2f}m\n"
                              f"W: {dims['dimensions'][1]:.2f}m\n"
                              f"H: {dims['dimensions'][2]:.2f}m")
            text_marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
            dimension_markers.markers.append(text_marker)
        
        self.marker_pub.publish(box_markers)
        self.dimensions_pub.publish(dimension_markers)

    def publish_coordinate_frames(self, boxes, dimensions_list, header):
        """Publish TF frames for each box"""
        transforms = []
        
        for box, dims in zip(boxes, dimensions_list):
            t = TransformStamped()
            t.header = header
            t.child_frame_id = dims['label']
            
            # Set translation
            t.transform.translation.x = box.center[0]
            t.transform.translation.y = box.center[1]
            t.transform.translation.z = box.center[2]
            
            # Set rotation
            t.transform.rotation = self.rotation_matrix_to_quaternion(box.R)
            
            transforms.append(t)
        
        # Broadcast all transforms
        self.tf_broadcaster.sendTransform(transforms)

    def create_overlay_image(self, boxes, dimensions_list):
        """Create 2D image overlay with box information"""
        overlay = self.current_image.copy()
        
        if self.camera_matrix is None:
            return overlay
            
        for box, dims in zip(boxes, dimensions_list):
            try:
                # Project box center to image
                center_3d = np.array([box.center[0], box.center[1], box.center[2]])
                center_2d, _ = cv2.projectPoints(
                    center_3d.reshape(1, 3),
                    np.zeros(3),
                    np.zeros(3),
                    self.camera_matrix,
                    self.distortion_coeffs)
                
                x, y = int(center_2d[0][0][0]), int(center_2d[0][0][1])
                
                # Draw information on image
                color = (
                    int(self.box_colors[dims['size_category']].b * 255),
                    int(self.box_colors[dims['size_category']].g * 255),
                    int(self.box_colors[dims['size_category']].r * 255)
                )
                
                cv2.circle(overlay, (x, y), 5, color, -1)
                cv2.putText(overlay, dims['label'], (x+10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(overlay, f"L:{dims['dimensions'][0]:.2f}m", (x+10, y+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(overlay, f"W:{dims['dimensions'][1]:.2f}m", (x+10, y+40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(overlay, f"H:{dims['dimensions'][2]:.2f}m", (x+10, y+60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
            except Exception as e:
                self.get_logger().warn(f"Image projection error: {str(e)}")
                continue
                
        return overlay

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        # Using scipy's implementation for robustness
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()
        
        q = Quaternion()
        q.x = quat[0]
        q.y = quat[1]
        q.z = quat[2]
        q.w = quat[3]
        return q

def main(args=None):
    rclpy.init(args=args)
    node = BoxDetectionSystem()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()