#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose


class BoxDetector(Node):
    def __init__(self):
        super().__init__('box_detector')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers (adjust topics as needed)
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.pc_sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.pc_callback, 10)
        
        # Publishers
        self.box_pub = self.create_publisher(PoseArray, '/detected_boxes', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        
        # Machine Learning Model (example using OpenCV's DNN)
        self.net = cv2.dnn.readNetFromTensorflow('box_detection.pb', 'box_detection.pbtxt')
        
        # Storage for current frame and point cloud
        self.current_frame = None
        self.current_pc = None
        
        self.get_logger().info("Box Detector Node Initialized")

    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_frame()
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")

    def pc_callback(self, msg):
        self.current_pc = msg

    def process_frame(self):
        if self.current_frame is None or self.current_pc is None:
            return
            
        # Preprocess image for ML model
        blob = cv2.dnn.blobFromImage(
            self.current_frame, 
            scalefactor=1.0/127.5, 
            size=(300, 300),
            mean=[127.5, 127.5, 127.5],
            swapRB=True,
            crop=False
        )
        
        # Run object detection
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Process detections
        boxes = PoseArray()
        boxes.header.frame_id = "camera_frame"  # Adjust to your frame
        
        (h, w) = self.current_frame.shape[:2]
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.7:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Get 3D position from point cloud (simplified)
                # In real implementation, you'd process the point cloud data
                center_x = (startX + endX) // 2
                center_y = (startY + endY) // 2
                
                # Create pose for the detected box
                pose = Pose()
                pose.position.x = center_x  # Replace with actual 3D position
                pose.position.y = center_y
                pose.position.z = 0.0  # Get from point cloud
                
                # Add to detected boxes
                boxes.poses.append(pose)
                
                # Draw bounding box (for debugging)
                cv2.rectangle(
                    self.current_frame, 
                    (startX, startY), 
                    (endX, endY),
                    (0, 255, 0), 2
                )
                
        # Publish results
        self.box_pub.publish(boxes)
        
        # Publish debug image
        debug_msg = self.bridge.cv2_to_imgmsg(self.current_frame, "bgr8")
        self.debug_pub.publish(debug_msg)
        
    # Modify process_frame method:
    def process_frame_with_yolo(self):
        # YOLO-specific processing
        height, width, channels = self.current_frame.shape
        
        # Detecting objects
        blob = cv2.dnn.blobFromImage(
            self.current_frame, 
            0.00392, 
            (416, 416), 
            (0, 0, 0), 
            True, 
            crop=False
        )
        
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

def main(args=None):
    rclpy.init(args=args)
    box_detector = BoxDetector()
    rclpy.spin(box_detector)
    box_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()