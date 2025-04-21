#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class MyFirstNode(Node):
    def __init__(self):
        super().__init__('my_first_node')
        self.get_logger().info('Hello, ROS 2!')
        
def main(args=None):
    rclpy.init(args=args)

    node = MyFirstNode()
    
    rclpy.shutdown()
if __name__ == '__main__':
    main()
