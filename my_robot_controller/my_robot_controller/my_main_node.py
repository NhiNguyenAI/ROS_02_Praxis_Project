#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class MyFirstNode(Node):
    def __init__(self):
        super().__init__('my_first_node')
        self.counter = 0
        # Create a timer that calls the timer_callback every second
        self.timer = self.create_timer(1.0, self.timer_callback)

    
    def timer_callback(self):
        self.get_logger().info('Hello, world! %d' % self.counter)
        # Increment the counter
        self.counter += 1
        
def main(args=None):
    rclpy.init(args=args)

    node = MyFirstNode()
    rclpy.spin(node)
    
    rclpy.shutdown()
if __name__ == '__main__':
    main() 