import rclpy
from rclpy.node import Node

from stage_srv_cpp.srv import StageInference

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        # 创建服务
        self.srv = self.create_service(StageInference, 'stage_inference', self.process_callback)
        self.get_logger().info('Service is ready to process images.')

    def process_callback(self, request, response):
        self.get_logger().info(f'Received request. State array length: {len(request.state)}')
        self.get_logger().info(f'Primary Image encoding: {request.image_primary.encoding}')
        
        if request.state and request.state[0] > 0.5:
            response.stage = 2
        else:
            response.stage = 1
            
        self.get_logger().info(f'Returning stage: {response.stage}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()