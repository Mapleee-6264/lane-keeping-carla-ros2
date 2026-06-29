import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
import math

class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")
        self.steering_sub = self.create_subscription(Float32, "/target_steering", self.steering_callback, 10)
        self.speed_sub = self.create_subscription(Float32, "/target_speed", self.speed_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/carla/ego_vehicle/odometry", self.odom_callback, 10)

        self.control_pub = self.create_publisher(CarlaEgoVehicleControl, "/carla/ego_vehicle/vehicle_control_cmd", 10)

        self.target_steering = 0.0
        self.target_speed = 0.0
        self.current_speed = 0.0

        self.last_steer = 0.0
        self.max_steer_rate = 2.0

        self.Kp = 0.5
        self.Ki = 0.1
        self.ff_gain = 0.15
        self.integral = 0.0
        self.max_integral = 1.5
        self.deadband = 0.1
        self.dt = 0.02

        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.throttle_rate = 0.5
        self.brake_rate = 0.8

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("Control Node (gentle) started!")

    def steering_callback(self, msg): self.target_steering = msg.data
    def speed_callback(self, msg): self.target_speed = msg.data
    def odom_callback(self, msg):
        self.current_speed = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)

    def control_loop(self):
        steer_cmd = self.target_steering
        max_step = self.max_steer_rate * self.dt
        if steer_cmd > self.last_steer + max_step:
            steer_cmd = self.last_steer + max_step
        elif steer_cmd < self.last_steer - max_step:
            steer_cmd = self.last_steer - max_step
        self.last_steer = steer_cmd

        if self.target_speed <= 0.0:
            throttle = 0.0
            brake = 1.0
            self.integral = 0.0
        else:
            error = self.target_speed - self.current_speed
            if abs(error) < self.deadband:
                throttle = 0.0
                brake = 0.0
                self.integral *= 0.98
            else:
                self.integral += error * self.dt
                self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
                pi = self.Kp * error + self.Ki * self.integral
                ff = self.ff_gain * self.target_speed
                effort = ff + pi

                if effort > 0.0:
                    throttle = min(effort, 1.0)
                    brake = 0.0
                    if throttle >= 1.0 and error > 0:
                        self.integral -= error * self.dt
                else:
                    throttle = 0.0
                    brake = min(-effort, 1.0)
                    if brake >= 1.0 and error < 0:
                        self.integral -= error * self.dt

        max_t = self.throttle_rate * self.dt
        max_b = self.brake_rate * self.dt
        if throttle > self.last_throttle + max_t: throttle = self.last_throttle + max_t
        elif throttle < self.last_throttle - max_t: throttle = self.last_throttle - max_t
        self.last_throttle = throttle

        if brake > self.last_brake + max_b: brake = self.last_brake + max_b
        elif brake < self.last_brake - max_b: brake = self.last_brake - max_b
        self.last_brake = brake

        if brake > 0.01: throttle = 0.0
        if throttle > 0.01: brake = 0.0

        cmd = CarlaEgoVehicleControl()
        cmd.steer = float(steer_cmd)
        cmd.throttle = float(throttle)
        cmd.brake = float(brake)
        self.control_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()