import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String
from nav_msgs.msg import Odometry
import math

class PlanningNode(Node):
    def __init__(self):
        super().__init__("planning_node")
        self.lookahead_sub = self.create_subscription(Float32, "/lookahead_error", self.lookahead_callback, 10)
        self.obstacle_sub = self.create_subscription(Bool, "/obstacle_detected", self.obstacle_callback, 10)
        self.obstacle_dist_sub = self.create_subscription(Float32, "/obstacle_distance", self.obstacle_dist_callback, 10)
        self.lane_left_sub = self.create_subscription(Bool, "/lane_clear_left", self.lane_left_callback, 10)
        self.lane_right_sub = self.create_subscription(Bool, "/lane_clear_right", self.lane_right_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/carla/ego_vehicle/odometry", self.odom_callback, 10)

        self.steering_pub = self.create_publisher(Float32, "/target_steering", 10)
        self.speed_pub = self.create_publisher(Float32, "/target_speed", 10)
        self.state_pub = self.create_publisher(String, "/vehicle_state", 10)

        self.lookahead_error = 0.0
        self.obstacle_detected = False
        self.obstacle_dist = 999.0
        self.lane_left_clear = True
        self.lane_right_clear = True
        self.state = "LANE_FOLLOWING"
        self.current_velocity = 0.0

        # Pure Pursuit cơ bản
        self.L = 2.9

        # Tham số cho hai chế độ
        self.Ld_straight = 6.0
        self.gain_straight = 0.8
        self.Ld_curve = 14.0
        self.gain_curve = 1.4
        self.max_steer = 0.4

        # Tốc độ (m/s)
        self.target_speed = 5.56
        self.slow_speed = 2.5
        self.lane_change_speed = 2.0

        # Ngưỡng vật cản
        self.stop_dist = 5.0
        self.resume_dist = 8.0
        self.slow_down_dist = 15.0
        self.lane_change_min_dist = 3.0

        # Lane change
        self.lane_change_offset = 0.8
        self.lane_change_min_time = 100
        self.lane_change_timer = 0
        self.lane_change_target = None
        self.lane_change_cooldown = 0
        self.lane_change_cooldown_max = 200

        # ---- Phát hiện cua dựa trên GÓC LÁI ----
        self.in_curve = False
        self.steer_curve_threshold = 0.15   # nếu góc lái vượt ngưỡng này -> vào cua
        self.steer_exit_threshold = 0.10    # nếu góc lái dưới ngưỡng này -> có thể thoát cua
        self.exit_counter = 0
        self.exit_required = 25             # 0.5 giây @ 50 Hz

        self.timer = self.create_timer(0.02, self.planning_loop)
        self.get_logger().info("Planning Node (steer-based curve detection) started!")

    def lookahead_callback(self, msg): self.lookahead_error = msg.data
    def obstacle_callback(self, msg): self.obstacle_detected = msg.data
    def lane_left_callback(self, msg): self.lane_left_clear = msg.data
    def lane_right_callback(self, msg): self.lane_right_clear = msg.data
    def obstacle_dist_callback(self, msg): self.obstacle_dist = msg.data
    def odom_callback(self, msg):
        self.current_velocity = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)

    def compute_steering(self, error, Ld, gain):
        amplified = error * gain
        alpha = math.atan2(amplified, Ld)
        delta = math.atan2(2.0 * self.L * math.sin(alpha), Ld)
        return max(-self.max_steer, min(self.max_steer, delta))

    def planning_loop(self):
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

        # Luôn tính góc lái ở chế độ đường thẳng để phát hiện cua
        steer_straight = self.compute_steering(self.lookahead_error, self.Ld_straight, self.gain_straight)

        # Cập nhật trạng thái cua dựa trên steer_straight
        if not self.in_curve:
            if abs(steer_straight) > self.steer_curve_threshold:
                self.in_curve = True
                self.exit_counter = 0
        else:
            if abs(steer_straight) < self.steer_exit_threshold:
                self.exit_counter += 1
                if self.exit_counter >= self.exit_required:
                    self.in_curve = False
                    self.exit_counter = 0
            else:
                self.exit_counter = 0

        # Chọn Ld và gain theo trạng thái cua
        if self.in_curve:
            Ld = self.Ld_curve
            gain = self.gain_curve
        else:
            Ld = self.Ld_straight
            gain = self.gain_straight

        # Tính góc lái cuối cùng với Ld/gain đã chọn
        steer = self.compute_steering(self.lookahead_error, Ld, gain)
        speed = self.target_speed

        # FSM
        if self.state == "STOP":
            steer = 0.0
            speed = 0.0
            if not self.obstacle_detected or self.obstacle_dist > self.resume_dist:
                self.state = "LANE_FOLLOWING"
            elif self.obstacle_detected and self.obstacle_dist < self.lane_change_min_dist:
                if self.lane_left_clear and self.lane_change_cooldown == 0:
                    self.state = "LANE_CHANGE_LEFT"
                    self.lane_change_timer = 0
                    self.lane_change_target = "left"
                elif self.lane_right_clear and self.lane_change_cooldown == 0:
                    self.state = "LANE_CHANGE_RIGHT"
                    self.lane_change_timer = 0
                    self.lane_change_target = "right"

        elif self.state in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
            self.lane_change_timer += 1
            sign = -1.0 if self.state == "LANE_CHANGE_LEFT" else 1.0
            error = self.lookahead_error + sign * self.lane_change_offset
            steer = self.compute_steering(error, Ld, gain)
            speed = self.lane_change_speed
            if self.lane_change_timer > self.lane_change_min_time:
                self.state = "LANE_FOLLOWING"
                self.lane_change_cooldown = self.lane_change_cooldown_max
                self.lane_change_target = None

        else:  # LANE_FOLLOWING
            steer = self.compute_steering(self.lookahead_error, Ld, gain)
            if self.obstacle_detected:
                if self.obstacle_dist < self.stop_dist:
                    self.state = "STOP"
                elif self.obstacle_dist < self.slow_down_dist:
                    speed = self.slow_speed
            if abs(steer) > 0.5:
                speed = max(2.0, self.target_speed * 0.6)

        # Publish
        steer_msg = Float32()
        steer_msg.data = float(steer)
        self.steering_pub.publish(steer_msg)

        speed_msg = Float32()
        speed_msg.data = float(speed)
        self.speed_pub.publish(speed_msg)

        state_msg = String()
        state_msg.data = self.state
        self.state_pub.publish(state_msg)

        self.get_logger().info(
            f"{self.state} | Curve={self.in_curve} | Err={self.lookahead_error:.3f}m | Ld={Ld:.1f} gain={gain:.2f} Steer={steer:.2f}",
            throttle_duration_sec=0.5)

def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()