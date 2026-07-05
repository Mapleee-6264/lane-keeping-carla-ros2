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
        self.obstacle_type_sub = self.create_subscription(String, "/obstacle_type", self.obstacle_type_callback, 10)
        self.obstacle_vehicle_side_sub = self.create_subscription(String, "/obstacle_vehicle_side", self.obstacle_vehicle_side_callback, 10)
        self.road_left_edge_sub = self.create_subscription(Float32, "/road_left_edge_error", self.road_left_edge_callback, 10)
        self.road_right_edge_sub = self.create_subscription(Float32, "/road_right_edge_error", self.road_right_edge_callback, 10)
        self.lane_left_sub = self.create_subscription(Bool, "/lane_clear_left", self.lane_left_callback, 10)
        self.lane_right_sub = self.create_subscription(Bool, "/lane_clear_right", self.lane_right_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/carla/ego_vehicle/odometry", self.odom_callback, 10)

        self.steering_pub = self.create_publisher(Float32, "/target_steering", 10)
        self.speed_pub = self.create_publisher(Float32, "/target_speed", 10)
        self.state_pub = self.create_publisher(String, "/vehicle_state", 10)

        self.lookahead_error = 0.0
        self.obstacle_detected = False
        self.obstacle_dist = 999.0
        self.obstacle_type = ""
        self.obstacle_vehicle_side = "unknown"
        self.road_left_edge_error = -3.5
        self.road_right_edge_error = 3.5
        self.current_lane = "right"
        self.lane_left_clear = True
        self.lane_right_clear = True
        self.state = "LANE_FOLLOWING"
        self.current_velocity = 0.0

        # Pure Pursuit cơ bản
        self.L = 2.9

        # Tham số cho hai chế độ
        self.lookahead_time = 0.55
        self.lookahead_distance_min = 3.0
        self.lookahead_distance_max = 12.0
        self.lookahead_curve_buffer = 2.0
        self.gain_straight = 0.75
        self.gain_curve = 1.0
        self.max_steer = 0.5

        # Tốc độ (m/s)
        self.target_speed = 11.11  # 40 km/h
        self.slow_speed = 3.5
        self.lane_change_speed = 4.0

        # Ngưỡng vật cản (có hysteresis rõ ràng để tránh chattering trạng thái)
        self.stop_dist = 2.5
        self.resume_dist = 10.0          # < lane_change_trigger_dist để tạo đệm
        self.slow_down_dist = 8.0
        self.lane_change_trigger_dist = 20.0
        self.lane_change_min_dist = 2.0

        # Lane change
        self.lane_follow_offset_ratio = 0.25
        self.lane_follow_offset_m = 3.5 * self.lane_follow_offset_ratio
        self.lane_change_min_time = 80
        self.lane_change_timer = 0
        self.lane_change_target = None
        self.lane_change_cooldown = 0
        self.lane_change_cooldown_max = 200

        self.brake_hold_time = 20
        self.brake_timer = 0

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
    def obstacle_type_callback(self, msg): self.obstacle_type = msg.data
    def obstacle_vehicle_side_callback(self, msg): self.obstacle_vehicle_side = msg.data
    def road_left_edge_callback(self, msg): self.road_left_edge_error = msg.data
    def road_right_edge_callback(self, msg): self.road_right_edge_error = msg.data
    def lane_left_callback(self, msg): self.lane_left_clear = msg.data
    def lane_right_callback(self, msg): self.lane_right_clear = msg.data
    def obstacle_dist_callback(self, msg): self.obstacle_dist = msg.data
    def odom_callback(self, msg):
        self.current_velocity = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)

    def compute_lookahead_distance(self, in_curve=False):
        speed_mps = max(0.1, self.current_velocity)
        Ld = speed_mps * self.lookahead_time
        Ld = max(self.lookahead_distance_min, min(self.lookahead_distance_max, Ld))
        if in_curve:
            Ld = min(self.lookahead_distance_max, Ld + self.lookahead_curve_buffer)
        return Ld

    def compute_steering(self, error, Ld, gain):
        amplified = error * gain
        alpha = math.atan2(amplified, Ld)
        delta = math.atan2(2.0 * self.L * math.sin(alpha), Ld)
        return max(-self.max_steer, min(self.max_steer, delta))

    def planning_loop(self):
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

        # Luôn tính góc lái bằng Ld động để phát hiện cua đúng hơn
        dynamic_ld_for_detection = self.compute_lookahead_distance(False)
        steer_straight = self.compute_steering(self.lookahead_error, dynamic_ld_for_detection, self.gain_straight)

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
            Ld = self.compute_lookahead_distance(True)
            gain = self.gain_curve
        else:
            Ld = self.compute_lookahead_distance(False)
            gain = self.gain_straight

        # Tính góc lái cuối cùng với Ld/gain đã chọn
        steer = self.compute_steering(self.lookahead_error, Ld, gain)
        speed = self.target_speed

        # FSM
        right_lane_error = self.lookahead_error
        left_lane_error = self.road_left_edge_error + self.lane_follow_offset_m
        current_lane_error = right_lane_error if self.current_lane == "right" else left_lane_error

        if self.state in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
            self.lane_change_timer += 1
            completed_direction = self.state
            target_error = left_lane_error if self.state == "LANE_CHANGE_LEFT" else right_lane_error
            steer = self.compute_steering(target_error, Ld, gain)
            speed = self.lane_change_speed
            if self.lane_change_timer > self.lane_change_min_time or abs(target_error - current_lane_error) < 0.25:
                self.get_logger().info(f"LANE_CHANGE complete (timer={self.lane_change_timer}, target_error={target_error:.3f}) -> LANE_FOLLOWING")
                self.state = "LANE_FOLLOWING"
                self.lane_change_cooldown = self.lane_change_cooldown_max
                self.lane_change_target = None
                self.current_lane = "left" if completed_direction == "LANE_CHANGE_LEFT" else "right"

        elif self.state == "BRAKE":
            self.brake_timer += 1
            steer = self.compute_steering(current_lane_error, Ld, gain)
            ignore_left_vehicle = self.obstacle_type == "vehicle" and self.obstacle_vehicle_side == "left"
            ignore_right_vehicle_while_left = self.current_lane == "left" and self.obstacle_type == "vehicle" and self.obstacle_vehicle_side == "right"
            if self.obstacle_type != "vehicle" or not self.obstacle_detected or self.obstacle_dist > self.resume_dist or ignore_left_vehicle or ignore_right_vehicle_while_left:
                self.state = "LANE_FOLLOWING"
                self.brake_timer = 0
                self.lane_change_target = None
            else:
                speed = 0.0 if self.obstacle_dist <= self.stop_dist else self.slow_speed
                if self.brake_timer >= self.brake_hold_time:
                    if self.lane_change_target == "left":
                        self.get_logger().info("BRAKE: hold complete -> initiating LANE_CHANGE_LEFT")
                        self.state = "LANE_CHANGE_LEFT"
                        self.lane_change_timer = 0
                    elif self.lane_change_target == "right":
                        self.get_logger().info("BRAKE: hold complete -> initiating LANE_CHANGE_RIGHT")
                        self.state = "LANE_CHANGE_RIGHT"
                        self.lane_change_timer = 0
                    else:
                        self.get_logger().info("BRAKE: hold complete but no lane target -> resuming LANE_FOLLOWING")
                        self.state = "LANE_FOLLOWING"
                        self.brake_timer = 0

        else:  # LANE_FOLLOWING
            steer = self.compute_steering(current_lane_error, Ld, gain)
            speed = self.target_speed

            ignore_left_vehicle = self.obstacle_type == "vehicle" and self.obstacle_vehicle_side == "left"
            ignore_right_vehicle_while_left = self.current_lane == "left" and self.obstacle_type == "vehicle" and self.obstacle_vehicle_side == "right"
            current_lane_blocked = self.obstacle_type == "vehicle" and self.obstacle_detected and self.obstacle_dist <= self.lane_change_trigger_dist and not (ignore_left_vehicle or ignore_right_vehicle_while_left)

            if self.current_lane == "left" and not self.obstacle_detected and self.lane_right_clear and self.lane_change_cooldown == 0:
                self.get_logger().info("Left lane objective reached and right lane clear -> returning to right lane")
                self.state = "LANE_CHANGE_RIGHT"
                self.lane_change_timer = 0
            elif self.current_lane == "left" and self.obstacle_dist > self.resume_dist and self.lane_right_clear and self.lane_change_cooldown == 0:
                self.get_logger().info("Obstacle passed -> returning to right lane")
                self.state = "LANE_CHANGE_RIGHT"
                self.lane_change_timer = 0
            elif current_lane_blocked:
                self.get_logger().info(f"Current lane blocked by vehicle obstacle at {self.obstacle_dist:.2f} m -> slow down and prepare lane change")
                self.state = "BRAKE"
                self.brake_timer = 0
                if self.obstacle_vehicle_side == "right" and self.lane_left_clear and self.lane_change_cooldown == 0:
                    self.lane_change_target = "left"
                    self.get_logger().info("BRAKE: vehicle obstacle on right road half -> plan LANE_CHANGE_LEFT after hold")
                elif self.lane_left_clear and self.lane_change_cooldown == 0:
                    self.lane_change_target = "left"
                    self.get_logger().info("BRAKE: lane_left_clear -> plan LANE_CHANGE_LEFT after hold")
                elif self.lane_right_clear and self.lane_change_cooldown == 0:
                    self.lane_change_target = "right"
                    self.get_logger().info("BRAKE: lane_right_clear -> plan LANE_CHANGE_RIGHT after hold")
                else:
                    self.lane_change_target = None
                    self.get_logger().info("BRAKE: no adjacent lane available, holding")
                speed = 0.0 if self.obstacle_dist <= self.stop_dist else self.slow_speed
            if self.state == "LANE_FOLLOWING" and abs(steer) > 0.45:
                speed = max(self.lane_change_speed, self.target_speed * 0.5)

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
            f"{self.state} | Curve={self.in_curve} | Vel={self.current_velocity:.2f}m/s | Err={self.lookahead_error:.3f}m | Ld={Ld:.1f} gain={gain:.2f} Steer={steer:.2f}",
            throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
