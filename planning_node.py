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
        self.gain_curve = 1.1
        self.max_steer = 0.6

        # Tốc độ (m/s)
        self.target_speed = 11.11  # 40 km/h
        self.slow_speed = 3.5
        self.lane_change_speed = 4.0

        # ---- Ngưỡng vật cản loại "other" (không phải xe hơi) ----
        self.other_slow_dist = 9.0                      # < 8m: slow down + tìm tâm làn trống
        self.other_lane_change_completion_error = 0.5   # đạt tâm làn (sai số < 0.5m) -> về lane following

        # ---- Ngưỡng vật cản loại "vehicle" (xe hơi), chỉ xét khi ở nửa phải đường ----
        self.car_slow_dist = 16            # < 15m: slow down
        self.car_avoid_dist = 12          # < 8m: né tạm sang trái
        self.car_avoid_hold_time = 300       # số chu kỳ (2s @ 50Hz) giữ tâm làn tạm rồi tự trả về lane following

        # Lane change (dùng cho AVOID_OTHER)
        self.lane_follow_offset_ratio = 0.50
        self.lane_follow_offset_m = 3.5 * self.lane_follow_offset_ratio
        self.lane_change_steer_gain = 1.35
        self.lane_change_lookahead_min = 4.0

        self.avoid_other_target_lane = None   # "left" hoặc "right": làn đang né tới
        self.avoid_car_timer = 0

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

        # ---- Các "tâm làn" có thể dùng ----
        right_lane_error = self.lookahead_error                              # tâm làn phải (mặc định)
        left_lane_error = self.road_left_edge_error + self.lane_follow_offset_m   # tâm làn trái
        road_center_error = (self.road_left_edge_error + self.road_right_edge_error) / 2.0
        # Tâm làn tạm khi né xe: trung điểm giữa mép trái đường và tâm đường
        avoid_car_target_error = (self.road_left_edge_error *0.85+ road_center_error*0.15) 

        current_lane_error = right_lane_error if self.current_lane == "right" else left_lane_error

        # Vật cản "vehicle" ở nửa trái đường (mép trái -> tâm đường) thì bỏ qua hoàn toàn
        car_relevant = self.obstacle_type == "vehicle" and self.obstacle_detected and self.obstacle_vehicle_side == "right"
        other_relevant = self.obstacle_type == "other" and self.obstacle_detected

        if self.state == "AVOID_CAR":
            # Né tạm sang trái trong một khoảng thời gian cố định rồi tự quay lại lane following
            self.avoid_car_timer += 1
            steer = self.compute_steering(avoid_car_target_error, Ld, gain * self.lane_change_steer_gain)
            speed = self.lane_change_speed
            if self.avoid_car_timer >= self.car_avoid_hold_time:
                self.get_logger().info("AVOID_CAR: hold time xong -> về LANE_FOLLOWING, tâm làn trở lại bình thường")
                self.state = "LANE_FOLLOWING"
                self.avoid_car_timer = 0

        elif self.state == "AVOID_OTHER":
            target_error = left_lane_error if self.avoid_other_target_lane == "left" else right_lane_error
            avoid_ld = max(self.lane_change_lookahead_min, Ld * 0.65)
            steer = self.compute_steering(target_error, avoid_ld, gain * self.lane_change_steer_gain)
            speed = self.slow_speed
            reached = abs(self.lookahead_error - target_error) < self.other_lane_change_completion_error
            obstacle_gone = not other_relevant or self.obstacle_dist > self.other_slow_dist
            if reached or obstacle_gone:
                if reached:
                    self.get_logger().info(f"AVOID_OTHER: đạt tâm làn {self.avoid_other_target_lane} -> về LANE_FOLLOWING")
                    self.current_lane = self.avoid_other_target_lane
                else:
                    self.get_logger().info("AVOID_OTHER: vật cản đã hết -> về LANE_FOLLOWING")
                self.state = "LANE_FOLLOWING"
                self.avoid_other_target_lane = None

        else:  # LANE_FOLLOWING
            steer = self.compute_steering(current_lane_error, Ld, gain)
            speed = self.target_speed

            if car_relevant and self.obstacle_dist < self.car_avoid_dist:
                self.get_logger().info(f"Vehicle obstacle (nửa phải đường) tại {self.obstacle_dist:.2f}m -> né tạm sang trái")
                self.state = "AVOID_CAR"
                self.avoid_car_timer = 0
                steer = self.compute_steering(avoid_car_target_error, Ld, gain * self.lane_change_steer_gain)
                speed = self.lane_change_speed
            elif car_relevant and self.obstacle_dist < self.car_slow_dist:
                speed = self.slow_speed

            elif other_relevant and self.obstacle_dist < self.other_slow_dist:
                # Tìm tâm làn trống để né: ưu tiên làn khác với làn hiện tại
                target_lane = None
                if self.current_lane == "right" and self.lane_left_clear:
                    target_lane = "left"
                elif self.current_lane == "left" and self.lane_right_clear:
                    target_lane = "right"

                if target_lane is not None:
                    self.get_logger().info(f"Other obstacle tại {self.obstacle_dist:.2f}m -> tìm tâm làn {target_lane}")
                    self.state = "AVOID_OTHER"
                    self.avoid_other_target_lane = target_lane
                    target_error = left_lane_error if target_lane == "left" else right_lane_error
                    avoid_ld = max(self.lane_change_lookahead_min, Ld * 0.65)
                    steer = self.compute_steering(target_error, avoid_ld, gain * self.lane_change_steer_gain)
                    speed = self.slow_speed
                else:
                    # Không có làn trống để né -> chỉ giảm tốc, giữ nguyên làn
                    speed = self.slow_speed

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
