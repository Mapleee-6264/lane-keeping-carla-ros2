import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32, Bool
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time

class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.camera_sub = self.create_subscription(Image, "/carla/ego_vehicle/semantic_segmentation_front/image", self.camera_callback, 10)
        self.lidar_sub = self.create_subscription(PointCloud2, "/carla/ego_vehicle/lidar", self.lidar_callback, 10)

        self.lookahead_pub = self.create_publisher(Float32, "/lookahead_error", 10)
        self.obstacle_pub = self.create_publisher(Bool, "/obstacle_detected", 10)
        self.obstacle_dist_pub = self.create_publisher(Float32, "/obstacle_distance", 10)
        self.lane_left_clear_pub = self.create_publisher(Bool, "/lane_clear_left", 10)
        self.lane_right_clear_pub = self.create_publisher(Bool, "/lane_clear_right", 10)

        self.bridge = CvBridge()
        self.fov_hor = math.radians(90.0)
        # lookahead_distance đồng bộ với Ld mới của planning
        self.lookahead_distance = 7.5
        self.roi_center_ratio = 0.40
        self.roi_height_ratio = 0.30
        self.image_width = None

        self.camera_obstacle_detected = False
        self.camera_obstacle_distance = 999.0
        self.vehicle_color_lower = np.array([0, 0, 130])
        self.vehicle_color_upper = np.array([60, 60, 180])

        # LiDAR
        self.obstacle_min_z = 0.0
        self.obstacle_min_x = 0.3     # bỏ qua vật cản ngay sát/phía sau xe
        self.lane_check_dist = 30.0
        self.lane_offset = 3.5
        self.lane_tolerance = 2.0
        self.clear_threshold = 10.0

        # Điểm bám cách mép phải 25% chiều rộng làn (dịch sang trái nhẹ)
        self.lateral_offset_ratio = 0.25

        self.last_camera_time = 0.0
        self.last_lidar_time = 0.0
        self.min_interval = 0.1

        self.filtered_error = 0.0
        self.error_alpha = 0.4          # lọc nhẹ

        self.get_logger().info("Perception (right edge, lower-mid ROI) started!")

    def camera_callback(self, msg):
        try:
            now = time.time()
            if now - self.last_camera_time < self.min_interval:
                return
            self.last_camera_time = now

            # 1. Đọc ảnh, resize
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frame = cv2.resize(frame, (640, 360))
            h, w = frame.shape[:2]

            if self.image_width is None:
                self.image_width = w
                self.get_logger().info(f"Image width: {w}")

            # 2. Mặt đường (dark purple) – loại bỏ vỉa hè và nhiễu nhỏ
            lower_road = np.array([100, 35, 95])
            upper_road = np.array([155, 95, 155])
            mask_road = cv2.inRange(frame, lower_road, upper_road)

            lower_sidewalk = np.array([200, 20, 220])
            upper_sidewalk = np.array([255, 60, 255])
            mask_sidewalk = cv2.inRange(frame, lower_sidewalk, upper_sidewalk)
            mask_road = cv2.bitwise_and(mask_road, cv2.bitwise_not(mask_sidewalk))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, kernel, iterations=1)

            mask_vehicle = cv2.inRange(frame, self.vehicle_color_lower, self.vehicle_color_upper)
            mask_vehicle = cv2.morphologyEx(mask_vehicle, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask_vehicle = cv2.morphologyEx(mask_vehicle, cv2.MORPH_OPEN, kernel, iterations=1)
            vehicle_pixels = cv2.countNonZero(mask_vehicle)
            if vehicle_pixels > 80:
                vehicle_positions = np.column_stack(np.where(mask_vehicle > 0))
                bottom_y = int(np.max(vehicle_positions[:, 0]))
                rel = max(0.0, min(1.0, (h - bottom_y) / float(h)))
                self.camera_obstacle_detected = True
                self.camera_obstacle_distance = float(max(1.0, min(25.0, 1.0 + rel * 24.0)))
            else:
                self.camera_obstacle_detected = False
                self.camera_obstacle_distance = 999.0

            # 3. Lọc nhiễu bằng connected components TRƯỚC khi cắt ROI
            #    (trước đây roi được cắt từ mask_road cũ nên bước lọc này không có
            #     tác dụng gì tới việc tìm mép làn — đã sửa lại thứ tự cho đúng)
            full_area = h * w
            min_component_area = max(120, int(full_area * 0.0015))
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_road, connectivity=8)
            if num_labels > 1:
                filtered_mask = np.zeros_like(mask_road)
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= min_component_area:
                        filtered_mask[labels == i] = 255
                mask_road = filtered_mask
            else:
                mask_road = np.zeros_like(mask_road)

            # 4. ROI ở vùng nhìn xa hơn (giữa-đỉnh ảnh) để giảm cua sớm
            #    Cắt SAU khi mask_road đã được lọc nhiễu ở trên.
            y_center = int(h * self.roi_center_ratio)
            roi_height = max(20, int(h * self.roi_height_ratio))
            y_start = max(0, y_center - roi_height // 2)
            y_end = min(h, y_center + roi_height // 2)
            roi = mask_road[y_start:y_end, :]

            # 5. Tìm mép phải (trung vị)
            right_edges = []
            for row in range(roi.shape[0]):
                road_pixels = np.nonzero(roi[row, :])[0]
                if len(road_pixels) >= 10:
                    right_edges.append(road_pixels[-1])
            if len(right_edges) < 3:
                # giữ giá trị lỗi cũ
                lookahead_error_m = self.filtered_error
            else:
                right_edge = int(np.median(right_edges))

                # 6. Tính chiều rộng làn (pixel) tại lookahead_distance
                width_m = 2.0 * self.lookahead_distance * math.tan(self.fov_hor / 2.0)
                meter_per_pixel = width_m / self.image_width
                lane_width_px = 3.5 / meter_per_pixel

                # 7. Target point: lùi sang trái 1 khoảng tỉ lệ
                target_x = right_edge - (lane_width_px * self.lateral_offset_ratio)

                # 8. Lỗi pixel
                error_px = target_x - (w / 2.0)

                # 9. Đổi sang mét
                lookahead_error_m = error_px * meter_per_pixel

                if abs(lookahead_error_m) < 0.001:
                    lookahead_error_m = 0.0

            # 10. Lọc & publish
            self.filtered_error = self.error_alpha * self.filtered_error + (1 - self.error_alpha) * lookahead_error_m
            msg_out = Float32()
            msg_out.data = self.filtered_error
            self.lookahead_pub.publish(msg_out)

            self.get_logger().info(f"Err: {self.filtered_error:.4f} m", throttle_duration_sec=0.5)

        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")

    def lidar_callback(self, msg):
        try:
            now = time.time()
            if now - self.last_lidar_time < self.min_interval:
                return
            self.last_lidar_time = now

            min_dist_obs = float("inf")
            min_x_left = self.lane_check_dist
            min_x_right = self.lane_check_dist

            counter = 0
            for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                counter += 1
                if counter % 2 != 0:
                    continue

                if z < self.obstacle_min_z:
                    continue

                if x < -self.lane_check_dist or x > self.lane_check_dist:
                    continue

                distance = abs(x)

                # Chỉ tính vật cản PHÍA TRƯỚC xe (x > 0) để tránh xe/vật phía sau
                # trong cùng làn kích hoạt phanh vô lý.
                if x > self.obstacle_min_x and abs(y) < 2.0:
                    if distance < min_dist_obs:
                        min_dist_obs = distance

                if 0.3 < distance < self.lane_check_dist:
                    if self.lane_offset - self.lane_tolerance < y < self.lane_offset + self.lane_tolerance:
                        if distance < min_x_left:
                            min_x_left = distance
                    if -self.lane_offset - self.lane_tolerance < y < -self.lane_offset + self.lane_tolerance:
                        if distance < min_x_right:
                            min_x_right = distance

            if self.camera_obstacle_detected and self.camera_obstacle_distance < min_dist_obs:
                min_dist_obs = self.camera_obstacle_distance

            obs_msg = Bool()
            dist_msg = Float32()
            obs_msg.data = bool(min_dist_obs < 25.0)
            dist_msg.data = float(min_dist_obs) if obs_msg.data else 999.0
            self.obstacle_pub.publish(obs_msg)
            self.obstacle_dist_pub.publish(dist_msg)

            left_clear = Bool()
            right_clear = Bool()
            left_clear.data = bool(min_x_left >= self.clear_threshold)
            right_clear.data = bool(min_x_right >= self.clear_threshold)
            self.lane_left_clear_pub.publish(left_clear)
            self.lane_right_clear_pub.publish(right_clear)

        except Exception as e:
            self.get_logger().error(f"LiDAR error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
