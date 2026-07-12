import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32, Bool, String
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
        # Loại vật cản: "vehicle" (xe, nhận qua màu semantic (0,0,142)) hoặc "other" (mọi vật cản khác từ LiDAR)
        self.obstacle_type_pub = self.create_publisher(String, "/obstacle_type", 10)
        self.obstacle_vehicle_side_pub = self.create_publisher(String, "/obstacle_vehicle_side", 10)
        self.lane_left_clear_pub = self.create_publisher(Bool, "/lane_clear_left", 10)
        self.lane_right_clear_pub = self.create_publisher(Bool, "/lane_clear_right", 10)
        # NEW: giới hạn vật lý (mét, cùng hệ quy chiếu với /lookahead_error) của mặt đường
        # (class "Roads" trong CARLA bao phủ TOÀN BỘ phần đường trải nhựa, gồm cả 2 làn),
        # dùng để planning_node kẹp (clamp) offset chuyển làn/vượt, tránh lấn sang vỉa hè.
        self.road_left_edge_pub = self.create_publisher(Float32, "/road_left_edge_error", 10)
        self.road_right_edge_pub = self.create_publisher(Float32, "/road_right_edge_error", 10)

        self.bridge = CvBridge()
        self.fov_hor = math.radians(90.0)
        # lookahead_distance đồng bộ với Ld mới của planning
        self.lookahead_distance = 7.5
        self.roi_center_ratio = 0.40
        self.roi_height_ratio = 0.30
        self.image_width = None

        self.camera_obstacle_detected = False
        self.camera_obstacle_distance = 999.0

        # ==== BẢNG MÀU SEMANTIC SEGMENTATION CHUẨN CỦA CARLA (RGB) ====
        # Vehicles     (0,   0,   142)
        # Roads        (128, 64,  128)
        # Sidewalks    (244, 35,  232)
        # RoadLines    (157, 234, 50)
        # Vegetation   (107, 142, 35)
        # TrafficLight (250, 170, 30)
        # TrafficSigns (220, 220, 0)
        # Sky          (70,  130, 180)
        # Pedestrians  (220, 20,  60)
        #
        # QUAN TRỌNG: ảnh sau imgmsg_to_cv2(..., "passthrough") từ carla_ros_bridge
        # thực chất là BGRA (không phải RGBA), nên phải ép chuyển kênh tường minh
        # về đúng RGB TRƯỚC khi so màu, thay vì tự suy ngược ngưỡng cho từng kênh.
        # Dùng ngưỡng SÁT với giá trị chuẩn (tolerance nhỏ) vì đây là ảnh segmentation,
        # không có nhiễu màu tự nhiên như ảnh thật -- chỉ cần chừa sai số cho nội suy/nén.
        TOL = 10
        def _bounds(rgb):
            r, g, b = rgb
            lower = np.array([max(0, r - TOL), max(0, g - TOL), max(0, b - TOL)])
            upper = np.array([min(255, r + TOL), min(255, g + TOL), min(255, b + TOL)])
            return lower, upper

        self.vehicle_color_lower, self.vehicle_color_upper = _bounds((0, 0, 142))
        self.road_color_lower, self.road_color_upper = _bounds((128, 64, 128))
        self.sidewalk_color_lower, self.sidewalk_color_upper = _bounds((244, 35, 232))

        # LiDAR
        # QUAN TRỌNG: dùng CHUNG một ngưỡng z thấp (-1.5) cho MỌI vật cản phía trước,
        # KHÔNG được để phụ thuộc vào việc camera đã nhận diện xe hay chưa.
        # Lý do: nếu ngưỡng mặc định để 0.0 (ngang/cao hơn độ cao gắn LiDAR trên xe),
        # gần như toàn bộ điểm phản xạ từ nóc một chiếc xe hơi bình thường sẽ có z < 0
        # và bị lọc hết -> nếu camera cũng chưa kịp gắn cờ "vehicle" (xe còn xa, số pixel < ngưỡng...),
        # thì LiDAR lẫn camera đều không thấy gì -> KHÔNG PHANH. Đây chính là lỗi vừa gặp.
        self.obstacle_min_z = -1.5
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

        # NEW: lọc nhẹ cho 2 mép đường, mặc định fallback conservative
        # (~3.5m mỗi bên quanh tâm ảnh, tương đương ~1 làn mỗi phía) trước khi
        # có dữ liệu thật, để không chặn/permissive quá mức lúc khởi động.
        self.filtered_left_edge_m = -3.5
        self.filtered_right_edge_m = 3.5
        self.edge_alpha = 0.4

        self.get_logger().info("Perception (right+left edge, lower-mid ROI, vehicle/other classification) started!")

    def camera_callback(self, msg):
        try:
            now = time.time()
            if now - self.last_camera_time < self.min_interval:
                return
            self.last_camera_time = now

            # 1. Đọc ảnh, ép về đúng RGB tường minh (carla_ros_bridge publish BGRA),
            #    resize bằng INTER_NEAREST để KHÔNG làm hỏng màu rời rạc của
            #    ảnh semantic segmentation (bilinear sẽ tạo màu pha trộn giả ở biên vật thể).
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)
            h, w = frame.shape[:2]

            if self.image_width is None:
                self.image_width = w
                self.get_logger().info(f"Image width: {w}")

            # 2. Mặt đường (Roads, chuẩn CARLA (128,64,128)) – loại bỏ vỉa hè và nhiễu nhỏ
            mask_road = cv2.inRange(frame, self.road_color_lower, self.road_color_upper)

            mask_sidewalk = cv2.inRange(frame, self.sidewalk_color_lower, self.sidewalk_color_upper)
            mask_road = cv2.bitwise_and(mask_road, cv2.bitwise_not(mask_sidewalk))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, kernel, iterations=1)

            # 2b. Vật cản là XE (màu semantic Vehicles = (0,0,142))
            mask_vehicle = cv2.inRange(frame, self.vehicle_color_lower, self.vehicle_color_upper)
            mask_vehicle = cv2.morphologyEx(mask_vehicle, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask_vehicle = cv2.morphologyEx(mask_vehicle, cv2.MORPH_OPEN, kernel, iterations=1)
            vehicle_pixels = cv2.countNonZero(mask_vehicle)
            vehicle_x_center = None
            if vehicle_pixels > 80:
                vehicle_positions = np.column_stack(np.where(mask_vehicle > 0))
                vehicle_x_center = float(np.median(vehicle_positions[:, 1]))
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

            # 5. Tìm mép phải VÀ mép trái (trung vị) của TOÀN BỘ mặt đường trải nhựa
            #    (class "Roads" của CARLA bao phủ cả 2 làn, nên đây chính là ranh giới
            #    vật lý thật giữa đường và vỉa hè hai bên).
            right_edges = []
            left_edges = []
            for row in range(roi.shape[0]):
                road_pixels = np.nonzero(roi[row, :])[0]
                if len(road_pixels) >= 10:
                    right_edges.append(road_pixels[-1])
                    left_edges.append(road_pixels[0])

            if len(right_edges) < 3:
                # giữ giá trị lỗi cũ (không đủ dữ liệu tin cậy trong khung hình này)
                lookahead_error_m = self.filtered_error
                left_edge_error_m = self.filtered_left_edge_m
                right_edge_error_m = self.filtered_right_edge_m
            else:
                right_edge = int(np.median(right_edges))
                left_edge = int(np.median(left_edges))

                # 6. Tính chiều rộng làn (pixel) tại lookahead_distance
                width_m = 2.0 * self.lookahead_distance * math.tan(self.fov_hor / 2.0)
                meter_per_pixel = width_m / self.image_width
                lane_width_px = 3.5 / meter_per_pixel

                # 7. Target point: lùi sang trái 1 khoảng tỉ lệ (điểm bám làn, không phải mép đường)
                target_x = right_edge - (lane_width_px * self.lateral_offset_ratio)

                # 8. Lỗi pixel (điểm bám) -> mét
                error_px = target_x - (w / 2.0)
                lookahead_error_m = error_px * meter_per_pixel
                if abs(lookahead_error_m) < 0.001:
                    lookahead_error_m = 0.0

                # 9. Mép đường trái/phải THẬT SỰ (ranh giới vật lý với vỉa hè), quy đổi sang mét
                #    trong CÙNG hệ quy chiếu với lookahead_error (0 = tâm ảnh/hướng xe).
                left_edge_error_m = (left_edge - (w / 2.0)) * meter_per_pixel
                right_edge_error_m = (right_edge - (w / 2.0)) * meter_per_pixel

            # 10. Lọc & publish điểm bám làn (lookahead_error)
            self.filtered_error = self.error_alpha * self.filtered_error + (1 - self.error_alpha) * lookahead_error_m
            msg_out = Float32()
            msg_out.data = self.filtered_error
            self.lookahead_pub.publish(msg_out)

            # 11. Lọc & publish 2 mép đường (dùng để planning kẹp offset chuyển làn/vượt)
            self.filtered_left_edge_m = self.edge_alpha * self.filtered_left_edge_m + (1 - self.edge_alpha) * left_edge_error_m
            self.filtered_right_edge_m = self.edge_alpha * self.filtered_right_edge_m + (1 - self.edge_alpha) * right_edge_error_m
            left_edge_msg = Float32()
            left_edge_msg.data = self.filtered_left_edge_m
            self.road_left_edge_pub.publish(left_edge_msg)
            right_edge_msg = Float32()
            right_edge_msg.data = self.filtered_right_edge_m
            self.road_right_edge_pub.publish(right_edge_msg)

            vehicle_side = "unknown"
            if self.camera_obstacle_detected and vehicle_x_center is not None:
                road_center = (left_edge + right_edge) / 2.0 if len(right_edges) >= 3 else (w / 2.0)
                vehicle_side = "right" if vehicle_x_center >= road_center else "left"

            side_msg = String()
            side_msg.data = vehicle_side
            self.obstacle_vehicle_side_pub.publish(side_msg)

            self.get_logger().info(
                f"Err: {self.filtered_error:.4f} m | RoadEdges: L={self.filtered_left_edge_m:.2f}m R={self.filtered_right_edge_m:.2f}m | VehicleSide={vehicle_side}",
                throttle_duration_sec=0.5)

        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")

    def lidar_callback(self, msg):
        try:
            now = time.time()
            if now - self.last_lidar_time < self.min_interval:
                return
            self.last_lidar_time = now

            min_dist_obs_lidar = float("inf")
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
                    if distance < min_dist_obs_lidar:
                        min_dist_obs_lidar = distance

                if 0.3 < distance < self.lane_check_dist:
                    if self.lane_offset - self.lane_tolerance < y < self.lane_offset + self.lane_tolerance:
                        if distance < min_x_left:
                            min_x_left = distance
                    if -self.lane_offset - self.lane_tolerance < y < -self.lane_offset + self.lane_tolerance:
                        if distance < min_x_right:
                            min_x_right = distance

            # Hợp nhất khoảng cách vật cản gần nhất giữa camera (xe) và LiDAR (mọi vật)
            min_dist_obs = min_dist_obs_lidar
            if self.camera_obstacle_detected and self.camera_obstacle_distance < min_dist_obs:
                min_dist_obs = self.camera_obstacle_distance

            obs_msg = Bool()
            dist_msg = Float32()
            type_msg = String()
            obs_msg.data = bool(min_dist_obs < 25.0)
            dist_msg.data = float(min_dist_obs) if obs_msg.data else 999.0

            # Phân loại vật cản: nếu camera đang thấy xe -> "vehicle";
            # nếu không nhưng LiDAR vẫn thấy vật cản khác -> "other"; nếu không có gì -> ""
            if not obs_msg.data:
                type_msg.data = ""
            elif self.camera_obstacle_detected:
                type_msg.data = "vehicle"
            else:
                type_msg.data = "other"

            self.obstacle_pub.publish(obs_msg)
            self.obstacle_dist_pub.publish(dist_msg)
            self.obstacle_type_pub.publish(type_msg)

            self.get_logger().info(
                f"Obstacle: detected={obs_msg.data} type='{type_msg.data}' dist={dist_msg.data:.1f}m "
                f"(lidar={min_dist_obs_lidar:.1f}, cam_flag={self.camera_obstacle_detected}, cam_dist={self.camera_obstacle_distance:.1f})",
                throttle_duration_sec=0.5)

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
