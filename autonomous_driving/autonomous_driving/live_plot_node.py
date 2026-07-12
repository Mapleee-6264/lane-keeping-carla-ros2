import numpy as np
import rclpy
from rclpy.node import Node

import carla

import matplotlib
matplotlib.use("TkAgg")  # cần môi trường có GUI (không chạy được trên máy headless/SSH không X11)
import matplotlib.pyplot as plt


class LivePlotNode(Node):
    """
    Vẽ real-time 2 đường trên cùng 1 đồ thị (tọa độ thế giới CARLA, mét):
      - "Đường tham chiếu": con đường NGOÀI CÙNG bao quanh toàn bộ map (ground-truth
        từ waypoint API của CARLA, không phụ thuộc vị trí xe, không qua perception).
      - "Quỹ đạo thực tế": vị trí (x, y) thật của xe ego, lấy trực tiếp từ actor CARLA,
        cập nhật liên tục theo thời gian.

    LƯU Ý: node này kết nối THẲNG tới CARLA server bằng carla-client (carla.Client),
    KHÔNG đi qua ROS bridge, vì ROS bridge không publish sẵn topic tâm làn/map.
    Vì vậy cần cài package `carla` (Python API) đúng version với CARLA server đang chạy
    cho MÔI TRƯỜNG PYTHON mà ROS2/colcon đang dùng (vd carla==0.9.15).
    """

    def __init__(self):
        super().__init__("live_plot_node")

        self.declare_parameter("carla_host", "localhost")
        self.declare_parameter("carla_port", 2000)
        self.declare_parameter("waypoint_spacing", 3.0)      # m giữa các waypoint quét toàn map (tăng lên để đỡ nặng)
        self.declare_parameter("position_update_hz", 2.0)    # giảm tần suất đọc vị trí để đỡ tải CPU

        host = self.get_parameter("carla_host").value
        port = self.get_parameter("carla_port").value
        spacing = self.get_parameter("waypoint_spacing").value
        hz = self.get_parameter("position_update_hz").value

        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        self.ego_vehicle = self._find_ego_vehicle()
        if self.ego_vehicle is None:
            self.get_logger().error(
                "Không tìm thấy actor có role_name='ego_vehicle' trong world CARLA. "
                "Kiểm tra lại objects.json / launch đã spawn đúng tên chưa."
            )

        self.ref_x, self.ref_y = self._build_outer_boundary_path(spacing)

        self.actual_x = []
        self.actual_y = []

        self.timer = self.create_timer(1.0 / hz, self.update_data)

        self.get_logger().info(
            f"live_plot_node started! Outer boundary reference points: {len(self.ref_x)} "
            f"(spacing={spacing}m)"
        )

    def _find_ego_vehicle(self):
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == "ego_vehicle":
                return actor
        return None

    def _build_outer_boundary_path(self, spacing):
        """
        Dựng đường tham chiếu = con đường NGOÀI CÙNG bao quanh toàn bộ map, bằng
        CONVEX HULL (bao lồi) của toàn bộ waypoint trên map.

        (Lưu ý: cách cũ dùng percentile khoảng cách tới tâm + sort theo góc sẽ méo
        với map hình chữ nhật/đa giác -- các góc map xa tâm hơn hẳn điểm giữa cạnh,
        nên bị lấy dư điểm ở góc và THIẾU điểm dọc theo cạnh thẳng. Convex hull giải
        quyết đúng vấn đề này: luôn ra đúng đa giác bao ngoài thực sự, kể cả góc vuông.)
        """
        waypoints = self.map.generate_waypoints(spacing)
        if not waypoints:
            return [], []

        pts = [(wp.transform.location.x, wp.transform.location.y) for wp in waypoints]
        hull = self._convex_hull(pts)
        if not hull:
            return [], []

        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        # khép kín vòng để vẽ liền mạch
        xs.append(xs[0])
        ys.append(ys[0])
        return xs, ys

    @staticmethod
    def _convex_hull(points):
        """Thuật toán monotone chain (Andrew), O(n log n), không cần scipy."""
        pts = sorted(set(points))
        if len(pts) <= 2:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    def update_data(self):
        if self.ego_vehicle is None:
            return
        loc = self.ego_vehicle.get_location()
        self.actual_x.append(loc.x)
        self.actual_y.append(loc.y)


def main(args=None):
    rclpy.init(args=args)
    node = LivePlotNode()

    fig, ax = plt.subplots(figsize=(8, 8))
    ref_line, = ax.plot([], [], "k--", linewidth=1.5, label="Đường bao ngoài cùng của map")
    actual_line, = ax.plot([], [], "b-", linewidth=2.0, label="Quỹ đạo xe thực tế")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("So sánh quỹ đạo thực tế và đường tham chiếu (tọa độ CARLA world)")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="datalim")

    if node.ref_x and node.ref_y:
        margin = 20.0
        ax.set_xlim(min(node.ref_x) - margin, max(node.ref_x) + margin)
        ax.set_ylim(min(node.ref_y) - margin, max(node.ref_y) + margin)

    ref_line.set_data(node.ref_x, node.ref_y)

    plt.ion()
    plt.show(block=False)

    # QUAN TRỌNG: không dùng thread riêng cho rclpy.spin() nữa.
    # carla.Client không an toàn khi gọi xuyên nhiều thread -> dễ crash ngầm.
    # Ở đây mọi thứ (đọc ROS, gọi CARLA lấy vị trí xe, vẽ matplotlib) chạy
    # tuần tự trong CÙNG 1 thread chính.
    try:
        last_len = 0
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if len(node.actual_x) != last_len:
                actual_line.set_data(node.actual_x, node.actual_y)
                fig.canvas.draw_idle()
                last_len = len(node.actual_x)

            plt.pause(0.1)  # bơm sự kiện GUI (chuột/resize) + nhường CPU, ~10Hz là đủ mượt
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
