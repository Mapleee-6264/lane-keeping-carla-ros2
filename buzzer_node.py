import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import serial
import time


class BuzzerNode(Node):
    def __init__(self):
        super().__init__("buzzer_node")

        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baud_rate", 9600)
        port = self.get_parameter("serial_port").value
        baud = self.get_parameter("baud_rate").value

        self.arduino = None
        try:
            self.arduino = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            self.arduino.reset_input_buffer()
            self.get_logger().info(f"Đã kết nối Arduino tại {port}")
        except serial.SerialException as e:
            self.get_logger().error(f"Không thể kết nối Arduino: {e}")

        # --- Lưu trạng thái mới nhất từ 3 topic ---
        self.obstacle_detected = False
        self.obstacle_type = ""          # "vehicle" | "other" | ""
        self.obstacle_side = "unknown"   # "left" | "right" | "unknown"

        self.create_subscription(Bool, "/obstacle_detected", self.detected_cb, 10)
        self.create_subscription(String, "/obstacle_type", self.type_cb, 10)
        self.create_subscription(String, "/obstacle_vehicle_side", self.side_cb, 10)

        self.last_state = None  # tránh gửi lệnh serial lặp lại

    def detected_cb(self, msg: Bool):
        self.obstacle_detected = msg.data
        self.evaluate()

    def type_cb(self, msg: String):
        self.obstacle_type = msg.data
        self.evaluate()

    def side_cb(self, msg: String):
        self.obstacle_side = msg.data
        self.evaluate()

    def evaluate(self):
        """Quyết định bật/tắt còi dựa trên trạng thái mới nhất của 3 topic."""
        should_alert = False

        if self.obstacle_detected:
            if self.obstacle_type == "other":
                should_alert = True
            elif self.obstacle_type == "vehicle" and self.obstacle_side == "right":
                should_alert = True

        self.send_command(should_alert)

    def send_command(self, state: bool):
        if self.arduino is None:
            return
        if state == self.last_state:
            return  # trạng thái không đổi -> không gửi lại

        self.last_state = state
        try:
            if state:
                self.arduino.write(b"1\n")
                self.get_logger().warn(
                    f"BUZZER ON (type={self.obstacle_type}, side={self.obstacle_side})"
                )
            else:
                self.arduino.write(b"0\n")
                self.get_logger().info("BUZZER OFF")
        except serial.SerialException as e:
            self.get_logger().error(f"Lỗi gửi serial: {e}")

    def destroy_node(self):
        if self.arduino is not None:
            try:
                self.arduino.write(b"0\n")
                self.arduino.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BuzzerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()