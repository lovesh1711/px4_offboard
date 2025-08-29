

__author__ = "Braden Wagstaff"
__contact__ = "braden@arkelectron.com"

import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleLocalPosition   # ✅ local position feedback
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Bool
import time

class OffboardControl(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        # super().__init__('offboard_control')
        self.last_log_time = time.time()
        # self.start_time = self.get_clock().now().nanoseconds / 1e9   # seconds
        # self.logged_once = False   # flag so we only log once

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Subscriptions ────────────────────────────────────────────────────────
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

        self.offboard_velocity_sub = self.create_subscription(
            Twist, '/offboard_velocity_cmd', self.offboard_velocity_callback, qos_profile)

        self.attitude_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_callback, qos_profile)

        self.my_bool_sub = self.create_subscription(
            Bool, '/arm_message', self.arm_message_callback, qos_profile)

        self.local_pos_sub = self.create_subscription(                  # ✅ NEW
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.local_pos_callback, qos_profile)

        # ── Publishers ──────────────────────────────────────────────────────────
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)

        self.publisher_velocity = self.create_publisher(
            Twist, '/fmu/in/setpoint_velocity/cmd_vel_unstamped', qos_profile)

        self.publisher_trajectory = self.create_publisher(              # ✅ back to TrajectorySetpoint
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)

        self.vehicle_command_publisher_ = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10)

        # ── Timers ──────────────────────────────────────────────────────────────
        self.arm_timer_ = self.create_timer(0.1, self.arm_timer_callback)   # ≥ 2Hz
        self.timer = self.create_timer(0.02, self.cmdloop_callback)         # 50Hz

        # ── State variables ─────────────────────────────────────────────────────
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = VehicleStatus.ARMING_STATE_ARMED
        self.velocity = Vector3()
        self.yaw = 0.0
        self.trueYaw = 0.0
        self.offboardMode = False
        self.flightCheck = False
        self.myCnt = 0
        self.arm_message = False
        self.failsafe = False
        self.current_state = "IDLE"
        self.last_state = self.current_state

        # Local position (NED): x (forward), y (right), z (down)
        self.pos_valid_xy = False
        self.pos_valid_z = False
        self.pos_ned = np.array([np.nan, np.nan, np.nan], dtype=float)

        # Mission
        self.mission_stage = "IDLE"  # ASCEND → CRUISE → LAND → DONE
        self.target_ned = np.array([0.0, 0.0, -10.0], dtype=float)  # default first target
        # self.target_v = np.array([0.0, 0.0, 0.0], dtype=float)  # default first target
    
        self.target_reached_once = False  # for debouncing
        self.sent_land = False

        # Thresholds
        self.xy_thresh = 0.5
        self.z_thresh = 0.2

    # ───────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ───────────────────────────────────────────────────────────────────────────

    def arm_message_callback(self, msg):
        self.arm_message = msg.data
        self.get_logger().info(f"Arm Message: {self.arm_message}")

    def arm_timer_callback(self):
        match self.current_state:
            case "IDLE":
                if self.flightCheck and self.arm_message:
                    self.current_state = "ARMING"
                    self.get_logger().info("Arming")

            case "ARMING":
                if not self.flightCheck:
                    self.current_state = "IDLE"
                    self.get_logger().info("Arming, Flight Check Failed")
                elif self.arm_state == VehicleStatus.ARMING_STATE_ARMED and self.myCnt > 10:
                    self.current_state = "TAKEOFF"
                    self.get_logger().info("Arming, Takeoff")
                self.arm()  # keep sending arm

            

            case "TAKEOFF":
                if not self.flightCheck:
                    self.current_state = "IDLE"
                    self.get_logger().info("Takeoff, Flight Check Failed")
                elif self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                    self.current_state = "LOITER"
                    self.get_logger().info("Takeoff, Loiter")
                self.arm()
                self.take_off()  # climb to ~10m by PX4 takeoff

            case "LOITER":
                if not self.flightCheck:
                    self.current_state = "IDLE"
                    self.get_logger().info("Loiter, Flight Check Failed")
                elif self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                    self.current_state = "OFFBOARD"
                    self.get_logger().info("Loiter, Offboard")
                self.arm()

            case "OFFBOARD":
                if (not self.flightCheck) or (self.arm_state != VehicleStatus.ARMING_STATE_ARMED) or self.failsafe:
                    self.current_state = "IDLE"
                    self.get_logger().info("Offboard, Flight Check Failed")
                else:
                    # Prime TrajectorySetpoint before switching to OFFBOARD
                    self.state_offboard()

        if self.arm_state != VehicleStatus.ARMING_STATE_ARMED:
            self.arm_message = False

        if self.last_state != self.current_state:
            self.last_state = self.current_state
            self.get_logger().info(self.current_state)

        self.myCnt += 1

    def state_offboard(self):
        # Initialize mission upon entering OFFBOARD
        if not self.offboardMode:
            self.mission_stage = "ASCEND"
            # status_forward=True
            self.target_ned = np.array([0.0, 0.0, -10.0], dtype=float)
            # self.target_v = np.array([0.0, 0.0, 0.0], dtype=float)
            self.target_reached_once = False
            self.sent_land = False

            # Prime PX4 with trajectory setpoints BEFORE switching mode
            for _ in range(20):  # ~0.4 s at 50 Hz
                # status_forward=True
                self.publish_offboard_mode(position=True, velocity=False)
                # self.publish_trajectory_setpoint(self.target_ned,self.target_v, yaw=0.0)
                self.publish_trajectory_setpoint(self.target_ned, yaw=0.0)

            # Switch to OFFBOARD (MAVLink base mode 1, custom 6)
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1., 6.)
            self.offboardMode = True
            self.get_logger().info("Switched to OFFBOARD and started mission")

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command sent")

    def take_off(self):
        # param7 = target altitude AMSL for GPS; in local sim it still initiates auto takeoff
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param1=1.0, param7=10.0)
        self.get_logger().info("Takeoff command sent")

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.param7 = param7
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher_.publish(msg)

    def vehicle_status_callback(self, msg):
        if msg.nav_state != self.nav_state:
            self.get_logger().info(f"NAV_STATUS: {msg.nav_state}")
        if msg.arming_state != self.arm_state:
            self.get_logger().info(f"ARM STATUS: {msg.arming_state}")
        if msg.failsafe != self.failsafe:
            self.get_logger().info(f"FAILSAFE: {msg.failsafe}")
        if msg.pre_flight_checks_pass != self.flightCheck:
            self.get_logger().info(f"FlightCheck: {msg.pre_flight_checks_pass}")

        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state
        self.failsafe = msg.failsafe
        self.flightCheck = msg.pre_flight_checks_pass

    def local_pos_callback(self, msg: VehicleLocalPosition):
        # validity flags exist; many builds set them implicitly. Use best-effort.
        self.pos_ned[0] = msg.x
        self.pos_ned[1] = msg.y
        self.pos_ned[2] = msg.z
        self.pos_valid_xy = np.isfinite(msg.x) and np.isfinite(msg.y)
        self.pos_valid_z = np.isfinite(msg.z)

  
        now=time.time()
        if now-self.last_log_time>=1.0:
            self.get_logger().info(
                f"Measured velocity: vx={msg.vx:.2f}, vy={msg.vy:.2f}, vz={msg.vz:.2f}"
            )
            self.last_log_time = now

    def offboard_velocity_callback(self, msg):
        # If you ever need velocity-based control; not used in this mission
        self.velocity.x = -msg.linear.y
        self.velocity.y = msg.linear.x
        self.velocity.z = -msg.linear.z
        self.yaw = msg.angular.z

    # def publish_velocity_setpoint(self, vx, vy, vz, yaw=None, yawspeed=None):
    #     msg=TrajectorySetpoint()
    #     msg.timestamp=int(Clock().now().nanoseconds/1000)
    #     msg.position[:]      = [np.nan, np.nan, np.nan]
    #     msg.velocity[:]      = [float(vx), float(vy), float(vz)]
    #     msg.acceleration[:]  = [np.nan, np.nan, np.nan]
    #     msg.jerk[:]          = [np.nan, np.nan, np.nan]
    #     if yawspeed is not None:
    #         msg.yaw=np.nan
    #         msg.yawspeed=float(yawspeed)
    #     else:
    #         msg.yaw=float(yaw) if yaw is not None else 0.0
    #         msg.yawspeed=np.nan
    #     self.publisher_trajectory.publish(msg)
        

    def attitude_callback(self, msg):
        q = msg.q
        self.trueYaw = -(np.arctan2(2.0*(q[3]*q[0] + q[1]*q[2]),
                                    1.0 - 2.0*(q[0]*q[0] + q[1]*q[1])))

    def publish_twist_velocity(self, vx, vy, vz, yawrate=0.0):
        msg=Twist()
        msg.linear.x=vx
        msg.linear.y=vy
        msg.linear.z=vz
        msg.angular.z=yawrate
        self.publisher_velocity.publish(msg)

    # ───────────────────────────────────────────────────────────────────────────
    # Command loop: send setpoints + mission FSM 
    # ───────────────────────────────────────────────────────────────────────────
    def cmdloop_callback(self):

        if not self.offboardMode:
            return

        # Always publish OffboardControlMode at control rate
        self.publish_offboard_mode(position=True)

        # self.publish_mode_velocity()
        # vx, vy, vz = self.velocity.x, self.velocity.y, self.velocity.z
        # yawrate     = self.yaw

        # If we don't have a valid position yet, just hold latest target
        if not (self.pos_valid_xy and self.pos_valid_z):
            # self.publish_trajectory_setpoint(self.target_ned, self.target_v, yaw=0.0)
            self.publish_trajectory_setpoint(self.target_ned, yaw=0.0)
            return
        
    

        px, py, pz = self.pos_ned
        tx, ty, tz = self.target_ned

        dx = tx - px
        dy = ty - py
        dz = tz - pz
        dist_xy = np.hypot(dx, dy)
        dist_z = abs(dz)

        # self.get_logger().info("Velocity", self.velocity.x)
        # ── Mission stages ─────────────────────────────────────────────────────
        if self.mission_stage == "ASCEND":
            
            # target: (0, 0, -15)
            
            self.target_ned[:] = [0.0, 0.0, -10.0]
            
            if dist_xy < self.xy_thresh and dist_z < self.z_thresh:
                if not self.target_reached_once:
                    # small debounce to avoid flicker at threshold
                    self.target_reached_once = True
                else:
                    self.mission_stage = "CRUISE"
                    self.target_reached_once = False
                    self.get_logger().info("Reached (0,0,-15). Proceeding to (10,10,-15).")
                   
            else:
                self.target_reached_once = False

        elif self.mission_stage == "CRUISE":
            # target: (10, 10, -15)
            self.target_ned[:] = [10.0, 0.0, -10.0]
            
           
            if dist_xy < self.xy_thresh and dist_z < self.z_thresh:
                if not self.target_reached_once:
                    self.target_reached_once = True
                else:
                    self.mission_stage = "DESCEND"
                    self.target_reached_once = False
                    self.get_logger().info("Reached (10,10,-15). descending to 3m.")
                    
            else:
                self.target_reached_once = False

        elif self.mission_stage == "DESCEND":
            self.target_ned[:] = [10.0, 0.0, -3.0]
            # self.target_v[:]=[0.0,0.0,2.0]
            if dist_xy < self.xy_thresh and dist_z < self.z_thresh:
                if not self.target_reached_once:
                    self.target_reached_once = True
                else:
                    self.mission_stage = "WAIT"
                    self.target_reached_once = False
                    self.wait_start_time = self.get_clock().now().nanoseconds / 1e9
                    self.get_logger().info("Reached (10,10,-3). Holding for 3 seconds.")
            else:
                self.target_reached_once = False

        elif self.mission_stage == "WAIT":
            # Hold position at (10, 10, -3)
            # self.get_logger().info("wait command sent")
            self.target_ned[:] = [10.0, 0.0, -3.0]
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.wait_start_time
            if elapsed >= 5.0:
                self.mission_stage = "ASCEND_back"
                self.target_reached_once = False
                self.get_logger().info("Wait complete. Ascending back to (10,10,-15).")

        elif self.mission_stage == "ASCEND_back":
            
            # target: (0, 0, -15)
            self.target_ned[:] = [10.0, 0.0, -10.0]
            # self.target_v[:]=[0.0,0.0,-2]
            if dist_xy < self.xy_thresh and dist_z < self.z_thresh:
                if not self.target_reached_once:
                    # small debounce to avoid flicker at threshold
                    self.target_reached_once = True
                else:
                    self.mission_stage = "CRUISE_back"
                    self.target_reached_once = False
                    self.get_logger().info("Reached (10,10,-15). cruising back to (0,0,-15).")
            else:
                self.target_reached_once = False

        elif self.mission_stage == "CRUISE_back":
            # target: (10, 10, -15)
            self.target_ned[:] = [0.0, 0.0, -10.0]
            # self.target_v[:]=[5.0,5.0,0.0]
            if dist_xy < self.xy_thresh and dist_z < self.z_thresh:
                if not self.target_reached_once:
                    self.target_reached_once = True
                else:
                    self.mission_stage = "LAND"
                    self.target_reached_once = False
                    self.get_logger().info("Reached (0,0,-15).landing command sent")
            else:
                self.target_reached_once = False

        

        elif self.mission_stage == "LAND":
            if not self.sent_land:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.sent_land = True
                self.get_logger().info("Landing command sent.")
            # keep target steady until PX4 switches modes
            self.target_ned[:] = [self.pos_ned[0], self.pos_ned[1], self.pos_ned[2]]
            
        

        # Publish the current target as TrajectorySetpoint
        # self.publish_trajectory_setpoint(self.target_ned, self.target_v, yaw=0.0)
        self.publish_trajectory_setpoint(self.target_ned, yaw=0.0)

    # ───────────────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────────────
    def publish_offboard_mode(self, position=True, velocity=False, acceleration=False):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = position
        offboard_msg.velocity = velocity
        offboard_msg.acceleration = acceleration
        self.publisher_offboard_mode.publish(offboard_msg)
    
   

    def publish_trajectory_setpoint(self, target_ned, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.position[0] = float(target_ned[0])  # x (N)
        msg.position[1] = float(target_ned[1])  # y (E)
        msg.position[2] = float(target_ned[2])  # z (Down, negative = up)

        # msg.velocity[0] = float(target_v[0])  # vx (N)
        # msg.velocity[1] = float(target_v[1])  # vy (E)
        # msg.velocity[2] = float(target_v[2])  # vz (Down, negative = up)

        msg.yaw = float(yaw)
        self.publisher_trajectory.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
