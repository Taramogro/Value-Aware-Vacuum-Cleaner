# === For all nodes ===
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data, qos_profile_services_default

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rcl_interfaces.msg import Log
from irobot_create_msgs.msg import HazardDetectionVector, HazardDetection, IrIntensityVector, DockStatus
from nav_msgs.msg import Odometry

from std_srvs.srv import Trigger

# === For specific nodes ===
import time
import random
import threading
import paramiko
import numpy as np
import os
import cv2
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
import math

# === For application node ===
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QStackedWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QTextCursor, QPixmap, QImage
import spacy

# === For LLM node ===
import openai
import base64
from dotenv import load_dotenv
from datetime import datetime, timedelta
import re

# === Initialize rclpy ===
rclpy.init()

# === Global use ===
ssh = paramiko.SSHClient()
shutdown_called = False
reentrant_callback_group = ReentrantCallbackGroup()

navigator = TurtleBot4Navigator()
move_cmd = Twist()
turn_cmd = Twist()
stop_cmd = Twist()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")
client = openai.OpenAI(api_key=api_key)

script_dir = os.path.dirname(os.path.abspath(__file__))
image_extensions = {'.jpg'}

###Signal emmiter used for communication between ROS2 and Qt###
class AppSignalEmitter(QObject):
    ###Signals send by multiple Nodes###
    # === Signal for transition locking and unlocking ===
    transition_lock = pyqtSignal()
    unlock_transition_lock = pyqtSignal()

    ###Signals send from RobotControlNode###
    # === Signal for updating status label ===
    dock_status_updated = pyqtSignal(bool)
    # === Signal for retrying action if undock was done instead ===
    retry_action = pyqtSignal()

    ###Signals send from Artificial_Intelligence_Node###
    start_iteration_thoughts_field = pyqtSignal(int)
    update_thoughts_field = pyqtSignal(str)
    thought_field_segment = pyqtSignal()
    endline_for_thtsf = pyqtSignal()
    # === Signal for final decision field update ===
    update_final_decision_field = pyqtSignal(str)
    clear_final_decision_field_if_not_empty = pyqtSignal()
    # === Signals for checkup field update ===
    start_iteration_checkup_field = pyqtSignal(int)
    update_checkup_field = pyqtSignal(str)
    checkup_field_segment = pyqtSignal()
    mode_action = pyqtSignal(str)
    summary_action = pyqtSignal()
    endline_for_chupf = pyqtSignal()
    # === Signal connect for current image print ===
    update_image = pyqtSignal(str)
    camera_intake = pyqtSignal(Image)
    # === Signals for AI continuation based on chosen mode ===
    observation_choice = pyqtSignal()
    cleaning_choice = pyqtSignal()
    docking_choice = pyqtSignal()

    def __init__(self):
        super().__init__()

###RobotControlNode###
class RobotControlNode(Node):
    def __init__(self, signal_emitter: AppSignalEmitter, node_name="robot_control_node"):
        super().__init__(node_name)

        self.signal_emitter = signal_emitter
        self.node_name = node_name

        # === used for keeping track of docking status ===
        self.docking_status = None
        self.create_subscription(
            DockStatus,
            'dock_status',
            self.dock_status_callback,
            qos_profile_sensor_data,
            callback_group=reentrant_callback_group
        )

        # === Basic naviagtor subscriber ===
        self.rosout_subscription = self.create_subscription(
            Log,
            '/rosout',
            self.basic_nav_callback,
            qos_profile_sensor_data,
            callback_group=reentrant_callback_group
        )

        # === flag to state that robot is in a transition ===
        self.in_transition = False

        # === Tracks current mode the robot is in ===
        self.current_state = None

        # === For robot movement fluidety ===
        self.time_wait = 0.2

        # === Checks if a new state call was received ===
        self.state_received = False

        ###Subscription, Publishers, and values for movement functionality###
        self.mode_subsriber = self.create_subscription(
            String,
            '/mode',
            self.mode_callback,
            qos_profile_sensor_data,
            callback_group=reentrant_callback_group
        )
        # === Cleaning mode ===
        # === Publisher for the cmd_vel topic used for driving the robot and bool to check if robot is or isn't driving ===
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            qos_profile_services_default,
            callback_group=reentrant_callback_group
        )
        self.is_cleaning = False
        # === Subscription for the hazard_detection topic and bool to track if the robot bumped into something ===
        self.hazard_detection_subscription = self.create_subscription(
            HazardDetectionVector,  
            '/hazard_detection',
            self.hazard_detection_callback,
            qos_profile_sensor_data,
            callback_group=reentrant_callback_group
        )
        self.bumped_into_something = False
        # === Subscription for ir_intensity to check if something is in front of the robot and bools to keep track of its location ===
        self.ir_intensity_subscription = self.create_subscription(
            IrIntensityVector,
            '/ir_intensity',
            self.ir_intensity_callback,
            qos_profile_sensor_data,
            callback_group=reentrant_callback_group
        )
        self.fcl_close_to_object = False
        self.fcr_close_to_object = False
        self.fl_close_to_object = False 
        self.fr_close_to_object = False 
        self.l_close_to_object = False
        self.r_close_to_object = False

        # === Observation mode ===
        self.is_observing = False

        self.ir_intensity_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data,
            callback_group=reentrant_callback_group
        )
        self.degrees = None

    # === Checks current robots status on wether is or isn't docked ===
    def dock_status_callback(self, msg: DockStatus):
        if not self.in_transition:
            self.docking_status = msg.is_docked
            self.signal_emitter.dock_status_updated.emit(self.docking_status)
            #GuiLogger.instance().log(f"Is docked: {self.docking_status}", "debug_field")
    # === Checks when a docking or undocking action has been completed ===
    def basic_nav_callback(self, msg: Log):
        if msg.name == "basic_navigator":
            if msg.msg == "Dock succeeded":
                self.in_transition = False
                self.signal_emitter.unlock_transition_lock.emit()
                GuiLogger.instance().log("Docking complete", "debug_field")
            if msg.msg == "Undock succeeded":
                self.in_transition = False
                self.signal_emitter.unlock_transition_lock.emit()
                self.signal_emitter.retry_action.emit()
                GuiLogger.instance().log("Undocking complete", "debug_field")

    # === Called to make the robot dock ===
    def nav_dock(self):
        if not self.in_transition:
            self.in_transition = True
            self.current_state = "Docking mode"
            GuiLogger.instance().log(f"Docking...", "debug_field")
            threading.Thread(target=self.dock_worker, daemon=True).start()
    def dock_worker(self):
        try:
            self.signal_emitter.transition_lock.emit()
            navigator.dock()
        except Exception as e:
            GuiLogger.instance().log(f"Docking error: {e}", "debug_field")
            self.in_transition = False

    # === Called to make the robot undock ===
    def nav_undock(self):
        if not self.in_transition:
            self.in_transition = True
            self.current_state = "Undocking state"
            GuiLogger.instance().log(f"Undocking...", "debug_field")
            threading.Thread(target=self.undock_worker, daemon=True).start()
    def undock_worker(self):
        try:
            self.signal_emitter.transition_lock.emit()
            navigator.undock()
        except Exception as e:
            GuiLogger.instance().log(f"Undocking error: {e}", "debug_field")
            self.in_transition = False

    def mode_callback(self, msg: String):
        self.mode_selection(msg.data)

    # === Function for Qt app where the buttons change the current state ===
    def mode_selection(self, state):
        if not self.in_transition:
            GuiLogger.instance().log(f"Changed to → {state}", "debug_field")
            self.current_state = state
            self.state_received = True

            start_mode = True

            if self.current_state == "Observation mode":
                self.start_observation_sequence(start_mode)
            if self.current_state == "Cleaning mode":
                self.start_cleaning_sequence(start_mode)
            if self.current_state == "Docking mode":
                self.nav_dock()

    # === Cleaning mode ===
    # === Function to start driving sequence ===
    def start_cleaning_sequence(self, start_driving):
        if not self.in_transition:
            self.is_cleaning = start_driving
            threading.Thread(target=self.Play_startup_sound_async, args=("/usr/share/sounds/alsa/StartUp-Vacuum-Cleaner.wav",), daemon=True).start()
            threading.Thread(target=self.is_cleaning_sound_check, daemon=True).start()
            threading.Thread(target=self.cleaning_sequence, daemon=True).start()
    # === Function for driving capability ===
    def cleaning_sequence(self):
        GuiLogger.instance().log(f"Cleaning...", "debug_field")

        while self.is_cleaning:
            if self.current_state != "Cleaning mode":
                self.stop_cleaning()
                return
            
            if self.bumped_into_something:
                self.turn_around()
                self.bumped_into_something = False

            if (self.fcl_close_to_object or self.fcr_close_to_object or
                self.fl_close_to_object or self.fr_close_to_object or
                self.l_close_to_object or self.r_close_to_object):
                move_cmd.linear.x = 0.05
            else:
                move_cmd.linear.x = 0.15
            
            self.cmd_vel_publisher.publish(move_cmd)
            time.sleep(self.time_wait)
    # === Function that stops the robot from driving ===
    def stop_cleaning(self):
        self.is_cleaning = False
        self.stop_movement()
    # === callback function for bumper triggers at the front of the robot ===
    def hazard_detection_callback(self, msg: HazardDetectionVector):
        if self.current_state == "Cleaning mode":
            bumper = HazardDetection.BUMP
            detection = HazardDetection()

            if self.current_state != "Cleaning mode":
                self.stop_cleaning()
                return

            for detection in msg.detections:
                detection_type = detection.type

                if detection_type == bumper:
                    #self.get_logger().info('Robot bumped into something!')
                    self.bumped_into_something = True
                    return
    # === callback function for sensors at the front of the robot inside the bumber ===
    def ir_intensity_callback(self, msg: IrIntensityVector):
        if self.current_state == "Cleaning mode":
            for ir_value in msg.readings:
                #self.get_logger().info(f'{ir_value.header.frame_id}')

                if self.current_state != "Cleaning mode":
                    self.stop_cleaning()
                    return

                if ir_value.header.frame_id == "ir_intensity_front_center_left":
                    if ir_value.value >= 600:
                        self.fcl_close_to_object = True 
                        #self.get_logger().info('Robot is close to an object')
                        #self.get_logger().info(f'{ir_value.header.frame_id}')
                        #self.get_logger().info(f'{ir_value.value}')
                        continue

                    self.fcl_close_to_object = False

                if ir_value.header.frame_id == "ir_intensity_front_center_right":
                    if ir_value.value >= 600:
                        self.fcr_close_to_object = True 
                        #self.get_logger().info('Robot is close to an object')
                        #self.get_logger().info(f'{ir_value.header.frame_id}')
                        #self.get_logger().info(f'{ir_value.value}')
                        continue
                        
                    self.fcr_close_to_object = False

                if ir_value.header.frame_id == "ir_intensity_front_left":
                    if ir_value.value >= 600:
                        self.fl_close_to_object = True 
                        #self.get_logger().info('Robot is close to an object')
                        #self.get_logger().info(f'{ir_value.header.frame_id}')
                        #self.get_logger().info(f'{ir_value.value}')
                        continue

                    self.fl_close_to_object = False

                if ir_value.header.frame_id == "ir_intensity_front_right":
                    if ir_value.value >= 600:
                        self.fr_close_to_object = True 
                        #self.get_logger().info('Robot is close to an object')
                        #self.get_logger().info(f'{ir_value.header.frame_id}')
                        #self.get_logger().info(f'{ir_value.value}')
                        continue
                        
                    self.fr_close_to_object = False

                if ir_value.header.frame_id == "ir_intensity_left":
                    if ir_value.value >= 600:
                        self.l_close_to_object = True 
                        #self.get_logger().info('Robot is close to an object')
                        #self.get_logger().info(f'{ir_value.header.frame_id}')
                        #self.get_logger().info(f'{ir_value.value}')
                        continue

                    self.l_close_to_object = False

                if ir_value.header.frame_id == "ir_intensity_right":
                    if ir_value.value >= 600:
                        self.r_close_to_object = True 
                        #self.get_logger().info('Robot is close to an object')
                        #self.get_logger().info(f'{ir_value.header.frame_id}')
                        #self.get_logger().info(f'{ir_value.value}')
                        continue
                        
                    self.r_close_to_object = False
    # === Function that stops the robot from moving ===
    def stop_movement(self):
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0 

        self.cmd_vel_publisher.publish(stop_cmd)
        #self.get_logger().info('Stopping the robot.')
    # === Function called to make the robot turn around (normally called wehn bumping into something) ===
    def turn_around(self):
        move_cmd.linear.x = -0.1
        for i in range(0,5):
            if self.current_state != "Cleaning mode":
                self.stop_cleaning()
                return
            self.cmd_vel_publisher.publish(move_cmd)
            time.sleep(self.time_wait)

        r_int = random.randint(-10,10)
        r_float = float(r_int)
        turn_cmd.angular.z = r_float
        for i in range(5):
            if self.current_state != "Cleaning mode":
                self.stop_cleaning()
                return
            self.cmd_vel_publisher.publish(turn_cmd)
            time.sleep(self.time_wait)

    # === Sound player ===
    def play_sound(self, sound_file):
        command = f'aplay "{sound_file}" &'
        stdin, stdout, stderr = ssh.exec_command(command)
        
        exit_status = stdout.channel.recv_exit_status()

        if exit_status != 0:
            error_output = stderr.read().decode()
            GuiLogger.instance().log(f"Sound playback error: {error_output}", "error")
    # === Play vacuum cleaner startup sound ===
    def Play_startup_sound_async(self, sound_file):
        GuiLogger.instance().log(f"Playing startup sound...", "debug_field")
        self.play_sound(sound_file)
        GuiLogger.instance().log(f"Finished playing startup sound", "debug_field")
        time.sleep(1)

        threading.Thread(target=self.play_driving_sound_async, args=("/usr/share/sounds/alsa/Vacuum-Cleaner-driving.wav",), daemon=True).start()
    # === Play vacuum cleaner driving sound ===
    def play_driving_sound_async(self, sound_file):
        GuiLogger.instance().log(f"Playing driving sound...", "debug_field")
        while self.is_cleaning:
            if self.current_state == "Cleaning mode":
                self.play_sound(sound_file)
            time.sleep(self.time_wait)
        GuiLogger.instance().log(f"Finished playing driving sound", "debug_field")
        time.sleep(self.time_wait)

        threading.Thread(target=self.play_stop_sound_async, args=("/usr/share/sounds/alsa/TurnOff-Vacuum-Cleaner.wav",), daemon=True).start()
    # === Play vacuum cleaner stopping sound ===
    def play_stop_sound_async(self, sound_file):
        GuiLogger.instance().log(f"Playing stopping sound...", "debug_field")
        self.play_sound(sound_file)
        GuiLogger.instance().log(f"Finished playing stopping sound", "debug_field")
    # === Thread to stop driving sound whenever cleaning stops ===
    def is_cleaning_sound_check(self):
        while True:
            time.sleep(0.2)
            if not self.is_cleaning:
                self.stop_current_sound()
                return
    # === Function to kill the current sound that is playing ===
    def stop_current_sound(self):
        GuiLogger.instance().log(f"killing all aplay currently running", "debug_field")
        ssh.exec_command("pkill -f aplay")

    # === Observation mode ===
    # === Function to start observation sequence ===
    def start_observation_sequence(self, start_observation):
        if not self.in_transition:
            self.is_observing = start_observation
            threading.Thread(target=self.observation_sequence, daemon=True).start()
    # === Function for observation capability ===
    def observation_sequence(self):
        last_log_time = time.time()
        turn_cmd = Twist()
        GuiLogger.instance().log(f"Observing...", "debug_field")

        while self.current_state == "Observation mode":
            self.state_received = False
            while self.is_observing:
                if self.current_state != "Observation mode":
                    self.stop_observation()
                    return

                start_pos = self.degrees
                left_pos = (start_pos + 85) % 360
                right_pos = (start_pos - 85) % 360

                self.stop_movement()

                turn_cmd.linear.x = 0.0
                turn_cmd.angular.z = 0.2
                while True:
                    #GuiLogger.instance().log(f"Degrees: {self.degrees:.2f}°, goal: {left_pos:.2f}°, difference: {(left_pos - self.degrees):.2f}", "debug_field")
                    if -5 <= (left_pos - self.degrees) <= 0:
                        break
                    if self.current_state != "Observation mode":
                        self.stop_observation()
                        return
                    self.cmd_vel_publisher.publish(turn_cmd)
                    time.sleep(self.time_wait)

                self.stop_movement()
                
                turn_cmd.linear.x = 0.0
                turn_cmd.angular.z = -0.2
                while True:
                    #GuiLogger.instance().log(f"Degrees: {self.degrees:.2f}°, goal: {right_pos:.2f}°, difference: {(right_pos - self.degrees):.2f}", "debug_field")
                    if 0 <= (right_pos - self.degrees) <= 5:
                        break
                    if self.current_state != "Observation mode":
                        self.stop_observation()
                        return
                    self.cmd_vel_publisher.publish(turn_cmd)
                    time.sleep(self.time_wait)

                self.stop_movement()

                turn_cmd.linear.x = 0.0
                turn_cmd.angular.z = 0.2
                while True:
                    #GuiLogger.instance().log(f"Degrees: {self.degrees:.2f}°, goal: {start_pos:.2f}°, difference: {(start_pos - self.degrees):.2f}", "debug_field")
                    if -5 <= (start_pos - self.degrees) <= 0:
                        break
                    if self.current_state != "Observation mode":
                        self.stop_observation()
                        return
                    self.cmd_vel_publisher.publish(turn_cmd)
                    time.sleep(self.time_wait)

                self.stop_movement()

                while not self.state_received:
                    if self.state_received:
                        self.stop_observation()
                        return
                    time.sleep(self.time_wait)
    # === callback function for keeping track of robot position ===
    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = self.get_yaw_from_quaternion(q)

        self.degrees = math.degrees(yaw) % 360

        #GuiLogger.instance().log(f"Yaw: {yaw:.2f} rad", "debug_field")
        #GuiLogger.instance().log(f"Degrees: {self.degrees:.2f}°", "debug_field")
    # === function to calculate the yaw ===
    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    # === Calculates the differnce for turning always the correct angle ===
    def angle_diff(self, current, target):
        return (target - current + 360) % 360
    # === Function that stops the robot from observing ===
    def stop_observation(self):
        self.is_observing = False
        self.stop_movement()

###CameraNode###
class CameraNode(Node):   
    def __init__(self, signal_emitter: AppSignalEmitter, node_name="camera_node"):
        super().__init__(node_name)

        self.signal_emitter = signal_emitter
        self.node_name = node_name

        # === Subscriber to camera on robot ===
        self.camera_subscription = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.camera_callback,
            qos_profile_sensor_data,
            callback_group=reentrant_callback_group
        )
        # === Variables for camera intake ===
        # flag that checks if an image was received
        self.image_received = False
        # counter that checks how long it has been since a image was received
        self.last_image_time = None

        # === Publisher for camera footage ===
        self.camera_publisher = self.create_publisher(
            Image,
            '/Camera_footage',
            qos_profile_services_default,
            callback_group=reentrant_callback_group
        )

        self.timer = self.create_timer(1, self.check_camera_status)

    # === Callback from subscription of robot's camera ===
    def camera_callback(self, msg: Image):
        self.image_received = True
        self.last_image_time = time.monotonic()
        #GuiLogger.instance().log(f"Image, Wigth: {msg.width} Height: {msg.height}", "debug_field")

        self.camera_publisher.publish(msg)
        self.signal_emitter.camera_intake.emit(msg)
    
    ###Functions for camera condition and restarting###
    # === Checks the status of the camera ===
    def check_camera_status(self):
        now = time.monotonic()

        if self.last_image_time is None:
            GuiLogger.instance().log("No image received yet, attempting to start the camera...", "debug_field")
            self.start_camera_service()
        elif now - self.last_image_time > 3.0:
            GuiLogger.instance().log("Camera image stream stopped, restarting camera...", "debug_field")
            self.image_received = False
            self.last_image_time = None
            self.start_camera_service()
    # === starts the camera if not on ===
    def start_camera_service(self):
        client = self.create_client(Trigger, '/oakd/start_camera')

        if not client.wait_for_service(timeout_sec=1.0):
            GuiLogger.instance().log(f"/oakd/start_camera service not available, waiting for next check...", "debug_field")
            return

        req = Trigger.Request()
        future = client.call_async(req)
        future.add_done_callback(self.camera_service_response)
    # === Response for whether the camera is working ===
    def camera_service_response(self, future):
        try:
            response = future.result()
            if response.success:
                GuiLogger.instance().log(f"Camera started successfully: {response.message}", "debug_field")
            else:
                GuiLogger.instance().log(f"Failed to start camera: {response.message}", "debug_field")
        except Exception as e:
            GuiLogger.instance().log(f"Service call failed: {e}", "debug_field")

###ArtificialIntelligenceNode###
class ArtificialIntelligenceNode(Node):
    def __init__(self, signal_emitter: AppSignalEmitter, node_name="artificial_intelligence_node"):
        super().__init__(node_name)

        self.signal_emitter = signal_emitter
        self.node_name = node_name

        # === used to help extract mode ===
        self.nlp = spacy.load("en_core_web_sm")

        ###Set-up and variables for image intake of dataset###
        # === Array to store all img folders ===
        self.activity_folders = []
        # === value for immage folder we will start with ===
        self.current_folder = 1

        # === Getting the image folders ===
        self.folder_dir = "Self_Made_Dataset"
        self.data_dir = os.path.join(script_dir, self.folder_dir)

        def list_folders_with_images(base_dir):
            folders = []
            
            for dirpath, dirnames, filenames in os.walk(base_dir):
                if any(f.lower().endswith(tuple(image_extensions)) for f in filenames):
                    folders.append(dirpath)
            
            return folders
        self.activity_folders = list_folders_with_images(self.data_dir)
        self.activity_folders = sorted(self.activity_folders)


        ###Set-up and variables for image intake of camera###
        #self.all_frames = [] <- could be used to save every frame to show the models entire process of decision making in real-time
        self.temp_frames = []
        self.current_frames = []
        self.current_frames_time = {}

        self.frames_folder = "Frames"
        self.frames_folder_path = os.path.join(script_dir, self.frames_folder)
        os.makedirs(self.frames_folder_path, exist_ok=True)

        ###Set-up and variables for contexts used for model knowledge base###
        self.model_use_dir = "Model_Use/"

        self.context_file = "Context_17.txt"
        self.context_dir = "Contexts/"
        self.context_dir_path = os.path.join(self.model_use_dir, self.context_dir)
        self.combine_context_file = os.path.join(self.context_dir_path, self.context_file)
        self.context_file_path = os.path.join(script_dir, self.combine_context_file)
        with open(f'{self.context_file_path}', 'r') as file:
            self.function_context = file.read()

        self.action_reflection_file = "Action_reflection_4.txt"
        self.action_reflection_dir = os.path.join(self.model_use_dir, self.action_reflection_file)
        self.action_reflection_file_path = os.path.join(script_dir, self.action_reflection_dir)
        with open(f'{self.action_reflection_file_path}', 'r', encoding='utf-8') as file:
            self.action_reflection = file.read()

        # === Keypoint files ===
        self.keypoints_contextt_dir = "Keypoint_context/"
        self.keypoints_context_dir_path = os.path.join(self.model_use_dir, self.keypoints_contextt_dir)

        self.keypoints_context = "Keypoint_context.txt"
        self.keypoints_context_file = os.path.join(self.keypoints_context_dir_path, self.keypoints_context)
        self.keypoints_context_file_path = os.path.join(script_dir, self.keypoints_context_file)
        with open(f'{self.keypoints_context_file_path}', 'r', encoding='utf-8') as file:
            self.keypoint_extraction_context = file.read()

        # === cleaning check files ===
        self.check_context_dir = "Cleaning_check_context/"
        self.check_context_dir_path = os.path.join(self.model_use_dir, self.check_context_dir)

        self.check_context = "Check_context.txt"
        self.check_context_file = os.path.join(self.check_context_dir_path, self.check_context)
        self.check_context_file_path = os.path.join(script_dir, self.check_context_file)
        with open(f'{self.check_context_file_path}', 'r', encoding='utf-8') as file:
            self.cleaning_check_context = file.read()

        ###Main GPT process variables###
        # === Variables to remember the GPT pages and which one is connected and where output needs to be written down ===
        self.gpt_page_names = {
            1: "GPT Page",
            2: "Real-time GPT Page"
        }
        self.connected_page = None

        # === Version of value-aware process the node currently is in
        self.version_choices_names = {
            1: "Observation version",
            2: "Cleaning version"
        }
        self.version_choice = None

        # === Context variables ===
        self.time_context = None

        self.observation_length_context = "00:00:00"
        self.start_time_observation = None
        self.current_time_observation = None

        # === Variable to track what iteration the value-aware process is in on the same page ===
        self.iteration = 1

        # === Conversation and thinking variables for remembering thinking process ===
        self.conversation_history = []
        self.check_context= []
        # Variables that store model output data
        self.keywords = []
        self.thought_process = ""
        self.final_decision = ""
        self.summarization_decision = ""

        # === Variables that save image data ===
        self.image_folder_path = None
        self.image_folder = None
        self.base64_image = None

        # === Variables to indicate the mode the robot ===
        self.current_mode = None
        self.previous_mode = None

        # === Variable to indicate the mode the robot is in ===
        self.cleaning_length = "00:10:00"
        self.time_past_cleaning = 0

        # === Variable to track if value-aware process is running ===
        self.is_process_running = False

        # === Variables for camera intake ===
        self.camera_subscription = self.create_subscription(
            Image,
            '/Camera_footage',
            self.camera_footge_callback,
            qos_profile_sensor_data,
            callback_group=reentrant_callback_group
        )
        self.allow_camera_intake = None

        self.mode_publisher = self.create_publisher(
            String,
            '/mode',
            qos_profile_sensor_data,
            callback_group=reentrant_callback_group
        )

    ###General functions###
    # === Function to get the current image folder ===
    def get_image_folder(self):
        if self.connected_page == self.gpt_page_names[1]:
            self.image_folder_path = self.activity_folders[self.current_folder]
            print(self.image_folder_path)
        if self.connected_page == self.gpt_page_names[2]:
            self.image_folder_path = self.frames_folder_path
            print(self.image_folder_path)
        #GuiLogger.instance().log(f"folder being used is: {self.image_folder_path}", "debug_field")

        self.image_folder = os.listdir(self.image_folder_path)
        #GuiLogger.instance().log(f"Before sorting img folder: {self.image_folder}", "debug_field")
        self.image_folder = sorted(self.image_folder, key=lambda x: int(os.path.splitext(x)[0]))
        #GuiLogger.instance().log(f"after sorting img folder: {self.image_folder}", "debug_field")
    # === Function to turn image paths to a base64 ===
    def image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    # === Function that keeps up the current date & time ===
    def time_upkeep(self):
        while self.is_process_running:
            self.current_datetime = datetime.now()
    # === extract the keywords from the tokens saved in the keyword array ===
    def get_keywords(self):
        points = []
        current_point = []

        for keyword in self.keywords:
            parts = keyword.split('\n')
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if self.unwanted_character(part):
                    if current_point:
                        points.append(' '.join(current_point).strip())
                        current_point = []
                    continue
                current_point.append(part)

        if current_point:
            points.append(' '.join(current_point).strip())

        self.keywords = points
        points = []
        current_point = []
    # === makes sure unwanted characters are not stored in the keyword array ===
    def unwanted_character(self, keyword):
        if keyword.strip() in {".", "-"}:
            return True
        try:
            float(keyword)
            return True
        except ValueError:
            return False
    # === Function to get desired mode out of final decision ===
    def extract_mode(self, sentence):
        doc = self.nlp(sentence)

        target_modes = {"observation mode", "cleaning mode", "docking mode"}

        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if chunk_text in target_modes:
                return chunk_text.capitalize()

        for mode in target_modes:
            if mode in sentence.lower():
                return mode.capitalize()

        return None
    # === Function to keep track of how long cleaning has been going for a room ===
    def cleaning_timer(self):
        try:
            h, m, s = map(int, self.cleaning_length.split(":"))
        except ValueError:
            print("Invalid cleaning_length format, expected HH:MM:SS")
            return

        total_seconds = h * 3600 + m * 60 + s

        print("Starting cleaning")
        while self.time_past_cleaning < total_seconds:
            time.sleep(1)
            self.time_past_cleaning += 1
            if self.current_mode != "Cleaning mode":
                print("Cleaning interrupted.")
                return

        print("Cleaning completed")
        self.time_past_cleaning = 0
        self.reset_process()

    ###Running of main GPT process of Node###
    # === Function that starts the AI process ===
    def start_AI_processing(self, connected_page, choice):
        self.version_choice = choice

        if self.version_choice == self.version_choices_names[1]:
            if self.connected_page != connected_page and self.connected_page != None:
                self.reset_process()
            if self.connected_page == None:
                GuiLogger.instance().log(f"Adding system contexts", "debug_field")
                conversation_system_input = []
                conversation_system_input.append({
                    "type": "text", "text": self.function_context
                })
                conversation_system_input.append({
                    "type": "text", "text": self.keypoint_extraction_context
                })
                conversation_system_input.append({
                    "type": "text", "text": self.action_reflection
                })
                self.conversation_history.append({
                    "role": "system",
                    "content": conversation_system_input
                })
            self.connected_page = connected_page
            self.signal_emitter.transition_lock.emit()
            
            self.is_process_running = True
            threading.Thread(target=self.time_upkeep, daemon=True).start()

            if self.connected_page == self.gpt_page_names[1]:
                threading.Thread(target=self.main_loop_value_aware_process, daemon=True).start()
            if self.connected_page == self.gpt_page_names[2]:
                self.start_camera_intake()
        if self.version_choice == self.version_choices_names[2]:
            if not self.check_context:
                GuiLogger.instance().log(f"Adding system contexts", "debug_field")
                check_system_input = []
                check_system_input.append({
                    "type": "text", "text": self.cleaning_check_context
                })
                self.check_context.append({
                    "role": "system",
                    "content": check_system_input
                })
            self.start_camera_intake()
    # === Function that resets process, clear all variables that had been saving data ===
    def main_loop_value_aware_process(self):
        GuiLogger.instance().log(f"Value-aware process started on {self.connected_page}", "debug_field")
        # Start timer for decision process
        start_timer = time.time()

        # Gets the images of the correct folder for each page
        self.get_image_folder()

        # Get current time and day
        self.time_context = f"The current day is {self.current_datetime.strftime('%A')}, and the current time is {self.current_datetime.strftime('%H:%M:%S')}"

        # Keeping track of length of observarion
        if self.current_mode == "Observation mode" and self.start_time_observation:
            self.Current_Observation_time = self.current_datetime.strftime('%H:%M:%S')

            start = datetime.strptime(self.start_time_observation, "%H:%M:%S")
            current = datetime.strptime(self.Current_Observation_time, "%H:%M:%S")
            delta = current - start

            if delta.total_seconds() < 0:
                delta += timedelta(days=1)

            difference_str = str(delta)
            if '.' in difference_str:
                difference_str = difference_str.split('.')[0]
            self.observation_length_context = difference_str

            print(f"observation start time: {self.start_time_observation}")
            print(f"Current observation time: {self.Current_Observation_time}")
            print(f"--------observation length: {self.observation_length_context}--------")
        else:
            self.start_time_observation = None
            self.Current_Observation_time = None

        # getting the correct image for GPT Page 1
        if self.connected_page == self.gpt_page_names[1]:
            # Check if the entire folder has been gone through and repeat
            if self.iteration >= len(self.image_folder):
                self.iteration = 1

            # Get the image
            image = self.image_folder[self.iteration - 1]
            image_path = os.path.join(self.image_folder_path, image)
            self.signal_emitter.update_image.emit(image_path)

            # print on GUI the current iteration in both thoughts and check-up field
            self.signal_emitter.start_iteration_thoughts_field.emit(self.iteration)
            self.signal_emitter.start_iteration_checkup_field.emit(self.iteration)

            # Turning the image into a base64 to be able to feed it to a gpt model
            self.base64_image = self.image_to_base64(image_path)
        # getting the correct image batch for GPT Page 2
        if self.connected_page == self.gpt_page_names[2]:
            # assign an empty base64 image array for new batch
            self.base64_images = []

            # store the batch in the array
            for image_path in self.current_frames:
                self.base64_image = self.image_to_base64(image_path)
                self.base64_images.append(self.base64_image)
            
            # print on GUI the current iteration in both thoughts and check-up field
            self.signal_emitter.start_iteration_thoughts_field.emit(self.iteration)
            self.signal_emitter.start_iteration_checkup_field.emit(self.iteration)
        
        # Extracts keypints that could impact whether to clean the room or not
        self.signal_emitter.update_thoughts_field.emit("Keypoints: ")
        self.signal_emitter.endline_for_thtsf.emit()
        self.extract_keypoints_from_iteration()
        self.signal_emitter.endline_for_thtsf.emit()
        # Value-aware thought process for whether to clean the room or not
        self.signal_emitter.update_thoughts_field.emit("Thought process: ")
        self.signal_emitter.endline_for_thtsf.emit()
        self.value_aware_thought_process()
        self.signal_emitter.endline_for_thtsf.emit()
        # segment thought field
        self.signal_emitter.thought_field_segment.emit()
        
        # Final decision made based on thought process
        self.signal_emitter.clear_final_decision_field_if_not_empty.emit()
        self.final_decision_process()

        # Extract chosen mode from final decision
        self.previous_mode = self.current_mode
        self.current_mode = self.extract_mode(self.final_decision)
        # Emit chosen mode to check up field
        self.signal_emitter.mode_action.emit(self.current_mode)
        # segment checkup field
        self.signal_emitter.checkup_field_segment.emit()

        # summarize and evaluate value-aware process
        self.signal_emitter.update_checkup_field.emit("Summarize: ")
        self.signal_emitter.endline_for_chupf.emit()
        self.summarize_and_evalute_decision_making()
        self.signal_emitter.endline_for_chupf.emit()
        self.signal_emitter.checkup_field_segment.emit()

        # calculate the time it took to go through the entire deicision process
        time_taken = time.time() - start_timer
        GuiLogger.instance().log(f"Time decision process took: {time_taken}", "debug_field")

        # Append all necesary information to conversational history
        self.append_conversational_history()

        self.iteration += 1
        self.keywords = []
        self.current_frames = []
        self.is_process_running = False
        self.signal_emitter.unlock_transition_lock.emit()
        if self.connected_page == self.gpt_page_names[2]:
            self.delete_frames()

        # Observation started 
        msg = String()
        msg.data = self.current_mode
        if self.current_mode == "Observation mode":
            if not self.start_time_observation:
                self.start_time_observation = self.current_datetime.strftime('%H:%M:%S')
            self.signal_emitter.observation_choice.emit()
            self.mode_publisher.publish(msg)
        # Cleaning started 
        if self.current_mode == "Cleaning mode":
            threading.Thread(target=self.cleaning_timer, daemon=True).start()
            self.signal_emitter.cleaning_choice.emit()
            self.mode_publisher.publish(msg)
        # Docking started 
        if self.current_mode == "Docking mode":
            self.reset_process()
            self.signal_emitter.docking_choice.emit()
    # === Function that resets process, clear all variables that had been saving data ===
    def reset_process(self):
        GuiLogger.instance().log(f"Resetting variables value-aware process", "debug_field")
        self.current_folder += 1

        # Variables to remember the GPT pages and which one is connected and where output needs to be written down
        self.connected_page = None

        # Context variables
        self.observation_length_context = "00:00:00"
        self.start_time_observation = None
        self.current_time_observation = None

        # Variable to track what iteration the value-aware process is in on the same page
        self.iteration = 1

        # Conversation and thinking variables for remembering thinking process
        self.conversation_history = []
        conversation_system_input = []
        conversation_system_input.append({
            "type": "text", "text": self.function_context
        })
        conversation_system_input.append({
            "type": "text", "text": self.keypoint_extraction_context
        })
        conversation_system_input.append({
            "type": "text", "text": self.action_reflection
        })
        self.conversation_history.append({
            "role": "system",
            "content": conversation_system_input
        })
# Variables that store model output data
        self.keywords = []
        self.thought_process = ""
        self.final_decision = ""

        # Variables that save image data
        self.image_folder_path = None
        self.image_folder = None
        self.base64_image = None

        self.current_frames = []

        # Variable to indicate the mode the robot is in
        self.current_mode = None
        self.previous_mode = None

        # Variable to indicate the mode
        self.time_past_cleaning = 0 

        # Variables for camera intake
        self.allow_camera_intake = None

    ###Functions for GPT value-aware decision process###
    # === Extracts keypints that could impact whether to clean the room or not ===
    def extract_keypoints_from_iteration(self):
        GuiLogger.instance().log(f"Extracting keypoints...", "debug_field")

        user_input = []
        if self.connected_page == self.gpt_page_names[1]:
            user_input.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"}})
        if self.connected_page == self.gpt_page_names[2]:
            for i, img in enumerate(self.base64_images, start=1):
                user_input.append(
                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                )
                if i in self.current_frames_time:
                    user_input.append(
                        { "type": "text", "text": f"time image taken: {self.current_frames_time[i]}"}
                    )
        user_input.append({
            "type": "text", "text": ("Extract out of what you see the keypoints that could impact whether you would clean the room or not, be sure to take only those you believe are important for a vacuum cleaner to consider. I want the keypoints in short keywords.")
        })
        key_points_extract_prompt = self.conversation_history + [{"role": "user", "content": user_input}]

        key_points_response = client.chat.completions.create(
            model="gpt-4o",
            messages=key_points_extract_prompt,
            stream=True
        )
        self.key_points = ""

        for chunk in key_points_response:
            if chunk.choices:
                token = chunk.choices[0].delta.content
                if token:
                    self.keywords.append(token)
                    self.key_points += token
                    self.signal_emitter.update_thoughts_field.emit(token)
                    time.sleep(0.05)

        self.get_keywords()
        #print(f"keywords: {self.keywords}")
    # === Value-aware thought process for whether to clean the room or not ===
    def value_aware_thought_process(self):
        GuiLogger.instance().log(f"Starting value-aware thought process...", "debug_field")

        user_input = []
        if self.current_mode == "Observation mode":
            user_input.append({
                "type": "text",
                "text": f"Iteration {self.iteration}: Current time: {self.time_context}, Current mode: {self.current_mode}, Length of observation: {self.observation_length_context}, Keypoints from image: {self.keywords}."
            })
        else:
           user_input.append({
                "type": "text",
                "text": f"Iteration {self.iteration}: Current time: {self.time_context}, Current mode: {self.current_mode}, Keypoints from image: {self.keywords}."
            }) 
        if self.connected_page == self.gpt_page_names[1]:
            user_input.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"}})
        if self.connected_page == self.gpt_page_names[2]:
            for i, img in enumerate(self.base64_images, start=1):
                user_input.append(
                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                )
                if i in self.current_frames_time:
                    user_input.append(
                        { "type": "text", "text": f"time image taken: {self.current_frames_time[i]}"}
                    )
        user_input.append({
            "type": "text", 
            "text": f"Based on all the information you know, go though the thought process step by step to see whether you would clean the room or not. I do not expect an answer for which mode to switch to, just the thought process."})
        thought_process_prompt = self.conversation_history + [{"role": "user", "content": user_input}]
        
        thought_process_response = client.chat.completions.create(
            model="gpt-4o",
            messages=thought_process_prompt,
            stream=True
        )
        self.thought_process = ""

        for chunk in thought_process_response:
            if chunk.choices:
                token = chunk.choices[0].delta.content
                if token:
                    self.thought_process += token
                    self.signal_emitter.update_thoughts_field.emit(token)
                    time.sleep(0.05)
    # === Final decision made based on thought process ===
    def final_decision_process(self):
        GuiLogger.instance().log(f"Making final decision...", "debug_field")

        user_input = []
        if self.current_mode == "Observation mode":
            user_input.append({
                    "type": "text",
                    "text": f"Iteration {self.iteration}: Current time: {self.time_context}, Current mode: {self.current_mode}, Length of observation: {self.observation_length_context}, Keypoints from image: {self.keywords}."
                })
        else:
           user_input.append({
                "type": "text",
                "text": f"Iteration {self.iteration}: Current time: {self.time_context}, Current mode: {self.current_mode}, Keypoints from image: {self.keywords}."
            })
        if self.connected_page == self.gpt_page_names[1]:
            user_input.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"}})
        if self.connected_page == self.gpt_page_names[2]:
            for i, img in enumerate(self.base64_images, start=1):
                user_input.append(
                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                )
                if i in self.current_frames_time:
                    user_input.append(
                        { "type": "text", "text": f"time image taken: {self.current_frames_time[i]}"}
                    )
        user_input.extend([
            {"type": "text", "text": f"Thought process made for all this infromation: {self.thought_process}."},
            {"type": "text", "text": f"Based on all you know make de final decision on which mode to switch to. Keep this short, and clearly state which mode you are switching to."}
        ])
        final_decision_prompt = self.conversation_history + [{"role": "user", "content": user_input}]
        
        final_decision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=final_decision_prompt,
            stream=True
        )
        self.final_decision = ""

        for chunk in final_decision_response:
            if chunk.choices:
                token = chunk.choices[0].delta.content
                if token:
                    self.final_decision += token
                    self.signal_emitter.update_final_decision_field.emit(token)
                    time.sleep(0.05)
    # === Summarize and evalute decision making ===
    def summarize_and_evalute_decision_making(self):

        user_input = []
        if self.current_mode == "Observation mode":
            user_input.append({
                "type": "text",
                "text": f"Iteration {self.iteration}: Current time: {self.time_context}, mode: {self.previous_mode} -> mode switched to: {self.current_mode}, Length of observation: {self.observation_length_context}."
            })
        else:
           user_input.append({
                "type": "text",
                "text": f"Iteration {self.iteration}: Current time: {self.time_context}, mode: {self.previous_mode} -> mode switched to: {self.current_mode}."
            })
        if self.connected_page == self.gpt_page_names[1]:
            user_input.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"}})
        if self.connected_page == self.gpt_page_names[2]:
            for i, img in enumerate(self.base64_images, start=1):
                user_input.append(
                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                )
                if i in self.current_frames_time:
                    user_input.append(
                        { "type": "text", "text": f"time image taken: {self.current_frames_time[i]}"}
                    )
        user_input.append({
            "type": "text", 
            "text": f"Keypoints from situation: {self.keywords}, Thought process for situation: {self.thought_process}, Final decision for situation: {self.final_decision}"
        })
        user_input.append({
            "type": "text", 
            "text": f"Summarize your decision making process and evaluate how value-aware you decisons were and how well you complied with your base functionality as a vacuum cleaner. Keep it short and to the point where you clearly state how well you believe your decision process was"
        })

        summarization_prompt = [{"role": "user", "content": user_input}]
        summarization_response = client.chat.completions.create(
            model="gpt-4o",
            messages=summarization_prompt,
            stream=True
        )
        self.summarization_decision = ""

        for chunk in summarization_response:
            if chunk.choices:
                token = chunk.choices[0].delta.content
                if token:
                    self.summarization_decision += token
                    self.signal_emitter.update_checkup_field.emit(token)
                    time.sleep(0.05)
    # === Append all necesary information to conversational history ===
    def append_conversational_history(self):
        # Append user input to conversational history
        user_history_input = []
        if self.current_mode == "Observation mode":
            user_history_input.append({
                "type": "text",
                "text": f"Iteration {self.iteration}: Current time: {self.time_context}, mode: {self.previous_mode} -> mode switched to: {self.current_mode}, Length of observation: {self.observation_length_context}."
            })
        else:
           user_history_input.append({
                "type": "text",
                "text": f"Iteration {self.iteration}: Current time: {self.time_context}, mode: {self.previous_mode} -> mode switched to: {self.current_mode}."
            })
        if self.connected_page == self.gpt_page_names[1]:
            user_history_input.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"}})
        if self.connected_page == self.gpt_page_names[2]:
            for i, img in enumerate(self.base64_images, start=1):
                user_history_input.append(
                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                )
                if i in self.current_frames_time:
                    user_history_input.append(
                        { "type": "text", "text": f"time image taken: {self.current_frames_time[i]}"}
                    )
        self.conversation_history.append({"role": "user", "content": user_history_input})
        
        # Append model output to conversational history
        assistant_history_input = {"role": "assistant","content": 
                                        [
                                            {"type": "text", "text": f" Keypoints from situation: {self.keywords}, Thought process for situation: {self.thought_process}, Final decision for situation: {self.final_decision}"}
                                        ]
                                    }
        self.conversation_history.append(assistant_history_input)
    # === Clean process disturbance check ===
    def check_if_cleaning_could_be_disturbed(self):
        # assign an empty base64 image array for new batch
        self.base64_images = []

        # store the batch in the array
        for image_path in self.current_frames:
            self.base64_image = self.image_to_base64(image_path)
            self.base64_images.append(self.base64_image)

        GuiLogger.instance().log(f"Checking if cleaning could be disturbed...", "debug_field")
        self.signal_emitter.update_thoughts_field.emit("Cleaning check: ")

        user_input = []
        for i, img in enumerate(self.base64_images, start=1):
            user_input.append(
                { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
            )
            if i in self.current_frames_time:
                user_input.append(
                    { "type": "text", "text": f"time image taken: {self.current_frames_time[i]}"}
                )
        user_input.append({
            "type": "text", "text": ("Based on what you see, does the current enviorment disrupt the cleaning decision you made, making it a time to stop cleaning? answer with a yes or no.")
        })
        check_prompt = self.check_context + [{"role": "user", "content": user_input}]

        check_response = client.chat.completions.create(
            model="gpt-4o",
            messages=check_prompt,
            stream=True
        )
        self.check_results = ""

        for chunk in check_response:
            if chunk.choices:
                token = chunk.choices[0].delta.content
                if token:
                    self.check_results += token
                    self.signal_emitter.update_thoughts_field.emit(token)
                    time.sleep(0.05)

        self.signal_emitter.endline_for_thtsf.emit()

        answer = self.check_results.strip().lower()
        match = re.search(r'\b(yes|no)\b', answer)

        if match:
            result = match.group(1)
            self.current_frames = []

            self.signal_emitter.thought_field_segment.emit()

            if result == "yes":
                self.version_choice = self.version_choices_names[1]
            self.start_AI_processing(self.connected_page, self.version_choice)
        else:
            print("No valid 'yes' or 'no' found in impact response.")

    ###Functions for camera intake###
    # === Allows camera footage to be taken ===
    def start_camera_intake(self):
        if self.version_choice == self.version_choices_names[1]:
            GuiLogger.instance().log(f"{self.version_choices_names[1]}: ", "debug_field")
            # Start observation mode
            if self.current_mode != "Observation mode":
                self.current_mode = "Observation mode"
                msg = String()
                msg.data = "Observation mode"
                self.mode_publisher.publish(msg)
        if self.version_choice == self.version_choices_names[2]:
            GuiLogger.instance().log(f"{self.version_choices_names[2]}: ", "debug_field")

        # Allow camera intake
        self.allow_camera_intake = True
        self.signal_emitter.transition_lock.emit()

        # Wait untill intake finished
        threading.Thread(target=self.wait_for_all_footage, daemon=True).start()
    # === Stops camera footage from being taken ===
    def stop_camera_intake(self):
        self.allow_camera_intake = False
        self.signal_emitter.unlock_transition_lock.emit()
    # === Loop for withholding the continuation of value-aware process ===
    def wait_for_all_footage(self):
        while self.allow_camera_intake:
            time.sleep(0.2)
        GuiLogger.instance().log(f"Waiting ended", "debug_field")
        if self.version_choice == self.version_choices_names[1]:
            threading.Thread(target=self.main_loop_value_aware_process, daemon=True).start()
        if self.version_choice == self.version_choices_names[2]:
            threading.Thread(target=self.check_if_cleaning_could_be_disturbed, daemon=True).start() 
    # === Get camera footage and save them in a frames folder to be used for the GPT model ===
    def camera_footge_callback(self, image):
        if not self.allow_camera_intake:
            return
        
        frame = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 3))
        self.temp_frames.append(frame)

        if self.version_choice == self.version_choices_names[1]:
            if len(self.temp_frames) % 36 == 0:
                    self.temp_frames = []

                    frame_name = f"{len(self.current_frames) + 1}.jpg"
                    frame_path = os.path.join(self.frames_folder_path, frame_name)
                    self.current_frames.append(frame_path)

                    height, width, _ = frame.shape
                    frame = cv2.resize(frame, (width, height))

                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(frame_path, frame)

                    self.current_frames_time[len(self.current_frames)] = self.current_datetime.strftime('%H:%M:%S')

                    GuiLogger.instance().log(f"current frames({len(self.current_frames)})", "debug_field")

                    if len(self.current_frames) == 15:
                        self.stop_camera_intake()
        if self.version_choice == self.version_choices_names[2]:
            if len(self.temp_frames) % 5 == 0:
                    self.temp_frames = []

                    frame_name = f"{len(self.current_frames) + 1}.jpg"
                    frame_path = os.path.join(self.frames_folder_path, frame_name)
                    self.current_frames.append(frame_path)

                    height, width, _ = frame.shape
                    frame = cv2.resize(frame, (width, height))

                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(frame_path, frame)

                    self.current_frames_time[len(self.current_frames)] = self.current_datetime.strftime('%H:%M:%S')

                    GuiLogger.instance().log(f"current frames({len(self.current_frames)})", "debug_field")
                    
                    if len(self.current_frames) == 3:
                        self.stop_camera_intake()
    # === Delete image from the frames folder and clear up curren_frames array ===
    def delete_frames(self):
        GuiLogger.instance().log(f"Deleting frames...", "debug_field")
        self.current_frames = []
        for filename in os.listdir(self.frames_folder_path):
            file_path = os.path.join(self.frames_folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

###Qt application###
class Ros2AppWindow(QMainWindow):
    def __init__(self, signal_emitter: AppSignalEmitter, camera_node: CameraNode, robotcontrol_node: RobotControlNode, ai_node: ArtificialIntelligenceNode):
        super().__init__()
        self.setWindowTitle("GPT-4o Thought Process")
        self.setGeometry(0, 0, 1650, 920)

        ###Variables###
        # === initialize Nodes and QObjects ===
        self.signal_emitter = signal_emitter
        self.camera_node = camera_node
        self.robotcontrol_node = robotcontrol_node
        self.ai_node = ai_node

        # === initialize variables usefull for all pages ===
        # flag to anotate that application initialization was done
        self.init_finished = False

        # Keeps track of the Docking mode of the robot
        self.is_docked = None

        # The names of all the pages
        self.page_names = {
            0: "Debug Page",
            1: "GPT Page",
            2: "Real-time GPT Page"
        }
        # Keeps track of the page the application is currently on
        self.current_page = self.page_names[0]

        # Keeps track which button was clicked
        self.button_clicked = None

        # flag that checks if a transition is happening
        self.in_transition = False

        # === initialize variables usefull for GPT pages ===
        self.connected_page = None

        ###Building base page###
        # === Central container and layout ===
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # === Stacked Pages ===
        self.stacked_widget = QStackedWidget()

        # === Navigation Bar (always on top) ===
        self.navbar = QVBoxLayout()
        self.button_row = QHBoxLayout()
        self.text_row = QHBoxLayout()

        self.page1_button = QPushButton("Debug Page")
        self.page1_button.setMinimumSize(100, 0)
        self.page1_button.setEnabled(False)
        self.button_row.addWidget(self.page1_button)

        self.page2_button = QPushButton("GPT Page")
        self.page2_button.setMinimumSize(100, 0)
        self.page2_button.setEnabled(False)
        self.button_row.addWidget(self.page2_button)

        self.page3_button = QPushButton("Real-time GPT Page")
        self.page3_button.setMinimumSize(100, 0)
        self.page3_button.setEnabled(False)
        self.button_row.addWidget(self.page3_button)

        self.button_row.addStretch()

        self.status_text = "Status: Unknown"
        self.status_label = QLabel(self.status_text)
        self.button_row.addWidget(self.status_label)

        self.page_label = QLabel("")
        self.current_page = None
        self.text_row.addWidget(self.page_label)

        self.text_row.addStretch()

        self.navbar.addLayout(self.button_row)
        self.navbar.addLayout(self.text_row)

        self.main_layout.addLayout(self.navbar)     

        # === add stacked widget to main layout ===
        self.main_layout.addWidget(self.stacked_widget)

        # === Pages ===
        self.page1 = self.create_debug_page()
        self.page2 = self.create_gpt_page()
        self.page3 = self.create_realtime_gpt_page()

        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)

        # === Connect navigation buttons ===
        self.page1_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.page2_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.page3_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        # === In app signale for stacked_widget change/going to another page ===
        self.stacked_widget.currentChanged.connect(self.update_current_page)
        self.stacked_widget.currentChanged.connect(self.signal_manager)

        ###Signal receivers###
        # === Signal for updating status label ===
        self.signal_emitter.dock_status_updated.connect(self.update_status_label)

        # === Signal for transition locking and unlocking ===
        self.signal_emitter.transition_lock.connect(self.transition_lock)
        self.signal_emitter.unlock_transition_lock.connect(self.unlock_transition_lock)

        # === Signal for retrying action if undock was done instead ===
        self.signal_emitter.retry_action.connect(self.action_after_undock)
    ###Functions for general use###
    # === Functions for keeping up with the current status ===
    def update_status_label(self, is_docked):
        self.is_docked = is_docked
        status_text = "✅ Docked" if is_docked else "❌ Not Docked"
        self.status_label.setText(f"Status: {status_text}")

        self.buttons_enabeler()
    # === Functions for enabeling buttons on launch an through actions ===
    def buttons_enabeler(self):
        if not self.in_transition:
            if not self.init_finished:
                # Page buttons
                self.page1_button.setEnabled(True)
                self.page2_button.setEnabled(True)
                self.page3_button.setEnabled(True)

                # start buttons
                self.start_button_1.setEnabled(True)
                self.start_button_2.setEnabled(True)

                # Robot mode buttons
                self.cleaning_mode_button.setEnabled(True)
                self.observation_mode_button.setEnabled(True)

                # flag for one time trigger
                self.init_finished = True

            if self.is_docked:
                # Docking status buttons
                self.dock_button.setEnabled(False)
                self.undock_button.setEnabled(True)

                # Robot mode buttons
                self.docking_mode_button.setEnabled(False)
            else:
                # Docking status buttons
                self.dock_button.setEnabled(True)
                self.undock_button.setEnabled(False)

                # Robot mode buttons
                self.docking_mode_button.setEnabled(True)
    # === Locks all action buttons when a transition happans ===
    def transition_lock(self):
        self.in_transition = True

        # === start buttons ===
        self.start_button_1.setEnabled(False)
        self.start_button_2.setEnabled(False)

        # === Docking status buttons ===
        self.dock_button.setEnabled(False)
        self.undock_button.setEnabled(False)

        # === Robot mode buttons ===
        self.cleaning_mode_button.setEnabled(False)
        self.observation_mode_button.setEnabled(False)
        self.docking_mode_button.setEnabled(False)
    # === Unlocks some action buttons when transition ends ===   
    def unlock_transition_lock(self):
        self.is_docked = False
        self.in_transition = False

        # === start buttons ===
        self.start_button_1.setEnabled(True)
        self.start_button_2.setEnabled(True)

        # === Robot mode buttons ===
        self.cleaning_mode_button.setEnabled(True)
        self.observation_mode_button.setEnabled(True)
    # === Updates page label to display what page you are on ===
    def update_current_page(self, index):
        if self.current_page == self.page_names[index]:
            return
        self.current_page = self.page_names[index]       

        self.page_label.setText(f"Page: {self.page_names.get(index, 'Unknown')}")
    # === Retries starting the action if undocking was done instead of it ===
    def action_after_undock(self):
        if self.button_clicked == "Cleaning mode":
            self.start_cleaning()
        if self.button_clicked == "Observation mode":
            self.start_observing()
        if self.button_clicked == "Start":
            self.start_ai()

    ###Shutdown functions###
    # === Function to help with safe shutdown ===
    def closeEvent(self, event):
        event.accept()
    
    ###Functions for workflow of GPT pages###
    # === Function for the connecting and disconnecting of GPT page signals ===
    def signal_manager(self, index):
        # if page is already connected won't need to connect it again
        if self.connected_page == self.page_names[index]:
            return
        # if a start button is not enabled it means the page is currently in use so changing signal connections could be a problem
        if not self.start_button_1.isEnabled() or not self.start_button_2.isEnabled():
            return
        # Connect signals for GPT page 1 and disconnect signals for GPT page 2
        if self.current_page == self.page_names[1]:
            self.connect_signals_for_gpt_page1()
            self.disconnect_signals_for_gpt_page2()
            self.connected_page = self.page_names[1]
        # Connect signals for GPT page 2 and disconnect signals for GPT page 1
        if self.current_page == self.page_names[2]:
            self.connect_signals_for_gpt_page2()
            self.disconnect_signals_for_gpt_page1()
            self.connected_page = self.page_names[2]
    # === Connecting and disconnecting of signals for GPT page ===
    def connect_signals_for_gpt_page1(self):
        GuiLogger.instance().log("connecting page 1 signals", "debug_field")
        # Signals for thoughts field update
        self.signal_emitter.start_iteration_thoughts_field.connect(self.start_iteration_thoughts_field_1_lambda)
        self.signal_emitter.update_thoughts_field.connect(self.update_thoughts_field_1_lambda)
        self.signal_emitter.thought_field_segment.connect(self.Thoughts_end_1_lambda)
        self.signal_emitter.endline_for_thtsf.connect(self.endline_for_thtsf_1_lambda)
        # Signal for final decision field update
        self.signal_emitter.update_final_decision_field.connect(self.update_final_decision_field_1_lambda)
        self.signal_emitter.clear_final_decision_field_if_not_empty.connect(self.empty_check_final_decision_field_1_lambda)
        # Signals for checkup field update
        self.signal_emitter.start_iteration_checkup_field.connect(self.start_iteration_checkup_field_1_lambda)
        self.signal_emitter.update_checkup_field.connect(self.update_checkup_field_1_lambda)
        self.signal_emitter.checkup_field_segment.connect(self.checkup_field_segment_1_lambda)
        self.signal_emitter.mode_action.connect(self.mode_action_1_lambda)
        self.signal_emitter.summary_action.connect(self.summary_action_1_lambda)
        self.signal_emitter.endline_for_chupf.connect(self.endline_for_chupf_1_lambda)
        # Signal connect for current image print
        self.signal_emitter.update_image.connect(self.update_image_1_lambda)
    def disconnect_signals_for_gpt_page1(self):
        try:
            GuiLogger.instance().log("disconnecting page 1 signals", "debug_field")
            # Signals for thoughts field update
            self.signal_emitter.start_iteration_thoughts_field.disconnect(self.start_iteration_thoughts_field_1_lambda)
            self.signal_emitter.update_thoughts_field.disconnect(self.update_thoughts_field_1_lambda)
            self.signal_emitter.thought_field_segment.disconnect(self.Thoughts_end_1_lambda)
            self.signal_emitter.endline_for_thtsf.disconnect(self.endline_for_thtsf_1_lambda)
            # Signal for final decision field update
            self.signal_emitter.update_final_decision_field.disconnect(self.update_final_decision_field_1_lambda)
            self.signal_emitter.clear_final_decision_field_if_not_empty.disconnect(self.empty_check_final_decision_field_1_lambda)
            # Signals for checkup field update
            self.signal_emitter.start_iteration_checkup_field.disconnect(self.start_iteration_checkup_field_1_lambda)
            self.signal_emitter.update_checkup_field.disconnect(self.update_checkup_field_1_lambda)
            self.signal_emitter.checkup_field_segment.disconnect(self.checkup_field_segment_1_lambda)
            self.signal_emitter.mode_action.disconnect(self.mode_action_1_lambda)
            self.signal_emitter.summary_action.disconnect(self.summary_action_1_lambda)
            self.signal_emitter.endline_for_chupf.disconnect(self.endline_for_chupf_1_lambda)
            # Signal connect for current image print
            self.signal_emitter.update_image.disconnect(self.update_image_1_lambda)
        except TypeError:
            pass
    # === Connecting and disconnecting of signals for Real-time GPT page ===
    def connect_signals_for_gpt_page2(self):
        GuiLogger.instance().log("connecting page 2 signals", "debug_field")
        # Signals for thoughts field update
        self.signal_emitter.start_iteration_thoughts_field.connect(self.start_iteration_thoughts_field_2_lambda)
        self.signal_emitter.update_thoughts_field.connect(self.update_thoughts_field_2_lambda)
        self.signal_emitter.thought_field_segment.connect(self.Thoughts_end_2_lambda)
        self.signal_emitter.endline_for_thtsf.connect(self.endline_for_thtsf_2_lambda)
        # Signal for final decision field update
        self.signal_emitter.update_final_decision_field.connect(self.update_final_decision_field_2_lambda)
        self.signal_emitter.clear_final_decision_field_if_not_empty.connect(self.empty_check_final_decision_field_2_lambda)
        # Signals for checkup field update
        self.signal_emitter.start_iteration_checkup_field.connect(self.start_iteration_checkup_field_2_lambda)
        self.signal_emitter.update_checkup_field.connect(self.update_checkup_field_2_lambda)
        self.signal_emitter.checkup_field_segment.connect(self.checkup_field_segment_2_lambda)
        self.signal_emitter.mode_action.connect(self.mode_action_2_lambda)
        self.signal_emitter.summary_action.connect(self.summary_action_2_lambda)
        self.signal_emitter.endline_for_chupf.connect(self.endline_for_chupf_2_lambda)
        # Signal for camera intake
        self.signal_emitter.camera_intake.connect(self.update_image_2_lambda)
        # Signal for AI continuation based on chosen mode
        self.signal_emitter.observation_choice.connect(self.start_observation_version_ai_process)
        self.signal_emitter.cleaning_choice.connect(self.start_cleaning_version_ai_process)
        self.signal_emitter.docking_choice.connect(self.Start_docking)
    def disconnect_signals_for_gpt_page2(self):
        try:
            GuiLogger.instance().log("disconnecting page 2 signals", "debug_field")
            # Signals for thoughts field update
            self.signal_emitter.start_iteration_thoughts_field.disconnect(self.start_iteration_thoughts_field_2_lambda)
            self.signal_emitter.update_thoughts_field.disconnect(self.update_thoughts_field_2_lambda)
            self.signal_emitter.thought_field_segment.disconnect(self.Thoughts_end_2_lambda)
            self.signal_emitter.endline_for_thtsf.disconnect(self.endline_for_thtsf_2_lambda)
            # Signal for final decision field update
            self.signal_emitter.update_final_decision_field.disconnect(self.update_final_decision_field_2_lambda)
            self.signal_emitter.clear_final_decision_field_if_not_empty.disconnect(self.empty_check_final_decision_field_2_lambda)
            # Signals for checkup field update
            self.signal_emitter.start_iteration_checkup_field.disconnect(self.start_iteration_checkup_field_2_lambda)
            self.signal_emitter.update_checkup_field.disconnect(self.update_checkup_field_2_lambda)
            self.signal_emitter.checkup_field_segment.disconnect(self.checkup_field_segment_2_lambda)
            self.signal_emitter.mode_action.disconnect(self.mode_action_2_lambda)
            self.signal_emitter.summary_action.disconnect(self.summary_action_2_lambda)
            self.signal_emitter.endline_for_chupf.disconnect(self.endline_for_chupf_2_lambda)
            # Signals for camera intake
            self.signal_emitter.camera_intake.disconnect(self.update_image_2_lambda)
            # Signal for AI continuation based on chosen mode
            self.signal_emitter.observation_choice.disconnect(self.start_observation_version_ai_process)
            self.signal_emitter.cleaning_choice.disconnect(self.start_cleaning_version_ai_process)
            self.signal_emitter.docking_choice.disconnect(self.Start_docking)
        except TypeError:
            pass

    ###Debug page###
    def create_debug_page(self):

        page = QWidget()
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        button_row_1_layout = QVBoxLayout()
        button_row_1 = QHBoxLayout()
        button_row_2_layout = QVBoxLayout()
        button_row_2 = QHBoxLayout()

        left_layout.addLayout(button_row_1_layout)
        left_layout.addLayout(button_row_2_layout)

        right_layout = QVBoxLayout()

        # === button row 1 ===
        self.docking_status_label = QLabel("Docking status buttons")
        self.docking_status_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        button_row_1_layout.addWidget(self.docking_status_label)

        self.dock_button = QPushButton("Dock")
        self.dock_button.setMinimumSize(200, 100)
        self.dock_button.setStyleSheet("font-size: 36px; font-weight: bold;")
        self.dock_button.setEnabled(False)
        self.dock_button.clicked.connect(self.dock)
        button_row_1.addWidget(self.dock_button)

        self.undock_button = QPushButton("Undock")
        self.undock_button.setMinimumSize(200, 100)
        self.undock_button.setStyleSheet("font-size: 36px; font-weight: bold;")
        self.undock_button.setEnabled(False)
        self.undock_button.clicked.connect(self.undock)
        button_row_1.addWidget(self.undock_button)

        button_row_1.addStretch()

        button_row_1_layout.addLayout(button_row_1)
        # === button row 2 ===
        self.robot_modes_label = QLabel("Robot modes buttons")
        self.robot_modes_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        button_row_2_layout.addWidget(self.robot_modes_label)

        self.cleaning_mode_button = QPushButton("Cleaning mode")
        self.cleaning_mode_button.setMinimumSize(280, 100)
        self.cleaning_mode_button.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.cleaning_mode_button.setEnabled(False)
        self.cleaning_mode_button.clicked.connect(self.start_cleaning)
        button_row_2.addWidget(self.cleaning_mode_button)

        self.observation_mode_button = QPushButton("Observation mode")
        self.observation_mode_button.setMinimumSize(280, 100)
        self.observation_mode_button.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.observation_mode_button.setEnabled(False)
        self.observation_mode_button.clicked.connect(self.start_observing)
        button_row_2.addWidget(self.observation_mode_button)

        self.docking_mode_button = QPushButton("Docking mode")
        self.docking_mode_button.setMinimumSize(280, 100)
        self.docking_mode_button.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.docking_mode_button.setEnabled(False)
        self.docking_mode_button.clicked.connect(self.Start_docking)
        button_row_2.addWidget(self.docking_mode_button)

        button_row_2.addStretch()

        button_row_2_layout.addLayout(button_row_2)

        # === left layout ===
        left_layout.addStretch()

        # === right layout ===
        self.debug_field = QTextEdit()
        self.debug_field.setReadOnly(True)
        self.debug_field.setStyleSheet("font-size: 16px;")
        GuiLogger.instance().register_output_widget("debug_field", self.debug_field)
        right_layout.addWidget(self.debug_field)

        # === Page layout ===
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        page.setLayout(main_layout)
        return page    
    # === Functions for Debug page ===
    # === Docking status buttons functions ===
    def dock(self):
        self.button_clicked = "Dock"
        self.robotcontrol_node.nav_dock()
    def undock(self):
        self.button_clicked = "Undock"
        self.robotcontrol_node.nav_undock()
    # === Robot modes buttons functions ===
    def start_cleaning(self):
        self.button_clicked = "Cleaning mode"
        if not self.is_docked:
            self.robotcontrol_node.mode_selection("Cleaning mode")
        else:
            self.robotcontrol_node.nav_undock()
    def start_observing(self):
        self.button_clicked = "Observation mode"
        if not self.is_docked:
            self.robotcontrol_node.mode_selection("Observation mode")
        else:
            self.robotcontrol_node.nav_undock()
    def Start_docking(self):
        self.button_clicked = "Docking mode"
        self.robotcontrol_node.mode_selection("Docking mode")

    ###GPT page###
    def create_gpt_page(self):

        page = QWidget()
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.image_label_1 = QLabel()
        self.image_label_1.setMinimumSize(640, 480)
        self.image_label_1.setMaximumSize(640, 480)
        self.image_label_1.setAlignment(Qt.AlignRight)
        right_layout.addWidget(self.image_label_1)

        self.checkup_field_label_1 = QLabel("CheckUp Field:")
        self.checkup_field_label_1.setStyleSheet("font-size: 24px; font-weight: bold;")
        right_layout.addWidget(self.checkup_field_label_1)

        self.checkup_field_text_1 = QTextEdit()
        self.checkup_field_text_1.setReadOnly(True)
        self.checkup_field_text_1.setStyleSheet("font-size: 16px;")
        right_layout.addWidget(self.checkup_field_text_1)

        self.thought_label_1 = QLabel("Thought Process:")
        self.thought_label_1.setStyleSheet("font-size: 24px; font-weight: bold;")
        left_layout.addWidget(self.thought_label_1)

        self.thought_field_text_1 = QTextEdit()
        self.thought_field_text_1.setMinimumSize(1160, 550)
        self.thought_field_text_1.setReadOnly(True)
        self.thought_field_text_1.setStyleSheet("font-size: 16px;")
        left_layout.addWidget(self.thought_field_text_1)

        self.final_decision_label_1 = QLabel("Final Response:")
        self.final_decision_label_1.setStyleSheet("font-size: 24px; font-weight: bold;")
        left_layout.addWidget(self.final_decision_label_1)

        self.final_decision_field_text_1 = QTextEdit()
        self.final_decision_field_text_1.setReadOnly(True)
        self.final_decision_field_text_1.setStyleSheet("font-size: 16px;")
        left_layout.addWidget(self.final_decision_field_text_1)

        self.start_button_1 = QPushButton("Start")
        self.start_button_1.setEnabled(False)
        self.start_button_1.clicked.connect(self.start_ai)
        left_layout.addWidget(self.start_button_1)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        page.setLayout(main_layout)

        ###Lambdas for signal connections of GPT page 1###
        # === Lambda for signals of thoughts field update ===
        self.start_iteration_thoughts_field_1_lambda = lambda iteration: self.add_current_iteration_to_field_thtsf(iteration, self.thought_field_text_1)
        self.update_thoughts_field_1_lambda = lambda text: self.append_thoughts_text(text, self.thought_field_text_1)
        self.Thoughts_end_1_lambda = lambda: self.add_new_lines_thtsf(self.thought_field_text_1)
        self.endline_for_thtsf_1_lambda = lambda: self.endline_for_thtsf(self.thought_field_text_1)
        # === Lambda for signal of final decision field update ===
        self.update_final_decision_field_1_lambda = lambda text: self.append_final_decision_text(text, self.final_decision_field_text_1)
        self.empty_check_final_decision_field_1_lambda = lambda: self.empty_check_final_decision_field(self.final_decision_field_text_1)
        # === Lambda for siganls of checkup field update ===
        self.start_iteration_checkup_field_1_lambda = lambda iteration: self.add_current_iteration_to_field_chupf(iteration, self.checkup_field_text_1)
        self.update_checkup_field_1_lambda = lambda text: self.append_checkup_field_text(text, self.checkup_field_text_1)
        self.mode_action_1_lambda = lambda state: self.chosen_mode(state, self.checkup_field_text_1)
        self.summary_action_1_lambda = lambda: self.begin_summary(self.checkup_field_text_1)
        self.checkup_field_segment_1_lambda = lambda: self.add_new_lines_chupf(self.checkup_field_text_1)
        self.endline_for_chupf_1_lambda = lambda: self.endline_for_chupf(self.checkup_field_text_1)
        # === Lambda for signal of image label ===
        self.update_image_1_lambda = lambda image_path: self.update_image_label(image_path, self.image_label_1)

        return page
    ###Real-time GPT page###
    def create_realtime_gpt_page(self):

        page = QWidget()
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.image_label_2 = QLabel()
        self.image_label_2.setMinimumSize(640, 480)
        self.image_label_2.setMaximumSize(640, 480)
        self.image_label_2.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.image_label_2)

        self.checkup_field_label_2 = QLabel("CheckUp Field:")
        self.checkup_field_label_2.setStyleSheet("font-size: 24px; font-weight: bold;")
        right_layout.addWidget(self.checkup_field_label_2)

        self.checkup_field_text_2 = QTextEdit()
        self.checkup_field_text_2.setReadOnly(True)
        self.checkup_field_text_2.setStyleSheet("font-size: 16px;")
        right_layout.addWidget(self.checkup_field_text_2)

        self.thought_label_2 = QLabel("Thought Process:")
        self.thought_label_2.setStyleSheet("font-size: 24px; font-weight: bold;")
        left_layout.addWidget(self.thought_label_2)

        self.thought_field_text_2 = QTextEdit()
        self.thought_field_text_2.setMinimumSize(1160, 550)
        self.thought_field_text_2.setReadOnly(True)
        self.thought_field_text_2.setStyleSheet("font-size: 16px;")
        left_layout.addWidget(self.thought_field_text_2)

        self.final_decision_label_2 = QLabel("Final Response:")
        self.final_decision_label_2.setStyleSheet("font-size: 24px; font-weight: bold;")
        left_layout.addWidget(self.final_decision_label_2)

        self.final_decision_field_text_2 = QTextEdit()
        self.final_decision_field_text_2.setReadOnly(True)
        self.final_decision_field_text_2.setStyleSheet("font-size: 16px;")
        left_layout.addWidget(self.final_decision_field_text_2)

        self.start_button_2 = QPushButton("Start")
        self.start_button_2.setEnabled(False)
        self.start_button_2.clicked.connect(self.start_ai)
        left_layout.addWidget(self.start_button_2)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        page.setLayout(main_layout)

        ###Lambdas for signal connections of GPT page 2###
        # Lambda for signals of thoughts field update
        self.start_iteration_thoughts_field_2_lambda = lambda iteration: self.add_current_iteration_to_field_thtsf(iteration, self.thought_field_text_2)
        self.update_thoughts_field_2_lambda = lambda text: self.append_thoughts_text(text, self.thought_field_text_2)
        self.Thoughts_end_2_lambda = lambda: self.add_new_lines_thtsf(self.thought_field_text_2)
        self.endline_for_thtsf_2_lambda = lambda: self.endline_for_thtsf(self.thought_field_text_2)
        # Lambda for signal of final decision field update
        self.update_final_decision_field_2_lambda = lambda text: self.append_final_decision_text(text, self.final_decision_field_text_2)
        self.empty_check_final_decision_field_2_lambda = lambda: self.empty_check_final_decision_field(self.final_decision_field_text_2)
        # Lambda for siganls of checkup field update
        self.start_iteration_checkup_field_2_lambda = lambda iteration: self.add_current_iteration_to_field_chupf(iteration, self.checkup_field_text_2)
        self.update_checkup_field_2_lambda = lambda text: self.append_checkup_field_text(text, self.checkup_field_text_2)
        self.mode_action_2_lambda = lambda state: self.chosen_mode(state, self.checkup_field_text_2)
        self.summary_action_2_lambda = lambda: self.begin_summary(self.checkup_field_text_2)
        self.checkup_field_segment_2_lambda = lambda: self.add_new_lines_chupf(self.checkup_field_text_2)
        self.endline_for_chupf_2_lambda = lambda: self.endline_for_chupf(self.checkup_field_text_2)
        # Lambda for signal of camera intake
        self.update_image_2_lambda = lambda frame: self.update_camera_output(frame, self.image_label_2)

        return page
    ###Functions for actions and use of GPT pages###
    # === Start ai process in ArtificialIntelligenceNode ===
    def start_ai(self):
        self.button_clicked = "Start"
        if not self.is_docked:
            self.start_observation_version_ai_process()
        else:
            self.robotcontrol_node.nav_undock()
    # AI processing version for observation mode
    def start_observation_version_ai_process(self):
        self.ai_node.start_AI_processing(self.connected_page, "Observation version")
    # AI processing version for cleaning mode
    def start_cleaning_version_ai_process(self):
        self.ai_node.start_AI_processing(self.connected_page, "Cleaning version")

    # === Functions related to though process field ===
    # Print current iteration that is in process
    def add_current_iteration_to_field_thtsf(self, iteration, thought_field_text):
        thought_field_text.append(f"iteration ({iteration}): \n")
    # Used to add streamed llm output to thoughts field
    def append_thoughts_text(self, text, thought_field_text):
        cursor = thought_field_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        thought_field_text.setTextCursor(cursor)
        thought_field_text.insertPlainText(text)
    # Add a new line with segmentation in thoughts field
    def add_new_lines_thtsf(self, thought_field_text):
        thought_field_text.append("\n====================================================================================\n\n")
    # Add a new line in thoughts field
    def endline_for_thtsf(self, thought_field_text):
        thought_field_text.append("\n")

    # === Functions related to final decision field ===
    # Used to add streamed llm output to final decision field
    def append_final_decision_text(self, text, final_decision_field_text):
        cursor = final_decision_field_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        final_decision_field_text.setTextCursor(cursor)
        final_decision_field_text.insertPlainText(text)
    # Check if final decision field is empty or not
    def empty_check_final_decision_field(self, final_decision_field_text):
        if final_decision_field_text.toPlainText().strip() != "":
            final_decision_field_text.clear()

    # === Functions related to check-up field ===
    # Print current iteration that is in process
    def add_current_iteration_to_field_chupf(self, iteration, checkup_field_text):
        checkup_field_text.append(f"iteration ({iteration}): \n")
    # Used to add streamed llm output to the chechup field
    def append_checkup_field_text(self, text, checkup_field_text):
        cursor = checkup_field_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        checkup_field_text.setTextCursor(cursor)
        checkup_field_text.insertPlainText(text)
    # Print before mode print
    def chosen_mode(self, state, checkup_field_text):
        checkup_field_text.append(f"State: [{state}] \n")
    # Print before summary stream of llm
    def begin_summary(self, checkup_field_text):
        checkup_field_text.append("Summary: \n")
    # Add a new line with segmentation in checkup field
    def add_new_lines_chupf(self, checkup_field_text):
        checkup_field_text.append("\n==============================================\n")
    # Print and endline in checkup field
    def endline_for_chupf(self, checkup_field_text):
        checkup_field_text.append("\n")

    # === Functions related to image/camera footage ===
    # Update label to display new image on ui
    def update_image_label(self, image_path, image_label):
        pixmap = QPixmap(image_path)
        image_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))
    # Update label to display camera output on ui
    def update_camera_output(self, image, image_label):
        frame = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 3))

        h, w, ch = frame.shape
        frame = cv2.resize(frame, (480, 480))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        image_label.setPixmap(QPixmap.fromImage(qt_img))

# === Classes to help with printing in the qt application ===
class GuiLogger:
    _instance = None

    def __init__(self):
        self.widgets = {}
        self.emitter = LogEmitter()
        self.emitter.log_signal.connect(self._log_to_widget)

    @staticmethod
    def instance():
        if GuiLogger._instance is None:
            GuiLogger._instance = GuiLogger()
        return GuiLogger._instance

    def register_output_widget(self, name: str, widget: QTextEdit):
        self.widgets[name] = widget

    def log(self, message: str, widget_name: str = "default"):
        self.emitter.log_signal.emit(message, widget_name)

    def _log_to_widget(self, message: str, widget_name: str):
        widget = self.widgets.get(widget_name)
        if widget:
            widget.append(message)
class LogEmitter(QObject):
    log_signal = pyqtSignal(str, str)

# === running of qt app on main thread, ros2 on subthread, and shutdown ===
class SpinRosThread(threading.Thread):
    def __init__(self, multi_executor):
        super().__init__()
        self.multi_executor = multi_executor

    def run(self):
        global ssh

        robot_ip = "192.168.1.103"
        robot_user = "ubuntu"
        robot_password = "turtlebot4"

        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(robot_ip, username=robot_user, password=robot_password)

        self.multi_executor.spin()

    def stop(self):
        rclpy.shutdown()
def main():
    signal_emitter = AppSignalEmitter()

    #battery_state_node = BatteryStateNode()
    robotcontrol_node = RobotControlNode(signal_emitter)
    camera_node = CameraNode(signal_emitter)
    ai_node = ArtificialIntelligenceNode(signal_emitter)


    multi_executor = MultiThreadedExecutor()
    #multi_executor.add_node(battery_state_node)
    multi_executor.add_node(robotcontrol_node)
    multi_executor.add_node(camera_node)
    multi_executor.add_node(ai_node)

    spin_thread = SpinRosThread(multi_executor)
    spin_thread.start()

    app = QApplication([])
    window = Ros2AppWindow(signal_emitter, camera_node, robotcontrol_node, ai_node)
    window.show()

    window.closeEvent = lambda event: shutdown(spin_thread, multi_executor, robotcontrol_node, camera_node, ai_node)

    try:
        app.exec_()
    except KeyboardInterrupt:
        shutdown(spin_thread, multi_executor, robotcontrol_node, camera_node, ai_node)
    finally:
        shutdown(spin_thread, multi_executor, robotcontrol_node, camera_node, ai_node)
def shutdown(spin_thread, multi_executor, robotcontrol_node, camera_node, ai_node):
    global ssh, shutdown_called

    if shutdown_called:
        return
    shutdown_called = True

    robotcontrol_node.current_state = None
    time.sleep(0.5)

    ai_node.current_frames = []
    for filename in os.listdir(ai_node.frames_folder_path):
        file_path = os.path.join(ai_node.frames_folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    multi_executor.shutdown()
    #battery_state_node.destroy_node()
    robotcontrol_node.destroy_node()
    camera_node.destroy_node()
    ai_node.destroy_node()

    ssh.close()
    cv2.destroyAllWindows()

    spin_thread.stop()

    QApplication.quit()
if __name__ == '__main__':
    main()
