# === For specific nodes ===
import signal
from enum import IntEnum
import time
import random
import threading
from PIL import Image as PILImage
import numpy as np
import pandas as pd
import os
import cv2
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

import openai
from dotenv import load_dotenv

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

    def __init__(self):
        super().__init__()

    def __init__(self):
        super().__init__()

class GPTThread(QThread):
    def __init__(self, signal_emitter: AppSignalEmitter):
        super().__init__()

        self.signal_emitter = signal_emitter

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

        ###Main GPT process variables###
        # === Context variables ===
        self.time_context = None

        self.observation_length_context = "00:00:00"
        self.start_time_observation = None
        self.current_time_observation = None

        # === Variable to track what iteration the value-aware process is in on the same page ===
        self.iteration = 1

        # === Conversation and thinking variables for remembering thinking process ===
        self.conversation_history = []
        # Variables that store model output data
        self.keywords = []
        self.thought_process = ""
        self.final_decision = ""

        # === Variables that save image data ===
        self.image_folder_path = None
        self.image_folder = None
        self.base64_image = None

        # === Variable to indicate the mode the robot is in ===
        self.current_mode = None
        self.previous_mode = None

        # === Variable to indicate the mode the robot is in ===
        self.cleaning_length = "00:10:00"
        self.time_past_cleaning = 0

        # === Variable to track if value-aware process is running ===
        self.is_process_running = False

        # === perparing for general context of keypoints ===
        """
        self.example_folders = self.keypoint_extraction_examples["folder"].unique().tolist()
        self.matching_paths = [path for path in self.activity_folders if os.path.basename(path) in self.example_folders]
        for path in self.matching_paths:
            images_in_folder = os.listdir(path)
            images_in_folder = sorted(images_in_folder, key=lambda x: int(os.path.splitext(x)[0]))
            folder_name = [folder for folder in self.example_folders if folder == os.path.basename(path)]
            for image in images_in_folder:
                image_path = os.path.join(path, image)
                base64_image = self.image_to_base64(image_path)
                row = self.keypoint_extraction_examples[
                    (self.keypoint_extraction_examples["folder"] == folder_name[0]) &
                    (self.keypoint_extraction_examples["image"] == image)
                ]
                if not row.empty:
                    keypoints = row.iloc[0]["keypoints"].split(";")
                    keypoints_str = "\n".join(keypoints)
                conversation_system_input.extend(
                    {"type": "text", "text": f"Keypoint examples: {keypoints_str}"}
                )
        """

    ###General functions###s
    # === Function to get the current image folder ===
    def get_image_folder(self):
        self.image_folder_path = self.activity_folders[self.current_folder]
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
    def start_AI_processing(self):
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
        self.signal_emitter.transition_lock.emit()
        
        self.is_process_running = True
        threading.Thread(target=self.time_upkeep, daemon=True).start()

        threading.Thread(target=self.main_loop_value_aware_process, daemon=True).start()
    # === Function that resets process, clear all variables that had been saving data ===
    def main_loop_value_aware_process(self):
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
        print(f"Time decision process took: {time_taken}", "debug_field")

        # Append all necesary information to conversational history
        self.append_conversational_history()

        self.iteration += 1
        self.keywords = []
        self.current_frames = []
        self.is_process_running = False
        self.signal_emitter.unlock_transition_lock.emit()

        # Observation started 
        if self.current_mode == "Observation mode":
            if not self.start_time_observation:
                self.start_time_observation = self.current_datetime.strftime('%H:%M:%S')
        # Cleaning started 
        if self.current_mode == "Cleaning mode":
            threading.Thread(target=self.cleaning_timer, daemon=True).start()
        # Docking started 
        if self.current_mode == "Docking mode":
            self.reset_process()
    # === Function that resets process, clear all variables that had been saving data ===
    def reset_process(self):
        print(f"Resetting variables value-aware process")
        self.current_folder += 1

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

        # Variable to indicate the mode the robot is in
        self.current_mode = None
        self.previous_mode = None

        # === Variable to indicate the mode the robot is in ===
        self.time_past_cleaning = 0

    ###Functions for GPT value-aware decision process###
    # === Extracts keypints that could impact whether to clean the room or not ===
    def extract_keypoints_from_iteration(self):
        print("Extracting keypoints...")

        text_prompt = "Extract out of what you see the keypoints that could impact whether you would clean the room or not, be sure to take only those you believe are important for a vacuum cleaner to consider. I want the keypoints in short keywords"
        key_points_extract_prompt = self.conversation_history + [{
                                "role": "user", 
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"}},
                                    {"type": "text", "text": text_prompt}
                                ]
                            }]

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
        print(f"keywords: {self.keywords}")
    # === Value-aware thought process for whether to clean the room or not ===
    def value_aware_thought_process(self):
        print("Starting value-aware thought process...")

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
        user_input.extend([
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"}},
            {"type": "text", "text": f"Based on all the information you know, go though the thought process step by step to see whether you would clean this room or not. I do not expect an answer for which mode to switch to, just the thought process."}
        ])
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
        print("Making final decision...")

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
        user_input.extend([
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"}},
            {"type": "text", "text": f"Thought process made for all this infromation: {self.thought_process}."},
            {"type": "text", "text": f"Based on all you know make de final decision on which mode to switch to for a value-aware decision. Keep this short, and clearly state which mode you are switching to."}
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
        user_input.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"}})
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
        user_history_input.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"}})
        self.conversation_history.append({"role": "user", "content": user_history_input})
        
        # Append model output to conversational history
        assistant_history_input = {"role": "assistant","content": 
                                        [
                                            {"type": "text", "text": f" Keypoints from situation: {self.keywords}, Thought process for situation: {self.thought_process}, Final decision for situation: {self.final_decision}"}
                                        ]
                                    }
        self.conversation_history.append(assistant_history_input)

class GPTWindow(QMainWindow):
    def __init__(self, signal_emitter: AppSignalEmitter, gpt_node: GPTThread):
        super().__init__()

        self.signal_emitter = signal_emitter
        self.gpt_node = gpt_node

        self.setWindowTitle("GPT-4o Thought Process")
        self.setGeometry(0, 0, 1910, 1000)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignRight)
        right_layout.addWidget(self.image_label)

        self.checkup_field_labael = QLabel("CheckUp Field:")
        self.checkup_field_labael.setStyleSheet("font-size: 24px; font-weight: bold;")
        right_layout.addWidget(self.checkup_field_labael)

        self.checkup_field_text = QTextEdit()
        self.checkup_field_text.setReadOnly(True)
        self.checkup_field_text.setStyleSheet("font-size: 16px;")
        right_layout.addWidget(self.checkup_field_text)

        self.thought_label = QLabel("Thought Process:")
        self.thought_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        left_layout.addWidget(self.thought_label)

        self.thought_field_text = QTextEdit()
        self.thought_field_text.setMinimumSize(1240, 600)
        self.thought_field_text.setReadOnly(True)
        self.thought_field_text.setStyleSheet("font-size: 16px;")
        left_layout.addWidget(self.thought_field_text)

        self.final_decision_label = QLabel("Final Response:")
        self.final_decision_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        left_layout.addWidget(self.final_decision_label)

        self.final_decision_field_text = QTextEdit()
        self.final_decision_field_text.setReadOnly(True)
        self.final_decision_field_text.setStyleSheet("font-size: 16px;")
        left_layout.addWidget(self.final_decision_field_text)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_ai)
        left_layout.addWidget(self.start_button)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.container = QWidget()
        self.container.setLayout(main_layout)
        self.setCentralWidget(self.container)

        # === Signals for thoughts field update ===
        self.signal_emitter.start_iteration_thoughts_field.connect(self.add_current_iteration_to_field_thtsf)
        self.signal_emitter.update_thoughts_field.connect(self.append_thoughts_text)
        self.signal_emitter.thought_field_segment.connect(self.add_new_lines_thtsf)
        self.signal_emitter.endline_for_thtsf.connect(self.endline_for_thtsf)
        # === Signal for final decision field update ===
        self.signal_emitter.update_final_decision_field.connect(self.append_final_decision_text)
        # === Signals for checkup field update ===
        self.signal_emitter.start_iteration_checkup_field.connect(self.add_current_iteration_to_field_chupf)
        self.signal_emitter.update_checkup_field.connect(self.append_checkup_field_text)
        self.signal_emitter.checkup_field_segment.connect(self.add_new_lines_chupf)
        self.signal_emitter.mode_action.connect(self.chosen_mode)
        self.signal_emitter.summary_action.connect(self.begin_summary)
        self.signal_emitter.endline_for_chupf.connect(self.endline_for_chupf)
        # === Signal connect for current image print ===
        self.signal_emitter.update_image.connect(self.update_image_label)
        # ===  ===
        self.signal_emitter.transition_lock.connect(self.disable_start_button)
        self.signal_emitter.unlock_transition_lock.connect(self.enable_start_button)
        # ===  ===
        self.signal_emitter.clear_final_decision_field_if_not_empty.connect(self.empty_check_final_decision_field)

    ###Functions for actions and use of GPT pages###
    # === Start ai process in ArtificialIntelligenceNode ===
    def start_ai(self):
        self.button_clicked = "Start"
        self.gpt_node.start_AI_processing()

    # === Functions related to though process field ===
    # Print current iteration that is in process
    def add_current_iteration_to_field_thtsf(self, iteration):
        self.thought_field_text.append(f"iteration ({iteration}): \n")
    # Used to add streamed llm output to thoughts field
    def append_thoughts_text(self, text):
        cursor = self.thought_field_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.thought_field_text.setTextCursor(cursor)
        self.thought_field_text.insertPlainText(text)
    # Add a new line with segmentation in thoughts field
    def add_new_lines_thtsf(self):
        self.thought_field_text.append("\n====================================================================================\n\n")
    # Add a new line in thoughts field
    def endline_for_thtsf(self):
        self.thought_field_text.append("\n")

    # === Functions related to final decision field ===
    # Used to add streamed llm output to final decision field
    def append_final_decision_text(self, text):
        cursor = self.final_decision_field_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.final_decision_field_text.setTextCursor(cursor)
        self.final_decision_field_text.insertPlainText(text)

    # === Functions related to check-up field ===
    # Print current iteration that is in process
    def add_current_iteration_to_field_chupf(self, iteration):
        self.checkup_field_text.append(f"iteration ({iteration}): \n")
    # Used to add streamed llm output to the chechup field
    def append_checkup_field_text(self, text):
        cursor = self.checkup_field_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.checkup_field_text.setTextCursor(cursor)
        self.checkup_field_text.insertPlainText(text)
    # Print before mode print
    def chosen_mode(self, mode):
        self.checkup_field_text.append(f"mode: [{mode}] \n")
    # Print before summary stream of llm
    def begin_summary(self):
        self.checkup_field_text.append("Summary: \n")
    # Add a new line with segmentation in checkup field
    def add_new_lines_chupf(self):
        self.checkup_field_text.append("\n==============================================\n")
    # Print and endline in checkup field
    def endline_for_chupf(self):
        self.checkup_field_text.append("\n")

    # === Functions related to image/camera footage ===
    # Update label to display new image on ui
    def update_image_label(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

    def enable_start_button(self):
        self.start_button.setEnabled(True)
    def disable_start_button(self):
        self.start_button.setEnabled(False)

    def empty_check_final_decision_field(self):
        if self.final_decision_field_text.toPlainText().strip() != "":
            self.final_decision_field_text.clear()

if __name__ == "__main__":
    signal_emitter = AppSignalEmitter()

    GPT_node = GPTThread(signal_emitter)

    app = QApplication([])
    window = GPTWindow(signal_emitter, GPT_node)
    window.show()
    app.exec_()