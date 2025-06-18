# Value-Aware AI

This project explores the integration of **Value-Awareness** into Artificial Intelligence, with a focus on robotic agents. It includes a prototype implementation on a **TurtleBot4**, simulating a value-aware vacuum cleaner that decides whether or not to clean a room based on visual context and inferred human values.

---

## Repository Structure

This repository includes two main components:

### 1. `Value_Aware_AI_Test_Code/`

An application designed for offline testing of the value-aware decision-making process:

- Loads an image from a specified folder.
- Sends the image to a value-aware AI agent (powered by GPT-4o or similar).
- The AI evaluates the scene and determines if it is appropriate to initiate cleaning, considering contextual, ethical, and social cues.

> Ideal for testing without robot hardware.

---

### 2. `Value_Aware_Vacuum_Cleaner_Code/`

A full application integrated with **TurtleBot4** using **PyQt5** for a graphical interface. The interface includes three main pages:

#### Page 1: Debug Page

- Manual control over the robot:
  - Dock / Undock
  - Trigger specific states (e.g., cleaning, observing, docking)
- Debug output field for internal state tracking

#### Page 2: GPT Page 1

- Standalone AI analysis, identical to `Value_Aware_AI_Test_Code`
- Loads a single image from the PC
- Sends it to the value-aware AI
- Displays the decision and reasoning

#### Page 3: GPT Page 2

- Displays live camera feed from the robot
- Feeds the image to the AI for real-time scene analysis
- Shows AI-generated decision and reasoning in the UI

> Both GPT pages mode decision control the robot.

---

## Paper

For an in-depth explanation of the concepts, methodology, and implementation details of this project, please refer to the following paper:

**Paper:** `Paper - Value-Aware Vacuum Cleaner - 2024-2025 - Senne Lenaerts - EN.pdf`

This document includes:
- Background on value-aware AI
- Architecture and design choices
- Implementation challenges
- Test results and analysis
