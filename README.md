# Value-Aware AI

This project explores the integration of **Value-Awareness** into Artificial Intelligence, with a focus on robotic agents. It includes a prototype implementation on a **TurtleBot4**, simulating a value-aware vacuum cleaner that decides whether or not to clean a room based on visual context and human values.

---

## Repository Structure

This repository includes two main components:

### 1. `Value_Aware_AI_Test_Code/`
An application that:
- Loads an image from a specified folder.
- Sends that image to a value-aware AI agent.
- The AI evaluates each scene and determines if it is appropriate to initiate cleaning, considering contextual and ethical factors.

### 2. `Value_Aware_Vacuum_Cleaner_code/`
An application that:
- Connects to a **TurtleBot4**.
- Opens an application made in **PyQt5** that has three pages.
   - Page 1: Debug page, this page contains buttons for docking and undocking, as well as buttons for moving into states on a Finite state machine, like cleaning, observing, and docking, along with a debug field.
   - Page 2: GPT page 1, this page does the exact same thing as the application in `Value_Aware_AI_Test_Code/`.
   - Page 3: GPT page 2, this page has all the same fields and buttons as GPT page 1, but instead is connected to the **TurtleBot4** where it shows camera output from the **TurtleBot4** and uses that as input for the AI's Value-Aware decision process.

 ## Paper
 For more details and explenation of this project the paper (`Paper- Value aware vacuum cleaner- 2024-2025 - Senne Lenaerts - EN.pdf`) will contain that.

