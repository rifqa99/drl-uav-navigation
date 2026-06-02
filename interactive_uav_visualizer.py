import os
# fmt: off
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
# fmt: on
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox)
from agents.dqn_agent import DQNAgent
from env.uav_env_dynamic import UAVLiDARDynamicEnv
from PyQt5.QtCore import QTimer, Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import os
import numpy as np
from collections import deque
import torch

import matplotlib
matplotlib.use('Qt5Agg')


# Direct structural imports from your verified codebase layout


class UAVInteractiveVisualizer(QMainWindow):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.setWindowTitle(
            "Advanced UAV Dynamic Navigation Ground Control Station")
        self.setGeometry(100, 100, 1300, 850)

        self.checkpoint_path = checkpoint_path
        self.device = "cpu"
        self.stack_size = 3
        self.n_obstacles = 4  # Default slider initial setting

        # Core Sim Execution Control State Variables
        self.env = None
        self.agent = None
        self.frame_stack = None
        self.state = None
        self.sim_loop_active = False

        # Initialize UI Layout panels
        self.init_ui()

        # Bootstrap RL Framework and Environment Elements
        self.reset_simulation_environment()

        # Set up a high-precision PyQt refresh timer loop for real-time physics drawing
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_single_sim_frame)
        # Triggers a sleek ~25 Frames Per Second refresh rate
        self.timer.start(40)

    def init_ui(self):
        # Create Main window layout splitters
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left Panel: Matplotlib Render Display Canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, stretch=4)

        # Right Panel: Interactive Setting Sidebar Controls
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        main_layout.addWidget(sidebar_widget, stretch=1)

        # Group Box 1: Map Difficulty Modification Parameters
        config_box = QGroupBox("Environment Architecture")
        config_box_layout = QVBoxLayout()

        self.obs_label = QLabel(f"Dynamic Obstacles: {self.n_obstacles}")
        config_box_layout.addWidget(self.obs_label)

        self.obs_slider = QSlider(Qt.Horizontal)
        self.obs_slider.setMinimum(0)
        self.obs_slider.setMaximum(12)
        self.obs_slider.setValue(self.n_obstacles)
        self.obs_slider.setTickPosition(QSlider.TicksBelow)
        self.obs_slider.setTickInterval(2)
        self.obs_slider.valueChanged.connect(self.handle_slider_movement)
        config_box_layout.addWidget(self.obs_slider)

        config_box.setLayout(config_box_layout)
        sidebar_layout.addWidget(config_box)

        # Group Box 2: Execution Hotkeys Panel
        control_box = QGroupBox("Execution Controls")
        control_box_layout = QVBoxLayout()

        self.play_btn = QPushButton("Launch Simulation Sequence")
        self.play_btn.clicked.connect(self.toggle_playback_state)
        control_box_layout.addWidget(self.play_btn)

        reset_btn = QPushButton("Generate New Unseen Map Layout")
        reset_btn.clicked.connect(self.reset_simulation_environment)
        control_box_layout.addWidget(reset_btn)

        control_box.setLayout(control_box_layout)
        sidebar_layout.addWidget(control_box)

        # Group Box 3: Live Telemetry Telecommunication Panel
        telemetry_box = QGroupBox("Live Flight Telemetry Stream")
        self.telemetry_layout = QVBoxLayout()
        self.telemetry_labels = {
            "steps": QLabel("Simulation Frame Step: 0"),
            "speed": QLabel("Current Linear Velocity: 0.00 m/s"),
            "dist": QLabel("Distance to Touchdown Target: 0.00 m"),
            "status": QLabel("Mission Status Operational Context: INITIALIZED")
        }
        for label in self.telemetry_labels.values():
            self.telemetry_layout.addWidget(label)
        telemetry_box.setLayout(self.telemetry_layout)
        sidebar_layout.addWidget(telemetry_box)

        sidebar_layout.addStretch()

        # Set dark-mode Ground Control Station styling sheets
        self.setStyleSheet("""
            QMainWindow { background-color: #1e222b; }
            QGroupBox { color: #ffffff; font-weight: bold; border: 2px solid #3f4452; border-radius: 6px; margin-top: 12px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            QLabel { color: #abb2bf; font-size: 13px; }
            QPushButton { background-color: #4b5263; color: white; border: none; border-radius: 4px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #61afef; color: #1e222b; }
        """)

    def handle_slider_movement(self, value):
        self.n_obstacles = value
        self.obs_label.setText(f"Dynamic Obstacles: {self.n_obstacles}")
        self.reset_simulation_environment()

    def toggle_playback_state(self):
        self.sim_loop_active = not self.sim_loop_active
        if self.sim_loop_active:
            self.play_btn.setText("Pause Autonomous Mode")
            self.play_btn.setStyleSheet(
                "background-color: #98c379; color: #1e222b;")
        else:
            self.play_btn.setText("Resume Autonomous Mode")
            self.play_btn.setStyleSheet(
                "background-color: #4b5263; color: white;")

    def reset_simulation_environment(self):
        # 1. Instantiate environment matching selected density configurations
        self.env = UAVLiDARDynamicEnv(n_obstacles=self.n_obstacles)

        # 2. Instantiate or adapt model matching state spatial properties
        obs_dim = self.env.observation_space.shape[0] * self.stack_size
        action_dim = self.env.action_space.n

        self.agent = DQNAgent(
            state_dim=obs_dim, action_dim=action_dim, device=self.device)

        # Load weights if path exists cleanly
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False)
            self.agent.q_network.load_state_dict(
                checkpoint["model_state_dict"])
            self.agent.epsilon = 0.0  # Lock evaluation protocol to deterministic pure exploitation
            self.agent.q_network.eval()
            self.telemetry_labels["status"].setText(
                "Status: LOADED CONVERGED DDQN WEIGHTS")
        else:
            self.telemetry_labels["status"].setText(
                "Status: EXECUTING RANDOM UNTRAINED EXPLORATION")

        # Reset runtime states
        obs, _ = self.env.reset()
        self.frame_stack = deque(
            [obs] * self.stack_size, maxlen=self.stack_size)
        self.state = np.concatenate(list(self.frame_stack), axis=0)

        self.telemetry_labels["status"].setStyleSheet("color: #61afef;")

        # Redraw pure initial environment profile map setup
        self.render_map_frame()

    def draw_realistic_quadcopter(self, pos, theta, radius=0.3):
        """Draws a clean, stylized quadcopter drone vector on the canvas layer."""
        # 1. Main cross arms layout beams
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_t45, sin_t45 = np.cos(theta + np.pi/4), np.sin(theta + np.pi/4)
        cos_t135, sin_t135 = np.cos(
            theta + 3*np.pi/4), np.sin(theta + 3*np.pi/4)

        # Draw physical carbon cross structural members
        arm1_x = [pos[0] - radius * cos_t45, pos[0] + radius * cos_t45]
        arm1_y = [pos[1] - radius * sin_t45, pos[1] + radius * sin_t45]
        arm2_x = [pos[0] - radius * cos_t135, pos[0] + radius * cos_t135]
        arm2_y = [pos[1] - radius * sin_t135, pos[1] + radius * sin_t135]

        self.ax.plot(arm1_x, arm1_y, color='#5c6370', linewidth=3, zorder=4)
        self.ax.plot(arm2_x, arm2_y, color='#5c6370', linewidth=3, zorder=4)

        # 2. Draw Rotor housing rings and propellers
        rotor_centers_x = [pos[0] + radius * cos_t45, pos[0] - radius *
                           cos_t45, pos[0] + radius * cos_t135, pos[0] - radius * cos_t135]
        rotor_centers_y = [pos[1] + radius * sin_t45, pos[1] - radius *
                           sin_t45, pos[1] + radius * sin_t135, pos[1] - radius * sin_t135]

        for idx, (rx, ry) in enumerate(zip(rotor_centers_x, rotor_centers_y)):
            # Distinguish front motor pairs using orange accent rings to monitor flight orientation
            ring_color = '#e5c07b' if idx in [0, 2] else '#4b5263'
            rotor_ring = plt.Circle(
                (rx, ry), radius*0.3, color=ring_color, fill=False, linewidth=1.5, zorder=5)
            self.ax.add_patch(rotor_ring)

            # Tiny propeller swoops lines inside the motor rings
            self.ax.plot([rx - radius*0.2, rx + radius*0.2],
                         [ry, ry], color='#abb2bf', linewidth=1, zorder=5)

        # 3. Central avionics fuselage hull pod shell
        center_pod = plt.Circle(
            (pos[0], pos[1]), radius*0.4, color='#61afef', fill=True, zorder=6)
        self.ax.add_patch(center_pod)

        # Draw explicit directional nose cone heading indicator pointing vector arrow
        self.ax.quiver(pos[0], pos[1], cos_t, sin_t,
                       color='#ffffff', scale=15, width=0.015, zorder=7)

    def render_map_frame(self):
        self.ax.clear()

        # Setup aesthetic dark mode digital grid environments background
        self.ax.set_facecolor('#21252b')
        self.fig.patch.set_facecolor('#1e222b')
        self.ax.set_xlim(0, self.env.world_size)
        self.ax.set_ylim(0, self.env.world_size)
        self.ax.grid(True, linestyle=':', color='#3e4451', alpha=0.6)

        # Clean up tick colors to blend cleanly into the ground station layout structure
        self.ax.tick_params(colors='#abb2bf', labelsize=10)

        # Draw Glow-style Touchdown Goal Circle Target landing pad zone
        goal_outer = plt.Circle(
            self.env.goal, self.env.goal_radius, color='#98c379', alpha=0.15, zorder=1)
        goal_inner = plt.Circle(
            self.env.goal, self.env.goal_radius*0.3, color='#98c379', alpha=0.4, zorder=2)
        self.ax.add_patch(goal_outer)
        self.ax.add_patch(goal_inner)
        self.ax.plot(self.env.goal[0], self.env.goal[1], color='#98c379', marker='P', markersize=12, alpha=0.7, zorder=2)
        # Render historical trailing path lines
        if len(self.env.trajectory) > 1:
            traj = np.array(self.env.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], color='#61afef',
                         linestyle=':', linewidth=1.5, alpha=0.7, zorder=3)

        # Draw Dynamic obstacles shapes
        for center, radius in self.env.obstacles:
            # Modern safety boundary visualization design accents
            hazard_glow = plt.Circle(
                center, radius, color='#e06c75', alpha=0.25, zorder=2)
            hazard_core = plt.Circle(
                center, radius*0.8, color='#e06c75', alpha=0.4, zorder=2)
            self.ax.add_patch(hazard_glow)
            self.ax.add_patch(hazard_core)

        # Inject our stylized high-fashion quadcopter rendering function layer
        self.draw_realistic_quadcopter(
            self.env.pos, self.env.theta, radius=0.35)

        # Force Matplotlib frame buffer redraw flip
        self.canvas.draw()

    def run_single_sim_frame(self):
        # Halt execution frames if paused by the user dashboard button
        if not self.sim_loop_active:
            return

        # Select action matching greedy inference weights
        action = self.agent.select_action(self.state)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Slide memory stack
        self.frame_stack.append(next_obs)
        self.state = np.concatenate(list(self.frame_stack), axis=0)

        # Push update updates to text panels
        self.telemetry_labels["steps"].setText(
            f"Simulation Frame Step: {self.env.steps}")
        self.telemetry_labels["speed"].setText(
            f"Current Linear Velocity: {info['speed']:.2f} m/s")
        self.telemetry_labels["dist"].setText(
            f"Distance to Touchdown Target: {info['distance_to_goal']:.2f} m")

        # Check condition triggers
        if done:
            self.sim_loop_active = False
            self.play_btn.setText("Simulation Terminal Hit")
            self.play_btn.setStyleSheet(
                "background-color: #e06c75; color: white;")

            if info.get('reached_goal', False):
                self.telemetry_labels["status"].setText(
                    "Mission Status: SUCCESSFUL TOUCHDOWN achieved!")
                self.telemetry_labels["status"].setStyleSheet(
                    "color: #98c379; font-weight: bold;")
            else:
                self.telemetry_labels["status"].setText(
                    "Mission Status: HARD COLLISION CRASH DETECTED")
                self.telemetry_labels["status"].setStyleSheet(
                    "color: #e06c75; font-weight: bold;")

        # Trigger canvas redraw step
        self.render_map_frame()


if __name__ == "__main__":
    # Point directly to your final 4000/6000 episode checkpoint file saved on your system drive
    checkpoint_file = "outputs/checkpoints/dqn_adaptive_obs_6_ep_6000.pth"

    app = QApplication(sys.argv)
    gcs_window = UAVInteractiveVisualizer(checkpoint_path=checkpoint_file)
    gcs_window.show()
    sys.exit(app.exec_())
