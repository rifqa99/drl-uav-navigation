import os
# Force CPU for PyQt visualizer
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import torch
import numpy as np
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox,
    QSpinBox
)
from PyQt5.QtCore import QTimer, Qt

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from agents.dqn_agent import DQNAgent
from env.uav_env_dynamic import UAVLiDARDynamicEnv


class UAVInteractiveVisualizer(QMainWindow):
    def __init__(self, checkpoint_path=None):
        super().__init__()

        self.setWindowTitle("Advanced UAV Dynamic Navigation Ground Control Station")
        self.setGeometry(100, 100, 1300, 850)

        self.checkpoint_path = checkpoint_path
        self.device = "cpu"
        self.stack_size = 3

        self.n_obstacles = 4
        self.current_seed = 42

        self.env = None
        self.agent = None
        self.frame_stack = None
        self.state = None
        self.sim_loop_active = False

        self.init_ui()
        self.reset_simulation_environment()

        self.timer = QTimer()
        self.timer.timeout.connect(self.run_single_sim_frame)
        self.timer.start(40)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, stretch=4)

        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        main_layout.addWidget(sidebar_widget, stretch=1)

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
        self.obs_slider.valueChanged.connect(self.handle_obstacle_slider)
        config_box_layout.addWidget(self.obs_slider)

        self.seed_label = QLabel(f"Seed: {self.current_seed}")
        config_box_layout.addWidget(self.seed_label)

        self.seed_box = QSpinBox()
        self.seed_box.setMinimum(0)
        self.seed_box.setMaximum(999999)
        self.seed_box.setValue(self.current_seed)
        self.seed_box.valueChanged.connect(self.handle_seed_change)
        config_box_layout.addWidget(self.seed_box)

        random_seed_btn = QPushButton("Generate Random Seed")
        random_seed_btn.clicked.connect(self.generate_random_seed)
        config_box_layout.addWidget(random_seed_btn)

        config_box.setLayout(config_box_layout)
        sidebar_layout.addWidget(config_box)

        control_box = QGroupBox("Execution Controls")
        control_box_layout = QVBoxLayout()

        self.play_btn = QPushButton("Launch Simulation Sequence")
        self.play_btn.clicked.connect(self.toggle_playback_state)
        control_box_layout.addWidget(self.play_btn)

        reset_btn = QPushButton("Reset Same Seed Map")
        reset_btn.clicked.connect(self.reset_simulation_environment)
        control_box_layout.addWidget(reset_btn)

        control_box.setLayout(control_box_layout)
        sidebar_layout.addWidget(control_box)

        telemetry_box = QGroupBox("Live Flight Telemetry Stream")
        telemetry_layout = QVBoxLayout()

        self.telemetry_labels = {
            "seed": QLabel(f"Seed: {self.current_seed}"),
            "steps": QLabel("Simulation Frame Step: 0"),
            "speed": QLabel("Current Linear Velocity: 0.00 m/s"),
            "dist": QLabel("Distance to Touchdown Target: 0.00 m"),
            "dmin": QLabel("Minimum LiDAR Distance: 0.00 m"),
            "status": QLabel("Mission Status: INITIALIZED"),
        }

        for label in self.telemetry_labels.values():
            telemetry_layout.addWidget(label)

        telemetry_box.setLayout(telemetry_layout)
        sidebar_layout.addWidget(telemetry_box)

        sidebar_layout.addStretch()

        self.setStyleSheet("""
            QMainWindow { background-color: #1e222b; }
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                border: 2px solid #3f4452;
                border-radius: 6px;
                margin-top: 12px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QLabel {
                color: #abb2bf;
                font-size: 13px;
            }
            QPushButton {
                background-color: #4b5263;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #61afef;
                color: #1e222b;
            }
            QSpinBox {
                background-color: #282c34;
                color: white;
                border: 1px solid #3f4452;
                padding: 4px;
            }
        """)

    def handle_obstacle_slider(self, value):
        self.n_obstacles = value
        self.obs_label.setText(f"Dynamic Obstacles: {self.n_obstacles}")
        self.reset_simulation_environment()

    def handle_seed_change(self, value):
        self.current_seed = int(value)
        self.seed_label.setText(f"Seed: {self.current_seed}")
        self.telemetry_labels["seed"].setText(f"Seed: {self.current_seed}")
        self.reset_simulation_environment()

    def generate_random_seed(self):
        self.current_seed = int(np.random.randint(0, 999999))
        self.seed_box.setValue(self.current_seed)
        self.seed_label.setText(f"Seed: {self.current_seed}")
        self.telemetry_labels["seed"].setText(f"Seed: {self.current_seed}")
        self.reset_simulation_environment()

    def toggle_playback_state(self):
        self.sim_loop_active = not self.sim_loop_active

        if self.sim_loop_active:
            self.play_btn.setText("Pause Autonomous Mode")
            self.play_btn.setStyleSheet("background-color: #98c379; color: #1e222b;")
        else:
            self.play_btn.setText("Resume Autonomous Mode")
            self.play_btn.setStyleSheet("background-color: #4b5263; color: white;")

    def reset_simulation_environment(self):
        self.sim_loop_active = False
        self.play_btn.setText("Launch Simulation Sequence")
        self.play_btn.setStyleSheet("background-color: #4b5263; color: white;")

        self.env = UAVLiDARDynamicEnv(
            n_obstacles=self.n_obstacles,
            seed=self.current_seed
        )

        obs_dim = self.env.observation_space.shape[0] * self.stack_size
        action_dim = self.env.action_space.n

        self.agent = DQNAgent(
            state_dim=obs_dim,
            action_dim=action_dim,
            device=self.device
        )

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False
            )

            self.agent.q_network.load_state_dict(checkpoint["model_state_dict"])
            self.agent.epsilon = 0.0
            self.agent.q_network.eval()

            self.telemetry_labels["status"].setText("Status: LOADED TRAINED DDQN WEIGHTS")
            self.telemetry_labels["status"].setStyleSheet("color: #98c379; font-weight: bold;")
        else:
            self.telemetry_labels["status"].setText("Status: RANDOM UNTRAINED EXPLORATION")
            self.telemetry_labels["status"].setStyleSheet("color: #e5c07b; font-weight: bold;")

        obs, _ = self.env.reset(seed=self.current_seed)

        self.frame_stack = deque([obs] * self.stack_size, maxlen=self.stack_size)
        self.state = np.concatenate(list(self.frame_stack), axis=0)

        self.telemetry_labels["seed"].setText(f"Seed: {self.current_seed}")
        self.telemetry_labels["steps"].setText("Simulation Frame Step: 0")
        self.telemetry_labels["speed"].setText("Current Linear Velocity: 0.00 m/s")
        self.telemetry_labels["dist"].setText(
            f"Distance to Touchdown Target: {self.env._distance_to_goal():.2f} m"
        )
        self.telemetry_labels["dmin"].setText("Minimum LiDAR Distance: 0.00 m")

        self.render_map_frame()

    def draw_realistic_quadcopter(self, pos, theta, radius=0.3):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        cos_t45 = np.cos(theta + np.pi / 4)
        sin_t45 = np.sin(theta + np.pi / 4)

        cos_t135 = np.cos(theta + 3 * np.pi / 4)
        sin_t135 = np.sin(theta + 3 * np.pi / 4)

        arm1_x = [pos[0] - radius * cos_t45, pos[0] + radius * cos_t45]
        arm1_y = [pos[1] - radius * sin_t45, pos[1] + radius * sin_t45]

        arm2_x = [pos[0] - radius * cos_t135, pos[0] + radius * cos_t135]
        arm2_y = [pos[1] - radius * sin_t135, pos[1] + radius * sin_t135]

        self.ax.plot(arm1_x, arm1_y, color="#5c6370", linewidth=3, zorder=4)
        self.ax.plot(arm2_x, arm2_y, color="#5c6370", linewidth=3, zorder=4)

        rotor_centers_x = [
            pos[0] + radius * cos_t45,
            pos[0] - radius * cos_t45,
            pos[0] + radius * cos_t135,
            pos[0] - radius * cos_t135,
        ]

        rotor_centers_y = [
            pos[1] + radius * sin_t45,
            pos[1] - radius * sin_t45,
            pos[1] + radius * sin_t135,
            pos[1] - radius * sin_t135,
        ]

        for idx, (rx, ry) in enumerate(zip(rotor_centers_x, rotor_centers_y)):
            # ring_color = "#e5c07b" if idx in [0, 2] else "#4b5263"
            ring_color = "#4b5263"
            rotor_ring = plt.Circle(
                (rx, ry),
                radius * 0.3,
                color=ring_color,
                fill=False,
                linewidth=1.5,
                zorder=5
            )
            self.ax.add_patch(rotor_ring)

            self.ax.plot(
                [rx - radius * 0.2, rx + radius * 0.2],
                [ry, ry],
                color="#abb2bf",
                linewidth=1,
                zorder=5
            )

        center_pod = plt.Circle(
            (pos[0], pos[1]),
            radius * 0.4,
            color="#61afef",
            fill=True,
            zorder=6
        )
        self.ax.add_patch(center_pod)

        # self.ax.quiver(
        #     pos[0],
        #     pos[1],
        #     cos_t,
        #     sin_t,
        #     color="#ffffff",
        #     scale=15,
        #     width=0.015,
        #     zorder=7
        # )

    def render_map_frame(self):
        self.ax.clear()

        self.ax.set_facecolor("#21252b")
        self.fig.patch.set_facecolor("#1e222b")
        self.ax.set_xlim(0, self.env.world_size)
        self.ax.set_ylim(0, self.env.world_size)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle=":", color="#3e4451", alpha=0.6)
        self.ax.tick_params(colors="#abb2bf", labelsize=10)

        goal_outer = plt.Circle(
            self.env.goal,
            self.env.goal_radius,
            color="#98c379",
            alpha=0.15,
            zorder=1
        )
        goal_inner = plt.Circle(
            self.env.goal,
            self.env.goal_radius * 0.3,
            color="#98c379",
            alpha=0.4,
            zorder=2
        )

        self.ax.add_patch(goal_outer)
        self.ax.add_patch(goal_inner)
        self.ax.plot(
            self.env.goal[0],
            self.env.goal[1],
            color="#98c379",
            marker="P",
            markersize=12,
            alpha=0.7,
            zorder=2
        )

        if len(self.env.trajectory) > 1:
            traj = np.array(self.env.trajectory)
            self.ax.plot(
                traj[:, 0],
                traj[:, 1],
                color="#61afef",
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                zorder=3
            )

        for center, radius in self.env.obstacles:
            hazard_glow = plt.Circle(
                center,
                radius,
                color="#e06c75",
                alpha=0.25,
                zorder=2
            )
            hazard_core = plt.Circle(
                center,
                radius * 0.8,
                color="#e06c75",
                alpha=0.4,
                zorder=2
            )

            self.ax.add_patch(hazard_glow)
            self.ax.add_patch(hazard_core)

        self.draw_realistic_quadcopter(
            self.env.pos,
            self.env.theta,
            radius=0.35
        )

        self.ax.set_title(
            f"Seed: {self.current_seed} | Obstacles: {self.n_obstacles}",
            color="white",
            fontsize=11
        )

        self.canvas.draw()

    def run_single_sim_frame(self):
        if not self.sim_loop_active:
            return

        action = self.agent.select_action(self.state)

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.frame_stack.append(next_obs)
        self.state = np.concatenate(list(self.frame_stack), axis=0)

        self.telemetry_labels["steps"].setText(
            f"Simulation Frame Step: {self.env.steps}"
        )
        self.telemetry_labels["speed"].setText(
            f"Current Linear Velocity: {info['speed']:.2f} m/s"
        )
        self.telemetry_labels["dist"].setText(
            f"Distance to Touchdown Target: {info['distance_to_goal']:.2f} m"
        )
        self.telemetry_labels["dmin"].setText(
            f"Minimum LiDAR Distance: {info.get('min_lidar_distance', 0.0):.3f} m"
        )

        if done:
            self.sim_loop_active = False
            self.play_btn.setText("Simulation Terminal Hit")
            self.play_btn.setStyleSheet("background-color: #e06c75; color: white;")

            if info.get("reached_goal", False):
                self.telemetry_labels["status"].setText("Mission Status: SUCCESSFUL TOUCHDOWN")
                self.telemetry_labels["status"].setStyleSheet("color: #98c379; font-weight: bold;")
            elif info.get("collision", False):
                self.telemetry_labels["status"].setText("Mission Status: COLLISION DETECTED")
                self.telemetry_labels["status"].setStyleSheet("color: #e06c75; font-weight: bold;")
            else:
                self.telemetry_labels["status"].setText("Mission Status: TIMEOUT")
                self.telemetry_labels["status"].setStyleSheet("color: #e5c07b; font-weight: bold;")

        self.render_map_frame()


if __name__ == "__main__":
    checkpoint_file = "outputs/checkpoints/dqn_adaptive_obs_8_ep_6000.pth"
 
    app = QApplication(sys.argv)
    gcs_window = UAVInteractiveVisualizer(checkpoint_path=checkpoint_file)
    gcs_window.show()
    sys.exit(app.exec_())