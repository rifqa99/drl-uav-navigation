import os

# Force CPU for PyQt visualizer
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import torch
import numpy as np

from pathlib import Path
from collections import deque

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QPushButton,
    QGroupBox,
    QComboBox,
)
from PyQt5.QtCore import QTimer, Qt

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from agents.dqn_agent import DQNAgent

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from env.uav_env import UAVLiDAREnv


class UnifiedUAVGCS(QMainWindow):

    def __init__(
        self,
        dynamic_weights_path=None,
        static_weights_path=None,
    ):
        super().__init__()

        self.setWindowTitle(
            "Unified UAV Mission Control Center"
        )
        self.setGeometry(
            100,
            100,
            1300,
            850
        )

        self.dynamic_weights = dynamic_weights_path
        self.static_weights = static_weights_path

        self.device = "cpu"
        self.stack_size = 3

        # ----------------------------------------------------
        # Runtime configuration
        # ----------------------------------------------------

        self.n_obstacles = 4
        self.regime_type = "Dynamic"

        self.current_seed = np.random.randint(
            0,
            1000000
        )

        self.env = None
        self.agent = None
        self.frame_stack = None
        self.state = None

        self.sim_loop_active = False

        # ----------------------------------------------------
        # NEW:
        # Stores all previous positions for every obstacle.
        #
        # Structure:
        #
        # [
        #     [pos0, pos1, pos2, ...],   # obstacle 0
        #     [pos0, pos1, pos2, ...],   # obstacle 1
        #     ...
        # ]
        # ----------------------------------------------------

        self.obstacle_position_history = []

        # Gray ghost appearance
        self.obstacle_ghost_color = "#9aa0a6"
        self.obstacle_ghost_alpha = 0.05

        # Record one obstacle ghost every 8 simulation steps
        self.ghost_stride = 10
        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.init_ui()
        self.reset_simulation_environment()

        # Approximately 25 FPS
        self.timer = QTimer()
        self.timer.timeout.connect(
            self.run_single_sim_frame
        )
        self.timer.start(40)


    # ========================================================
    # USER INTERFACE
    # ========================================================

    def init_ui(self):

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(
            main_widget
        )

        # ----------------------------------------------------
        # Left panel: Matplotlib
        # ----------------------------------------------------

        self.fig, self.ax = plt.subplots(
            figsize=(8, 8)
        )

        self.canvas = FigureCanvas(
            self.fig
        )

        main_layout.addWidget(
            self.canvas,
            stretch=4
        )

        # ----------------------------------------------------
        # Right sidebar
        # ----------------------------------------------------

        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(
            sidebar_widget
        )

        main_layout.addWidget(
            sidebar_widget,
            stretch=1
        )

        # ====================================================
        # GROUP 1: Environment
        # ====================================================

        config_box = QGroupBox(
            "Mission Target Setup"
        )

        config_box_layout = QVBoxLayout()

        self.obs_label = QLabel(
            f"Obstacle Saturated Count: {self.n_obstacles}"
        )

        config_box_layout.addWidget(
            self.obs_label
        )

        self.obs_slider = QSlider(
            Qt.Horizontal
        )

        self.obs_slider.setMinimum(0)
        self.obs_slider.setMaximum(12)
        self.obs_slider.setValue(
            self.n_obstacles
        )

        self.obs_slider.setTickPosition(
            QSlider.TicksBelow
        )

        self.obs_slider.setTickInterval(1)

        self.obs_slider.valueChanged.connect(
            self.handle_obstacle_slider
        )

        config_box_layout.addWidget(
            self.obs_slider
        )

        regime_label = QLabel(
            "Operating Flight Regime:"
        )

        config_box_layout.addWidget(
            regime_label
        )

        self.regime_dropdown = QComboBox()

        self.regime_dropdown.addItems(
            [
                "Dynamic",
                "Static",
            ]
        )

        self.regime_dropdown.currentTextChanged.connect(
            self.handle_regime_change
        )

        config_box_layout.addWidget(
            self.regime_dropdown
        )

        config_box.setLayout(
            config_box_layout
        )

        sidebar_layout.addWidget(
            config_box
        )

        # ====================================================
        # GROUP 2: Controls
        # ====================================================

        control_box = QGroupBox(
            "Execution Controls"
        )

        control_box_layout = QVBoxLayout()

        self.play_btn = QPushButton(
            "Simulate Route Sequence"
        )

        self.play_btn.clicked.connect(
            self.toggle_playback_state
        )

        control_box_layout.addWidget(
            self.play_btn
        )

        reset_btn = QPushButton(
            "Reset Same Map Configuration"
        )

        reset_btn.clicked.connect(
            self.reset_simulation_environment
        )

        control_box_layout.addWidget(
            reset_btn
        )

        next_seed_btn = QPushButton(
            "Generate Unseen Map"
        )

        next_seed_btn.clicked.connect(
            self.generate_random_map
        )

        control_box_layout.addWidget(
            next_seed_btn
        )

        control_box.setLayout(
            control_box_layout
        )

        sidebar_layout.addWidget(
            control_box
        )

        # ====================================================
        # GROUP 3: Telemetry
        # ====================================================

        telemetry_box = QGroupBox(
            "Live Flight Telemetry Stream"
        )

        telemetry_layout = QVBoxLayout()

        self.telemetry_labels = {

            "steps": QLabel(
                "Simulation Frame Step: 0"
            ),

            "speed": QLabel(
                "Current Linear Velocity: 0.00 m/s"
            ),

            "dmin": QLabel(
                "Minimum LiDAR Distance: 0.000 m"
            ),

            "dist": QLabel(
                "Distance to Touchdown Target: 0.00 m"
            ),

            "status": QLabel(
                "Mission Status: INITIALIZED"
            ),
        }

        for label in self.telemetry_labels.values():
            telemetry_layout.addWidget(
                label
            )

        telemetry_box.setLayout(
            telemetry_layout
        )

        sidebar_layout.addWidget(
            telemetry_box
        )

        sidebar_layout.addStretch()

        # ----------------------------------------------------
        # Styling
        # ----------------------------------------------------

        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e222b;
            }

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

            QComboBox {
                background-color: #282c34;
                color: white;
                border: 1px solid #3f4452;
                padding: 4px;
                border-radius: 3px;
            }
        """)


    # ========================================================
    # CALLBACKS
    # ========================================================

    def handle_obstacle_slider(
        self,
        value
    ):

        self.n_obstacles = value

        self.obs_label.setText(
            f"Obstacle Saturated Count: {self.n_obstacles}"
        )

        self.current_seed = np.random.randint(
            0,
            1000000
        )

        self.reset_simulation_environment()


    def handle_regime_change(
        self,
        text
    ):

        self.regime_type = text

        self.reset_simulation_environment()


    def generate_random_map(self):

        self.current_seed = np.random.randint(
            0,
            1000000
        )

        self.reset_simulation_environment()


    def toggle_playback_state(self):

        self.sim_loop_active = (
            not self.sim_loop_active
        )

        if self.sim_loop_active:

            self.play_btn.setText(
                "Pause Autonomous Mode"
            )

            self.play_btn.setStyleSheet(
                "background-color: #98c379; "
                "color: #1e222b;"
            )

        else:

            self.play_btn.setText(
                "Resume Autonomous Mode"
            )

            self.play_btn.setStyleSheet(
                "background-color: #4b5263; "
                "color: white;"
            )


    # ========================================================
    # ENVIRONMENT RESET
    # ========================================================

    def reset_simulation_environment(self):

        self.sim_loop_active = False

        self.play_btn.setText(
            "Simulate Route Sequence"
        )

        self.play_btn.setStyleSheet(
            "background-color: #4b5263; "
            "color: white;"
        )

        # ----------------------------------------------------
        # Select environment
        # ----------------------------------------------------

        if self.regime_type == "Dynamic":

            self.env = UAVLiDARDynamicEnv(
                n_obstacles=self.n_obstacles,
                seed=self.current_seed
            )

            target_path = self.dynamic_weights

        else:

            self.env = UAVLiDAREnv(
                n_obstacles=self.n_obstacles,
                seed=self.current_seed
            )

            target_path = self.static_weights

        # ----------------------------------------------------
        # Agent
        # ----------------------------------------------------

        obs_dim = (
            self.env.observation_space.shape[0]
            * self.stack_size
        )

        action_dim = (
            self.env.action_space.n
        )

        self.agent = DQNAgent(
            state_dim=obs_dim,
            action_dim=action_dim,
            device=self.device
        )

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        if (
            target_path
            and Path(target_path).exists()
        ):

            checkpoint = torch.load(
                target_path,
                map_location=self.device,
                weights_only=False
            )

            self.agent.q_network.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            self.agent.epsilon = 0.0

            self.agent.q_network.eval()

            self.telemetry_labels[
                "status"
            ].setText(
                f"Status: LOADED ANCHOR "
                f"{self.regime_type.upper()} MODEL"
            )

            self.telemetry_labels[
                "status"
            ].setStyleSheet(
                "color: #98c379; "
                "font-weight: bold;"
            )

        else:

            self.telemetry_labels[
                "status"
            ].setText(
                "Status: RANDOM UNTRAINED SEED BALANCING"
            )

            self.telemetry_labels[
                "status"
            ].setStyleSheet(
                "color: #e5c07b; "
                "font-weight: bold;"
            )

        # ----------------------------------------------------
        # Reset environment
        # ----------------------------------------------------

        obs, _ = self.env.reset(
            seed=self.current_seed
        )

        # ----------------------------------------------------
        # NEW:
        # initialize obstacle histories
        # ----------------------------------------------------

        if self.regime_type == "Dynamic":

            self.obstacle_position_history = [

                [
                    center.copy()
                ]

                for center, radius
                in self.env.obstacles
            ]

        else:

            self.obstacle_position_history = []

        # ----------------------------------------------------
        # Frame stack
        # ----------------------------------------------------

        self.frame_stack = deque(
            [obs] * self.stack_size,
            maxlen=self.stack_size
        )

        self.state = np.concatenate(
            list(self.frame_stack),
            axis=0
        )

        # ----------------------------------------------------
        # Reset telemetry
        # ----------------------------------------------------

        self.telemetry_labels[
            "steps"
        ].setText(
            "Simulation Frame Step: 0"
        )

        self.telemetry_labels[
            "speed"
        ].setText(
            "Current Linear Velocity: 0.00 m/s"
        )

        self.telemetry_labels[
            "dmin"
        ].setText(
            "Minimum LiDAR Distance: 0.000 m"
        )

        self.telemetry_labels[
            "dist"
        ].setText(
            f"Distance to Touchdown Target: "
            f"{self.env._distance_to_goal():.2f} m"
        )

        self.render_map_frame()


    # ========================================================
    # UAV DRAWING
    # ========================================================

    def draw_realistic_quadcopter(
        self,
        pos,
        theta,
        radius=0.35
    ):

        cos_t45 = np.cos(
            theta + np.pi / 4
        )

        sin_t45 = np.sin(
            theta + np.pi / 4
        )

        cos_t135 = np.cos(
            theta + 3 * np.pi / 4
        )

        sin_t135 = np.sin(
            theta + 3 * np.pi / 4
        )

        # Arms
        self.ax.plot(

            [
                pos[0] - radius * cos_t45,
                pos[0] + radius * cos_t45
            ],

            [
                pos[1] - radius * sin_t45,
                pos[1] + radius * sin_t45
            ],

            color="#5c6370",
            linewidth=3,
            zorder=4
        )

        self.ax.plot(

            [
                pos[0] - radius * cos_t135,
                pos[0] + radius * cos_t135
            ],

            [
                pos[1] - radius * sin_t135,
                pos[1] + radius * sin_t135
            ],

            color="#5c6370",
            linewidth=3,
            zorder=4
        )

        # Rotor positions
        rotors_x = [

            pos[0] + radius * cos_t45,
            pos[0] - radius * cos_t45,
            pos[0] + radius * cos_t135,
            pos[0] - radius * cos_t135
        ]

        rotors_y = [

            pos[1] + radius * sin_t45,
            pos[1] - radius * sin_t45,
            pos[1] + radius * sin_t135,
            pos[1] - radius * sin_t135
        ]

        for rx, ry in zip(
            rotors_x,
            rotors_y
        ):

            self.ax.add_patch(

                plt.Circle(

                    (rx, ry),

                    radius * 0.3,

                    color="#4b5263",

                    fill=False,

                    linewidth=1.5,

                    zorder=5
                )
            )

            self.ax.plot(

                [
                    rx - radius * 0.2,
                    rx + radius * 0.2
                ],

                [ry, ry],

                color="#abb2bf",

                linewidth=1,

                zorder=5
            )

        # Central body
        self.ax.add_patch(

            plt.Circle(

                (
                    pos[0],
                    pos[1]
                ),

                radius * 0.4,

                color="#61afef",

                fill=True,

                zorder=6
            )
        )


    # ========================================================
    # MAP RENDER
    # ========================================================

    def render_map_frame(self):

        self.ax.clear()

        # ----------------------------------------------------
        # World
        # ----------------------------------------------------

        self.ax.set_facecolor(
            "#21252b"
        )

        self.fig.patch.set_facecolor(
            "#1e222b"
        )

        self.ax.set_xlim(
            0,
            self.env.world_size
        )

        self.ax.set_ylim(
            0,
            self.env.world_size
        )

        self.ax.set_aspect(
            "equal",
            adjustable="box"
        )

        self.ax.grid(
            True,
            linestyle=":",
            color="#3e4451",
            alpha=0.6
        )

        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # ====================================================
        # GOAL
        # ====================================================

        self.ax.add_patch(

            plt.Circle(

                self.env.goal,

                self.env.goal_radius,

                color="#98c379",

                alpha=0.15,

                zorder=1
            )
        )

        self.ax.add_patch(

            plt.Circle(

                self.env.goal,

                self.env.goal_radius * 0.3,

                color="#98c379",

                alpha=0.4,

                zorder=2
            )
        )

        self.ax.plot(

            self.env.goal[0],

            self.env.goal[1],

            color="#98c379",

            marker="P",

            markersize=12,

            alpha=0.7,

            zorder=2
        )

        # ====================================================
        # UAV START
        # ====================================================

        start_pos = self.env.trajectory[0]

        self.ax.plot(

            start_pos[0],

            start_pos[1],

            marker="^",

            color="#61afef",

            markersize=10,

            zorder=5
        )

        # ====================================================
        # UAV TRAJECTORY
        # ====================================================

        if len(
            self.env.trajectory
        ) > 1:

            traj = np.array(
                self.env.trajectory
            )

            self.ax.plot(

                traj[:, 0],

                traj[:, 1],

                color="#61afef",

                linestyle=":",

                linewidth=1.5,

                alpha=0.7,

                zorder=3
            )

        # ====================================================
        # NEW:
        # DYNAMIC OBSTACLE GHOST HISTORY
        # ====================================================

        if self.regime_type == "Dynamic":

            for obstacle_index, history in enumerate(
                self.obstacle_position_history
            ):

                if obstacle_index >= len(
                    self.env.obstacles
                ):
                    continue

                radius = (
                    self.env.obstacles[
                        obstacle_index
                    ][1]
                )

                # --------------------------------------------
                # Draw OLD frame positions only.
                #
                # history[:-1]
                # excludes current frame so that current
                # obstacle remains clearly red.
                # --------------------------------------------

                for old_center in history[:-1]:

                    # Outer footprint
                    self.ax.add_patch(

                        plt.Circle(

                            old_center,

                            radius,

                            facecolor=self.obstacle_ghost_color,

                            edgecolor="none",

                            alpha=self.obstacle_ghost_alpha,

                            zorder=1
                        )
                    )

                    # Inner footprint
                    #
                    # Matches the visual structure of the
                    # current obstacle, but gray and faint.
                    self.ax.add_patch(

                        plt.Circle(

                            old_center,

                            radius * 0.8,

                            facecolor=self.obstacle_ghost_color,

                            edgecolor="none",

                            alpha=self.obstacle_ghost_alpha,

                            zorder=1
                        )
                    )

        # ====================================================
        # CURRENT OBSTACLES
        # ====================================================

        color_theme = (

            "#e06c75"

            if self.regime_type == "Dynamic"

            else "#d19a66"
        )

        for center, radius in self.env.obstacles:

            self.ax.add_patch(

                plt.Circle(

                    center,

                    radius,

                    color=color_theme,

                    alpha=0.23,

                    zorder=2
                )
            )

            self.ax.add_patch(

                plt.Circle(

                    center,

                    radius * 0.8,

                    color=color_theme,

                    alpha=0.40,

                    zorder=2
                )
            )

        # ====================================================
        # UAV
        # ====================================================

        self.draw_realistic_quadcopter(

            self.env.pos,

            self.env.theta,

            radius=0.35
        )

        # ====================================================
        # TITLE
        # ====================================================

        self.ax.set_title(

            f"Operational Framework: "
            f"{self.regime_type} "
            f"Environment Configuration",

            color="white",

            fontsize=11
        )

        # ====================================================
        # LIDAR
        # ====================================================

        try:

            lidar_ranges = (
                self.env._lidar_scan()
                * self.env.world_size
            )

            angles = np.linspace(

                0,

                2 * np.pi,

                len(
                    lidar_ranges
                ),

                endpoint=False
            )

            for angle, dist in zip(
                angles,
                lidar_ranges
            ):

                end_x = (
                    self.env.pos[0]
                    + dist * np.cos(angle)
                )

                end_y = (
                    self.env.pos[1]
                    + dist * np.sin(angle)
                )

                self.ax.plot(

                    [
                        self.env.pos[0],
                        end_x
                    ],

                    [
                        self.env.pos[1],
                        end_y
                    ],

                    color="#56b6c2",

                    alpha=0.22,

                    linewidth=0.45,

                    zorder=3
                )

        except Exception as e:

            print(
                "LiDAR draw error:",
                e
            )

        # ====================================================
        # LEGEND
        # ====================================================

        legend_elements = [

            Line2D(

                [0],
                [0],

                marker="^",

                color="w",

                markerfacecolor="#61afef",

                markersize=9,

                linestyle="None",

                label="UAV Start"
            ),

            Line2D(

                [0],
                [0],

                marker="P",

                color="w",

                markerfacecolor="#98c379",

                markersize=9,

                linestyle="None",

                label="Goal"
            ),

            Line2D(

                [0],
                [0],

                color="#61afef",

                linestyle=":",

                linewidth=2,

                label="UAV Trajectory"
            ),

            Line2D(

                [0],
                [0],

                marker="o",

                color="#e06c75",

                markerfacecolor="#e06c75",

                markersize=9,

                linestyle="None",

                label="Current Dynamic Obstacle"
            ),

            Line2D(

                [0],
                [0],

                marker="o",

                color=self.obstacle_ghost_color,

                markerfacecolor=self.obstacle_ghost_color,

                alpha=0.35,

                markersize=9,

                linestyle="None",

                label="Previous Obstacle Positions"
            ),
        ]

        # If you want the legend visible, uncomment:
        #
        # self.ax.legend(
        #     handles=legend_elements,
        #     loc="upper left",
        #     fontsize=8
        # )

        self.canvas.draw()


    # ========================================================
    # SIMULATION FRAME
    # ========================================================

    def run_single_sim_frame(self):

        if not self.sim_loop_active:
            return

        # ----------------------------------------------------
        # Agent action
        # ----------------------------------------------------

        action = self.agent.select_action(
            self.state
        )

        next_obs, _reward, terminated, truncated, info = (
            self.env.step(action)
        )

        done = (
            terminated
            or truncated
        )

        # ====================================================
        # NEW:
        # Record obstacle centers AFTER environment update.
        #
        # This means every simulation frame becomes one gray
        # historical footprint during future frames.
        # ====================================================

        if self.regime_type == "Dynamic":

            if len(self.obstacle_position_history) != len(self.env.obstacles):

                self.obstacle_position_history = [
                    [center.copy()]
                    for center, radius in self.env.obstacles
                ]

            elif self.env.steps % self.ghost_stride == 0:

                for i, (center, radius) in enumerate(self.env.obstacles):

                    self.obstacle_position_history[i].append(
                        center.copy()
                    )
        # ----------------------------------------------------
        # Frame stack
        # ----------------------------------------------------

        self.frame_stack.append(
            next_obs
        )

        self.state = np.concatenate(
            list(self.frame_stack),
            axis=0
        )

        # ----------------------------------------------------
        # Telemetry
        # ----------------------------------------------------

        current_speed = info.get(

            "speed",

            np.linalg.norm(
                self.env.vel
            )

            if hasattr(
                self.env,
                "vel"
            )

            else 0.0
        )

        current_dist = info.get(

            "distance_to_goal",

            self.env._distance_to_goal()
        )

        if "min_lidar_distance" in info:

            min_lidar = info[
                "min_lidar_distance"
            ]

        else:

            min_lidar = (

                float(
                    np.min(
                        next_obs[
                            :self.env.n_lidar
                        ]
                    )
                )

                * self.env.world_size
            )

        self.telemetry_labels[
            "steps"
        ].setText(

            f"Simulation Frame Step: "
            f"{self.env.steps}"
        )

        self.telemetry_labels[
            "speed"
        ].setText(

            f"Current Linear Velocity: "
            f"{current_speed:.2f} m/s"
        )

        self.telemetry_labels[
            "dmin"
        ].setText(

            f"Minimum LiDAR Distance: "
            f"{min_lidar:.3f} m"
        )

        self.telemetry_labels[
            "dist"
        ].setText(

            f"Distance to Target: "
            f"{current_dist:.2f} m"
        )

        # ----------------------------------------------------
        # Terminal conditions
        # ----------------------------------------------------

        if done:

            self.sim_loop_active = False

            self.play_btn.setText(
                "Simulation Terminal Hit"
            )

            self.play_btn.setStyleSheet(
                "background-color: #e06c75; "
                "color: white;"
            )

            if info.get(
                "reached_goal",
                current_dist <= self.env.goal_radius
            ):

                self.telemetry_labels[
                    "status"
                ].setText(

                    "Mission Status: "
                    "SUCCESSFUL TOUCHDOWN"
                )

                self.telemetry_labels[
                    "status"
                ].setStyleSheet(

                    "color: #98c379; "
                    "font-weight: bold;"
                )

            elif info.get(
                "collision",
                False
            ):

                self.telemetry_labels[
                    "status"
                ].setText(

                    "Mission Status: "
                    "COLLISION DETECTED"
                )

                self.telemetry_labels[
                    "status"
                ].setStyleSheet(

                    "color: #e06c75; "
                    "font-weight: bold;"
                )

            else:

                self.telemetry_labels[
                    "status"
                ].setText(

                    "Mission Status: "
                    "TIMEOUT"
                )

                self.telemetry_labels[
                    "status"
                ].setStyleSheet(

                    "color: #e5c07b; "
                    "font-weight: bold;"
                )

        # ----------------------------------------------------
        # Render updated frame
        # ----------------------------------------------------

        self.render_map_frame()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = (
        Path(__file__).resolve().parent
    )

    dynamic_model = (

        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / "dqn_dynamic_standard_obs_8_ep_6000.pth"
    )

    static_model = (

        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / "dqn_static_standard_obs_6_ep_3000.pth"
    )

    app = QApplication(
        sys.argv
    )

    gcs_window = UnifiedUAVGCS(

        dynamic_weights_path=dynamic_model,

        static_weights_path=static_model
    )

    gcs_window.show()

    sys.exit(
        app.exec_()
    )