#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import magnum as mn
from typing import List
import cv2

import habitat
from habitat.articulated_agents.humanoids import KinematicHumanoid
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.config.default_structured_configs import (
    HumanoidJointActionConfig,
    ThirdRGBSensorConfig,
)
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils.common import quat_from_angle_axis
from omegaconf import DictConfig, OmegaConf

# Movement parameters - ADJUSTED FOR SMOOTH MOTION
# These are now speeds (per second), not per-frame deltas
LINEAR_SPEED = 1.0  # meters per second (realistic human walking speed)
ROTATION_SPEED = 1.0  # radians per second (~57 degrees/sec)
CTRL_FREQ = 120.0  # Default simulator control frequency (Hz)


def flat_to_matrix(flat_transform: List[float]) -> mn.Matrix4:
    """Convert flat 16-element list to 4x4 Matrix."""
    return mn.Matrix4(np.array(flat_transform).reshape(4, 4).T)


class ManualHumanoidController(HumanoidRearrangeController):
    """Extended controller with manual movement capabilities."""

    def __init__(self, walk_pose_path: str):
        super().__init__(walk_pose_path)
        # Will be set via set_framerate_for_linspeed after initialization

    def apply_movement(self, forward_delta: float, rot_delta: float, humanoid):
        """
        Apply movement using the CORRECT method with PROPER SPEED CALIBRATION.

        Note: forward_delta and rot_delta should be SPEEDS (m/s and rad/s),
        not per-frame deltas. The set_framerate_for_linspeed() call handles
        the conversion to proper per-step motion.
        """
        if abs(forward_delta) < 0.001 and abs(rot_delta) < 0.001:
            # Standing still - use stop pose
            self.calculate_stop_pose()
        else:
            # Moving - use translate_and_rotate_with_gait
            # Convert speeds to per-step deltas
            meters_per_step = forward_delta / CTRL_FREQ
            radians_per_step = rot_delta / CTRL_FREQ

            self.translate_and_rotate_with_gait(meters_per_step, radians_per_step)

        # Get the computed pose
        new_pose = self.get_pose()

        # Apply to humanoid
        humanoid.set_joint_transform(
            new_pose[:-32],
            flat_to_matrix(new_pose[-32:-16]),
            flat_to_matrix(new_pose[-16:]),
        )


def prepare_config(args):
    """Prepare configuration from reference code pattern."""
    config_path = "benchmark/rearrange/play.yaml"

    # Load config
    config = habitat.get_config(config_path)

    # Make config writable
    with habitat.config.read_write(config):
        # Set seed
        config.habitat.seed = 42

        # Configure dataset
        config.habitat.dataset.split = "train"
        config.habitat.dataset.data_path = "data/datasets/hssd/rearrange/train/social_rearrange.json.gz"

        # Ensure we have the required sensors
        if "third_rgb_sensor" not in config.habitat.simulator.agents.main_agent.sim_sensors:
            config.habitat.simulator.agents.main_agent.sim_sensors.third_rgb_sensor = ThirdRGBSensorConfig()

        # Ensure we have humanoid joint action
        if "humanoid_joint_action" not in config.habitat.task.actions:
            config.habitat.task.actions.humanoid_joint_action = HumanoidJointActionConfig()

        # Make sure ctrl_freq is set
        if not hasattr(config.habitat.simulator, 'ctrl_freq'):
            config.habitat.simulator.ctrl_freq = CTRL_FREQ

    return config


def get_nearest_object(env, humanoid_pos, max_distance=3.0):
    """Find the nearest graspable object to the humanoid."""
    sim = env.sim
    rom = sim.get_rigid_object_manager()

    nearest_obj = None
    nearest_dist = max_distance

    for obj_id in rom.get_object_handles():
        obj = rom.get_object_by_handle(obj_id)
        if obj is None:
            continue

        obj_pos = obj.translation
        dist = np.linalg.norm(np.array(obj_pos) - np.array(humanoid_pos))

        if dist < nearest_dist:
            nearest_dist = dist
            nearest_obj = obj

    return nearest_obj, nearest_dist


def step_env(env, arm_action_name, args_dict):
    """Step the environment with proper action handling."""
    # Use no-op action for the arm
    action_dict = {
        arm_action_name: args_dict,
    }
    return env.step(action_dict)


def main():
    # Prepare configuration
    config = prepare_config(None)

    # Create environment
    print("Creating environment...")
    with habitat.Env(config=config) as env:
        print("Environment created successfully!")

        # Get agent and humanoid
        agent_name = config.habitat.simulator.agents_order[0]
        articulated_agent = env.sim.agents_mgr[0].articulated_agent
        kin_humanoid = articulated_agent

        # Verify it's a humanoid
        if not isinstance(kin_humanoid, KinematicHumanoid):
            print(f"Error: Agent is {type(kin_humanoid)}, not KinematicHumanoid")
            return

        print(f"Humanoid type: {type(kin_humanoid)}")

        # Get the walk pose path from config
        walk_pose_path = kin_humanoid._get_humanoid_params().cameras
        # Use the standard walk pose path
        walk_pose_path = "data/humanoids/humanoid_data/walking_motion_processed.pkl"

        # Create humanoid controller
        print("Creating humanoid controller...")
        humanoid_controller = ManualHumanoidController(walk_pose_path)

        # ===== KEY FIX FOR SMOOTH MOTION =====
        # Calibrate motion playback speed to simulator physics rate
        print(f"Calibrating motion for smooth playback...")
        print(f"  Linear speed: {LINEAR_SPEED} m/s")
        print(f"  Angular speed: {ROTATION_SPEED} rad/s ({np.degrees(ROTATION_SPEED):.1f} deg/s)")
        print(f"  Control frequency: {CTRL_FREQ} Hz")

        humanoid_controller.set_framerate_for_linspeed(
            lin_speed=LINEAR_SPEED,
            ang_speed=ROTATION_SPEED,
            ctrl_freq=CTRL_FREQ
        )
        print("Motion calibration complete!")
        # ====================================

        # Reset controller
        humanoid_controller.reset(kin_humanoid.base_transformation)

        # Get action name for arm
        arm_action_name = None
        for action_name in env.action_space.keys():
            if "arm_action" in action_name.lower():
                arm_action_name = action_name
                break

        if arm_action_name is None:
            arm_action_name = list(env.action_space.keys())[0]

        print(f"Using arm action: {arm_action_name}")
        print(f"Action space: {env.action_space.keys()}")

        # Reset environment
        obs = env.reset()

        # Initialize OpenCV window
        cv2.namedWindow("Humanoid Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Humanoid Control", 1280, 720)

        # Control state
        held_object = None

        print("\n" + "="*60)
        print("HUMANOID CONTROL - SMOOTH MOTION ENABLED")
        print("="*60)
        print("Controls:")
        print("  I/K - Move Forward/Backward")
        print("  J/L - Strafe Left/Right")
        print("  U/O - Rotate Left/Right")
        print("  SPACE - Pick up nearest object")
        print("  T - Throw held object")
        print("  G - Release held object (gentle)")
        print("  Q - Quit")
        print("="*60)
        print("\nSmoothing enabled via set_framerate_for_linspeed()")
        print(f"Motion calibrated for {LINEAR_SPEED} m/s walk speed\n")

        # Main loop
        frame_count = 0
        while True:
            # Get keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Movement deltas (these are now SPEEDS in m/s and rad/s)
            forward_speed = 0.0
            rot_speed = 0.0

            # Process movement keys
            if key == ord('i'):  # Forward
                forward_speed = LINEAR_SPEED
            elif key == ord('k'):  # Backward
                forward_speed = -LINEAR_SPEED
            elif key == ord('j'):  # Strafe left (rotate while moving)
                forward_speed = LINEAR_SPEED * 0.5
                rot_speed = ROTATION_SPEED * 0.5
            elif key == ord('l'):  # Strafe right (rotate while moving)
                forward_speed = LINEAR_SPEED * 0.5
                rot_speed = -ROTATION_SPEED * 0.5
            elif key == ord('u'):  # Rotate left
                rot_speed = ROTATION_SPEED
            elif key == ord('o'):  # Rotate right
                rot_speed = -ROTATION_SPEED
            elif key == ord('q'):  # Quit
                break
            elif key == ord(' '):  # Pick up object
                humanoid_pos = kin_humanoid.base_pos
                nearest_obj, dist = get_nearest_object(env, humanoid_pos, max_distance=3.0)
                if nearest_obj is not None:
                    print(f"Picking up object at distance {dist:.2f}m")
                    # Simple grasp - just parent object to humanoid
                    held_object = nearest_obj
                    # You could enhance this with proper reaching motion
                else:
                    print("No object nearby to pick up")
            elif key == ord('t'):  # Throw object
                if held_object is not None:
                    print("Throwing object!")
                    # Apply impulse based on humanoid orientation
                    base_T = kin_humanoid.base_transformation
                    forward_dir = base_T.transform_vector(mn.Vector3(1, 0, 0))
                    throw_vel = forward_dir * 5.0 + mn.Vector3(0, 3.0, 0)  # Forward and up
                    held_object.linear_velocity = throw_vel
                    held_object.angular_velocity = mn.Vector3(0, 0, 0)
                    held_object = None
                else:
                    print("No object held to throw")
            elif key == ord('g'):  # Release object gently
                if held_object is not None:
                    print("Releasing object")
                    held_object.linear_velocity = mn.Vector3(0, 0, 0)
                    held_object.angular_velocity = mn.Vector3(0, 0, 0)
                    held_object = None
                else:
                    print("No object held to release")

            # Step environment first (required)
            arm_action_args = np.zeros(7)  # No-op for arm
            obs = step_env(env, arm_action_name, arm_action_args)

            # Apply humanoid movement with SMOOTH MOTION
            humanoid_controller.apply_movement(forward_speed, rot_speed, kin_humanoid)

            # Update held object position if any
            if held_object is not None:
                # Position object in front of humanoid
                base_T = kin_humanoid.base_transformation
                offset = base_T.transform_vector(mn.Vector3(0.5, 0.5, 0))
                held_object.translation = base_T.translation + offset
                held_object.linear_velocity = mn.Vector3(0, 0, 0)
                held_object.angular_velocity = mn.Vector3(0, 0, 0)

            # Step physics
            env.sim.step_physics(1.0 / 60.0)

            # Render
            info = env.get_metrics()
            draw = observations_to_image(obs, info)
            draw = overlay_frame(draw, info)

            # Add smoothing status overlay
            if frame_count % 60 == 0:  # Update every 60 frames
                status_text = f"SMOOTH MOTION @ {LINEAR_SPEED}m/s | Frame: {frame_count}"
                cv2.putText(draw, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Humanoid Control", draw)

            frame_count += 1

        cv2.destroyAllWindows()
        print("\nExiting...")


if __name__ == "__main__":
    main()
