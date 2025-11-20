#!/usr/bin/env python3
"""
humanoid_ultimate_control_smooth.py

PROPER manual control with SMOOTH MOTION using translate_and_rotate_with_gait
+ set_framerate_for_linspeed calibration
+ MULTI-CAMERA SUPPORT with 'C' key switching
"""

import argparse
import os
import sys
import time
import pickle as pkl
from typing import Optional, Dict, Any, List

import cv2
import magnum as mn
import numpy as np
from omegaconf import DictConfig

import habitat
import habitat.tasks.rearrange.rearrange_task
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig
from habitat.core.logging import logger
from habitat.tasks.rearrange.utils import euler_to_quat
from habitat.utils.visualizations.utils import observations_to_image, overlay_frame
import habitat.articulated_agents.humanoids.kinematic_humanoid as kinematic_humanoid
import habitat_sim


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
WINDOW_NAME = "Humanoid Manual Control - SMOOTH + Multi-Cam"

HUMANOID_NAME = "female_2"
HUMANOID_URDF = f"data/humanoids/humanoid_data/{HUMANOID_NAME}/{HUMANOID_NAME}.urdf"
MOTION_DATA_PATH = f"data/humanoids/humanoid_data/{HUMANOID_NAME}/{HUMANOID_NAME}_motion_data_smplx.pkl"

if not os.path.exists(HUMANOID_URDF):
    HUMANOID_NAME = "female_0"
    HUMANOID_URDF = f"data/hab3_bench_assets/humanoids/{HUMANOID_NAME}/{HUMANOID_NAME}.urdf"
    MOTION_DATA_PATH = f"data/hab3_bench_assets/humanoids/{HUMANOID_NAME}/{HUMANOID_NAME}_motion_data_smplx.pkl"

# Movement parameters - NOW AS SPEEDS (m/s and rad/s) for smooth motion
LINEAR_SPEED = 1.2   # meters per second (realistic human walking speed)
ANGULAR_SPEED = 3.0  # radians per second (~172 degrees/sec)
CTRL_FREQ = 60.0     # Physics step rate (from step_physics(1.0/60.0))
PICK_RADIUS = 3.0
THROW_FORCE = 5.0


# ============================================================================
# FREE CAMERA SYSTEM (from interactive_play.py)
# ============================================================================

class FreeCamHelper:
    """
    Free camera movement with NO limits - fly through the scene freely.
    Based on examples/interactive_play.py FreeCamHelper.
    """
    def __init__(self):
        self._is_free_cam_mode = False
        self._free_rpy = np.zeros(3)  # Roll, Pitch, Yaw (Euler angles)
        self._free_xyz = np.zeros(3)  # X, Y, Z position

    @property
    def is_free_cam_mode(self):
        return self._is_free_cam_mode

    def toggle_mode(self):
        """Toggle free camera mode on/off."""
        self._is_free_cam_mode = not self._is_free_cam_mode
        if self._is_free_cam_mode:
            print("🎥 FREE CAMERA MODE - Fly anywhere!")
            print("   W/S: Forward/Back | A/D: Left/Right | Q/E: Up/Down")
            print("   Arrow Keys: Rotate | B: Reset camera | Z: Exit free cam")
        else:
            print("👤 FOLLOW CAMERA MODE - Following humanoid")
            # Reset camera when exiting free mode
            self._free_rpy = np.zeros(3)
            self._free_xyz = np.zeros(3)

    def update_camera(self, key: Optional[str], sim):
        """Update free camera position and rotation."""
        if not self._is_free_cam_mode:
            return None

        # Rotation controls (using arrow keys-like pattern)
        offset_rpy = np.zeros(3)
        if key == "i":  # Pitch down (look down)
            offset_rpy[2] += 1
        elif key == "k":  # Pitch up (look up)
            offset_rpy[2] -= 1
        elif key == "j":  # Yaw left (turn left)
            offset_rpy[0] += 1
        elif key == "l":  # Yaw right (turn right)
            offset_rpy[0] -= 1
        elif key == "u":  # Roll left
            offset_rpy[1] += 1
        elif key == "o":  # Roll right
            offset_rpy[1] -= 1

        # Translation controls
        offset_xyz = np.zeros(3)
        if key == "q":  # Up
            offset_xyz[1] += 1
        elif key == "e":  # Down
            offset_xyz[1] -= 1
        elif key == "w":  # Forward
            offset_xyz[2] += 1
        elif key == "s":  # Backward
            offset_xyz[2] -= 1
        elif key == "a":  # Left
            offset_xyz[0] += 1
        elif key == "d":  # Right
            offset_xyz[0] -= 1
        elif key == "b":  # Reset camera
            self._free_rpy = np.zeros(3)
            self._free_xyz = np.zeros(3)
            print("🔄 Camera reset to origin")

        # Apply offsets
        offset_rpy *= 0.1
        offset_xyz *= 0.1
        self._free_rpy += offset_rpy
        self._free_xyz += offset_xyz

        # Create quaternion from Euler angles
        quat = euler_to_quat(self._free_rpy)

        # Use the CORRECT Habitat API to get observations from arbitrary position/rotation
        obs = sim.get_observations_at(
            position=mn.Vector3(*self._free_xyz),
            rotation=quat,
            keep_agent_at_new_pose=False  # Don't move the agent
        )

        return obs  # Return observation from free camera position


# ============================================================================
# CAMERA SYSTEM
# ============================================================================

class CameraManager:
    """Manages multiple camera viewpoints and switching between them."""

    def __init__(self, sim):
        self.sim = sim
        self.camera_configs = []
        self.current_camera_idx = 0
        self.camera_names = []

    def add_camera(self, name: str, position: mn.Vector3, look_at: mn.Vector3):
        """Add a camera with position and look-at target."""
        self.camera_names.append(name)
        self.camera_configs.append({
            'name': name,
            'position': position,
            'look_at': look_at
        })

    def setup_corner_cameras(self, scene_bounds: tuple = None):
        """Setup cameras at scene corners."""
        # Handle different bounds formats
        if scene_bounds is None:
            # Default scene bounds
            bounds_min = mn.Vector3(-10, 0, -10)
            bounds_max = mn.Vector3(10, 5, 10)
        elif isinstance(scene_bounds, tuple) and len(scene_bounds) == 2:
            # Pathfinder returns (min_vec, max_vec) tuple
            bounds_min = scene_bounds[0]
            bounds_max = scene_bounds[1]
        else:
            # Fallback
            bounds_min = mn.Vector3(-10, 0, -10)
            bounds_max = mn.Vector3(10, 5, 10)

        center = (bounds_min + bounds_max) * 0.5
        height = bounds_max.y * 0.8  # 80% of max height

        # Calculate corner positions (elevated for better view)
        corners = [
            (mn.Vector3(bounds_min.x, height, bounds_min.z), "Corner 1 (SW)"),
            (mn.Vector3(bounds_max.x, height, bounds_min.z), "Corner 2 (SE)"),
            (mn.Vector3(bounds_max.x, height, bounds_max.z), "Corner 3 (NE)"),
            (mn.Vector3(bounds_min.x, height, bounds_max.z), "Corner 4 (NW)"),
        ]

        # Add overhead camera
        overhead_pos = mn.Vector3(center.x, bounds_max.y * 1.5, center.z)
        self.add_camera("Overhead", overhead_pos, center)

        # Add corner cameras looking at center
        for pos, name in corners:
            self.add_camera(name, pos, center)

        print(f"✓ Setup {len(self.camera_configs)} fixed cameras")

    def add_follow_camera(self, humanoid_pos: mn.Vector3):
        """Add a third-person follow camera (updated each frame)."""
        # This is handled by the existing third_rgb sensor
        pass

    def cycle_camera(self):
        """Switch to next camera."""
        self.current_camera_idx = (self.current_camera_idx + 1) % len(self.camera_configs)
        cam = self.camera_configs[self.current_camera_idx]
        print(f"📷 Camera: {cam['name']}")

    def get_current_camera_name(self) -> str:
        """Get name of current camera."""
        if self.current_camera_idx == 0:
            return "Follow Cam"
        return self.camera_configs[self.current_camera_idx]['name']

    def get_camera_observation(self, camera_idx: int, humanoid_pos: mn.Vector3):
        """
        Get observation from a specific camera using get_observations_at().
        This is the CORRECT way from Habitat documentation.
        """
        if camera_idx == 0:
            # Follow camera - return None to use default observation
            return None

        # Get the camera configuration
        cam = self.camera_configs[camera_idx]

        # Calculate look-at orientation
        look_at_matrix = mn.Matrix4.look_at(
            cam['position'],      # camera position
            cam['look_at'],       # look at target (scene center)
            mn.Vector3(0, 1, 0)   # up vector
        )

        # Extract rotation as quaternion
        rotation_quat = mn.Quaternion.from_matrix(look_at_matrix.rotation())

        # Use the CORRECT Habitat API to get observations from arbitrary position/rotation
        obs = self.sim.get_observations_at(
            position=cam['position'],
            rotation=rotation_quat,
            keep_agent_at_new_pose=False  # Don't actually move the agent
        )

        return obs


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def step_env(env, action_name: str, action_args: Dict[str, Any]):
    return env.step({"action": action_name, "action_args": action_args})


def key_to_char(key_code: int) -> Optional[str]:
    if key_code < 0:
        return None
    if key_code == 27:
        return "escape"
    if key_code == 32:
        return "space"
    key_code &= 0xFF
    if 32 <= key_code <= 126:
        return chr(key_code).lower()
    return None


def flat_to_matrix(flat: np.ndarray) -> mn.Matrix4:
    rows = [mn.Vector4(flat[i * 4 : (i + 1) * 4]) for i in range(4)]
    return mn.Matrix4(*rows)


# ============================================================================
# RENDERING
# ============================================================================

class CvRenderer:
    def __init__(self, window_name: str, width: int, height: int, enabled: bool):
        self.enabled = enabled
        self.window_name = window_name
        if enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, width, height)

    def show(self, rgb_frame: np.ndarray):
        if not self.enabled:
            return
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, bgr)

    def read_key(self) -> Optional[str]:
        if not self.enabled:
            return None
        key = cv2.waitKeyEx(1)
        return key_to_char(key)

    def close(self):
        if self.enabled:
            cv2.destroyWindow(self.window_name)


# ============================================================================
# HUMANOID CONTROLLER - SMOOTH MOTION IMPLEMENTATION
# ============================================================================

class ManualHumanoidController(HumanoidRearrangeController):
    """Properly uses translate_and_rotate_with_gait with speed calibration for smooth motion."""

    def __init__(self, walk_pose_path: str):
        super().__init__(walk_pose_path)

        with open(walk_pose_path, "rb") as f:
            data = pkl.load(f)

        self.standing_pose = data.get("stop_pose", None)
        self.held_object_id = None
        self.held_object = None

    def apply_movement(self, forward_speed: float, rot_speed: float, humanoid):
        """
        Apply movement using the CORRECT method with proper speed conversion.

        Args:
            forward_speed: Linear speed in m/s
            rot_speed: Angular speed in rad/s
            humanoid: The kinematic humanoid to apply motion to
        """
        if abs(forward_speed) < 0.001 and abs(rot_speed) < 0.001:
            # Standing still - use stop pose
            self.calculate_stop_pose()
        else:
            # Moving - convert speeds to per-step deltas
            # forward_speed and rot_speed are SPEEDS (m/s, rad/s)
            # We need per-step deltas for translate_and_rotate_with_gait
            meters_per_step = forward_speed / CTRL_FREQ
            radians_per_step = rot_speed / CTRL_FREQ

            self.translate_and_rotate_with_gait(meters_per_step, radians_per_step)

        # Get the computed pose
        new_pose = self.get_pose()

        # Apply to humanoid
        humanoid.set_joint_transform(
            new_pose[:-32],
            flat_to_matrix(new_pose[-32:-16]),
            flat_to_matrix(new_pose[-16:]),
        )

    def pick_object(self, sim, humanoid):
        """Pick nearest object."""
        humanoid_pos = humanoid.base_pos
        rom = sim.get_rigid_object_manager()

        nearest_obj = None
        nearest_dist = PICK_RADIUS

        for obj_id in sim.scene_obj_ids:
            obj = rom.get_object_by_id(obj_id)
            obj_pos = obj.translation
            dist = (mn.Vector3(obj_pos[0], humanoid_pos[1], obj_pos[2]) - humanoid_pos).length()

            if dist < nearest_dist:
                nearest_dist = dist
                nearest_obj = (obj_id, obj_pos, obj)

        if nearest_obj is None:
            print(f"⚠️ No objects within {PICK_RADIUS}m")
            return False

        obj_id, obj_pos, obj = nearest_obj

        print(f"📦 Picking at {nearest_dist:.2f}m")

        # Reach
        reach_target = obj_pos + mn.Vector3(0, 0.1, 0)
        self.calculate_reach_pose(reach_target, index_hand=1)
        new_pose = self.get_pose()

        humanoid.set_joint_transform(
            new_pose[:-32],
            flat_to_matrix(new_pose[-32:-16]),
            flat_to_matrix(new_pose[-16:]),
        )

        # Grasp
        grasp_mgr = sim.agents_mgr[0].grasp_mgrs[0]
        grasp_mgr.snap_to_obj(obj_id)

        self.held_object_id = obj_id
        self.held_object = obj
        print(f"✓ Picked")
        return True

    def throw_object(self, sim, humanoid):
        """Throw object."""
        if self.held_object_id is None:
            print("⚠️ Nothing to throw")
            return

        # Get direction from obj_transform_base
        forward_vec = self.obj_transform_base.transform_vector(mn.Vector3(1, 0, 0))
        throw_vel = forward_vec.normalized() * THROW_FORCE + mn.Vector3(0, 2, 0)

        # Release
        grasp_mgr = sim.agents_mgr[0].grasp_mgrs[0]
        grasp_mgr.desnap()

        if self.held_object is not None:
            self.held_object.linear_velocity = throw_vel
            self.held_object.angular_velocity = mn.Vector3(np.random.randn(3) * 2)

        print(f"🎯 Threw at {throw_vel.length():.1f} m/s")
        self.held_object_id = None
        self.held_object = None

    def release_object(self, sim):
        """Release object."""
        if self.held_object_id is None:
            print("⚠️ Nothing held")
            return

        grasp_mgr = sim.agents_mgr[0].grasp_mgrs[0]
        grasp_mgr.desnap()

        print("✓ Released")
        self.held_object_id = None
        self.held_object = None


# ============================================================================
# CONFIG
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--skip-render-text", action="store_true", default=False)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def prepare_config(args):
    config = habitat.get_config(args.cfg, args.opts)

    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        env_config.max_episode_steps = 0
        sim_config.debug_render = True

        agent_config = get_agent_config(sim_config=sim_config)
        agent_config.sim_sensors.update({
            "third_rgb_sensor": ThirdRGBSensorConfig(
                height=args.play_cam_res,
                width=args.play_cam_res
            )
        })

        if "pddl_success" in task_config.measurements:
            task_config.measurements.pddl_success.must_call_stop = False
        if "force_terminate" in task_config.measurements:
            task_config.measurements.force_terminate.max_accum_force = -1.0
            task_config.measurements.force_terminate.max_instant_force = -1.0

    return config


def configure_humanoid(sim, walk_pose_path: str):
    """Configure humanoid with SMOOTH MOTION calibration."""
    humanoid_cfg = DictConfig({
        "articulated_agent_urdf": HUMANOID_URDF,
        "motion_data_path": walk_pose_path,
        "auto_update_sensor_transform": True,
    })

    kin_humanoid = kinematic_humanoid.KinematicHumanoid(humanoid_cfg, sim)
    kin_humanoid.reconfigure()
    kin_humanoid.update()

    if sim.pathfinder.is_loaded:
        base_pos = sim.pathfinder.get_random_navigable_point()
    else:
        base_pos = mn.Vector3(0.0, 0.0, 0.0)

    kin_humanoid.base_pos = base_pos

    controller = ManualHumanoidController(walk_pose_path)

    # ===== KEY FIX FOR SMOOTH MOTION =====
    # Calibrate motion playback speed to simulator physics rate
    # This synchronizes the walk cycle animation with the physics steps
    controller.set_framerate_for_linspeed(
        lin_speed=LINEAR_SPEED,
        ang_speed=ANGULAR_SPEED,
        ctrl_freq=CTRL_FREQ
    )
    print(f"✓ Motion calibrated: {LINEAR_SPEED}m/s linear, {ANGULAR_SPEED}rad/s angular @ {CTRL_FREQ}Hz")
    # ====================================

    controller.reset(kin_humanoid.base_transformation)

    return kin_humanoid, controller


def expand_action_space(env, agent_idx: int = 0):
    multi_agent = len(env._sim.agents_mgr) > 1
    agent_prefix = f"agent_{agent_idx}_" if multi_agent else ""

    arm_action_name = f"{agent_prefix}arm_action"
    arm_key = "arm_action"
    grip_key = "grip_action"

    base_action_name = None
    for candidate in (
        f"{agent_prefix}base_velocity_non_cylinder",
        f"{agent_prefix}base_velocity",
        f"{agent_prefix}base_velocity_cylinder",
    ):
        if candidate in env.action_space.spaces:
            base_action_name = candidate
            break

    base_key = "base_vel" if base_action_name is not None else None
    return arm_action_name, base_action_name, base_key, {"arm": arm_key, "grip": grip_key}


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not os.path.exists(HUMANOID_URDF):
        print(f"❌ Humanoid not found: {HUMANOID_URDF}")
        sys.exit(1)

    args = parse_args()
    config = prepare_config(args)

    renderer = CvRenderer(
        WINDOW_NAME,
        width=args.render_width,
        height=args.render_height,
        enabled=not args.no_render,
    )

    print("\n" + "="*80)
    print("🎮 HUMANOID CONTROL + FREE CAMERA")
    print("="*80)
    print(f"\n  Motion Speed: {LINEAR_SPEED} m/s linear, {ANGULAR_SPEED} rad/s angular")
    print(f"  Physics Rate: {CTRL_FREQ} Hz")
    print("\n  === HUMANOID CONTROL ===")
    print("  I/K : Walk forward/backward")
    print("  J/L : Strafe left/right")
    print("  U/O : Rotate left/right")
    print("\n  === OBJECT INTERACTION ===")
    print("  SPACE : Pick nearest")
    print("  T     : Throw")
    print("  G     : Release")
    print("\n  === FREE CAMERA MODE ===")
    print("  Z     : Toggle FREE CAMERA (fly anywhere!)")
    print("    W/S : Forward/Backward (in free cam)")
    print("    A/D : Left/Right (in free cam)")
    print("    Q/E : Up/Down (in free cam)")
    print("    I/K/J/L/U/O : Rotate camera (in free cam)")
    print("    B   : Reset camera position")
    print("\n  N : NavMesh | P : Position | M : Reset | ESC : Quit")
    print("\n" + "="*80 + "\n")

    with habitat.Env(config=config) as env:
        obs = env.reset()
        sim = env._sim

        # Initialize humanoid with smooth motion
        try:
            kin_humanoid, humanoid_controller = configure_humanoid(sim, MOTION_DATA_PATH)
            logger.info("✅ Humanoid initialized with smooth motion")
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            return

        if sim.pathfinder.is_loaded:
            logger.info("✅ NavMesh loaded")

        # ===== SETUP CAMERAS =====
        camera_mgr = CameraManager(sim)

        # Get scene bounds from pathfinder if available
        if sim.pathfinder.is_loaded:
            bounds = sim.pathfinder.get_bounds()
            print(f"Scene bounds: {bounds}")
        else:
            bounds = None

        camera_mgr.setup_corner_cameras(bounds)
        # Start with follow camera
        print("📷 Camera: Follow Cam (third_rgb)")
        # =========================

        # ===== SETUP FREE CAMERA =====
        free_cam = FreeCamHelper()
        print("✅ Free camera system initialized (press Z to toggle)")
        # =============================

        # Action space
        arm_action_name, base_action_name, base_key, key_map = expand_action_space(env)
        arm_dim = (
            env.action_space.spaces[arm_action_name].spaces[key_map["arm"]].shape[0]
            if arm_action_name in env.action_space.spaces
            else 7
        )

        total_reward = 0.0
        frame_idx = 0
        target_fps = 60.0
        prev_time = time.time()

        # Main loop
        while True:
            key = renderer.read_key()

            if key == "escape":
                logger.info("👋 Exiting")
                break

            if key == "n":
                sim.navmesh_visualization = not sim.navmesh_visualization

            if key == "m":
                obs = env.reset()
                total_reward = 0.0
                continue

            if key == "p":
                pos = kin_humanoid.base_pos
                cam_pos = free_cam._free_xyz
                print(f"Humanoid: [{pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}]")
                print(f"Camera: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")

            # Toggle free camera mode
            if key == "z":
                free_cam.toggle_mode()

            # Camera switching (only in follow mode)
            if key == "c" and not free_cam.is_free_cam_mode:
                camera_mgr.cycle_camera()

            # ================================================================
            # STEP 1: Step environment (always do this for proper observations)
            # ================================================================
            base_action = [0, 0]
            arm_action = np.zeros(arm_dim, dtype=np.float32)
            magic_grasp = 0.0

            args_dict = {key_map["arm"]: arm_action, key_map["grip"]: magic_grasp}

            # ================================================================
            # CHECK IF IN FREE CAMERA MODE
            # ================================================================
            if free_cam.is_free_cam_mode:
                # FREE CAMERA MODE - use get_observations_at() API
                free_cam_obs = free_cam.update_camera(key, sim)
                if free_cam_obs is not None:
                    obs = free_cam_obs
                # Skip humanoid control and physics in free camera mode

            else:
                # NORMAL MODE - control humanoid

                # Object interaction
                if key == "space":
                    humanoid_controller.pick_object(sim, kin_humanoid)
                if key == "t":
                    humanoid_controller.throw_object(sim, kin_humanoid)
                if key == "g":
                    humanoid_controller.release_object(sim)

                # Step environment
                obs = step_env(env, arm_action_name, args_dict)

                # ================================================================
                # STEP 2: Apply humanoid movement with SMOOTH MOTION
                # ================================================================
                forward_speed = 0.0  # m/s
                rot_speed = 0.0      # rad/s

                # Read keys - THESE CONTROL SPEED (m/s and rad/s)
                if key == "i":
                    forward_speed = LINEAR_SPEED
                if key == "k":
                    forward_speed = -LINEAR_SPEED
                if key == "j":
                    forward_speed = LINEAR_SPEED * 0.5  # Strafe left
                    rot_speed = ANGULAR_SPEED * 0.5
                if key == "l":
                    forward_speed = LINEAR_SPEED * 0.5  # Strafe right
                    rot_speed = -ANGULAR_SPEED * 0.5
                if key == "u":
                    rot_speed = ANGULAR_SPEED
                if key == "o":
                    rot_speed = -ANGULAR_SPEED

                # Apply movement using CORRECT method with calibrated speeds
                humanoid_controller.apply_movement(forward_speed, rot_speed, kin_humanoid)

                # ================================================================
                # STEP 3: Step physics
                # ================================================================
                sim.step_physics(1.0 / 60.0)

                # ================================================================
                # STEP 4: Get camera observation if needed
                # ================================================================
                if camera_mgr.current_camera_idx != 0:
                    # Get observation from fixed camera using get_observations_at()
                    camera_obs = camera_mgr.get_camera_observation(
                        camera_mgr.current_camera_idx,
                        kin_humanoid.base_pos
                    )
                    if camera_obs is not None:
                        obs = camera_obs

            # Get metrics
            info = env.get_metrics()
            reward_key = [k for k in info if "reward" in k]
            reward = info[reward_key[0]] if reward_key else 0.0
            total_reward += reward
            info["Total Reward"] = total_reward

            # Humanoid info
            hp = kin_humanoid.base_pos
            info["Humanoid"] = f"[{hp[0]:.2f}, {hp[1]:.2f}, {hp[2]:.2f}]"
            info["Holding"] = "Yes" if humanoid_controller.held_object_id else "No"
            info["Frame"] = humanoid_controller.walk_mocap_frame
            info["SMOOTH"] = f"{LINEAR_SPEED}m/s"

            # Camera mode info
            if free_cam.is_free_cam_mode:
                cam_pos = free_cam._free_xyz
                info["Mode"] = "🎥 FREE CAMERA"
                info["Cam Pos"] = f"[{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]"
            else:
                info["Mode"] = "👤 Follow Humanoid"
                info["Camera"] = camera_mgr.get_current_camera_name()

            # ================================================================
            # RENDER - Camera is already positioned by apply_camera_transform
            # ================================================================
            draw = observations_to_image(obs, info)
            if not args.skip_render_text:
                draw = overlay_frame(draw, info)

            renderer.show(draw)

            # Episode over
            if env.episode_over:
                logger.info("Episode over -> reset")
                total_reward = 0.0
                obs = env.reset()

            frame_idx += 1

            # Timing
            curr_time = time.time()
            dt = curr_time - prev_time
            delay = max(1.0 / target_fps - dt, 0.0)
            time.sleep(delay)
            prev_time = curr_time

    renderer.close()
    logger.info("✓ Done")


if __name__ == "__main__":
    main()
