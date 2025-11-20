# Humanoid Joint Control Guide

## Overview

The humanoid has **17 spherical joints** that can be controlled individually to create poses, gestures, and animations. Each joint is controlled by a **quaternion** (4 values: x, y, z, w).

## Quick Start - Keyboard Controls

Run the program and use these keys to control the humanoid's arms:

```bash
python humanoid_ultimate_control_smooth.py
```

| Key | Action |
|-----|--------|
| `1` | Raise left arm to the side (90°) |
| `2` | Raise right arm to the side (90°) |
| `3` | Raise both arms |
| `4` | Wave hand (right hand wave gesture) |
| `5` | Arms forward (reaching pose, 45°) |
| `6` | Bend left elbow (90°) |
| `7` | Bend right elbow (90°) |
| `0` | Reset arms to standing pose |

## Joint Structure

### Complete Joint List (22 joints)

```
Lower Body:
  0:  pelvis         (controlled via base transform)
  1:  left_hip
  2:  right_hip
  3:  spine1
  4:  left_knee
  5:  right_knee
  6:  spine2
  7:  left_ankle
  8:  right_ankle
  9:  spine3
  10: left_foot
  11: right_foot

Upper Body:
  12: neck
  13: left_collar
  14: right_collar
  15: head
  16: left_shoulder   ← Key for arm control
  17: right_shoulder  ← Key for arm control
  18: left_elbow      ← Key for arm control
  19: right_elbow     ← Key for arm control
  20: left_wrist
  21: right_wrist
```

## Programmatic Usage

### Example 1: Simple Gesture

```python
from humanoid_ultimate_control_smooth import JointController

# Initialize (assuming you have humanoid_controller)
joint_ctrl = JointController(humanoid_controller)

# Raise right arm
joint_ctrl.raise_right_arm(90)  # 90 degrees
joint_ctrl.apply_to_humanoid(kin_humanoid)
sim.step_physics(1.0 / 60.0)
```

### Example 2: Custom Joint Rotation

```python
# Set a specific joint to a custom rotation
# Quaternion format: (x, y, z, w)
custom_rotation = (0.0, 0.707, 0.0, 0.707)  # 90° around Y-axis
joint_ctrl.set_joint_rotation("left_shoulder", custom_rotation)
joint_ctrl.apply_to_humanoid(kin_humanoid)
```

### Example 3: Multiple Joint Control

```python
# Create a complex pose
joint_ctrl.raise_both_arms(120)      # Raise arms high
joint_ctrl.bend_left_elbow(90)       # Bend left elbow
joint_ctrl.bend_right_elbow(90)      # Bend right elbow
joint_ctrl.apply_to_humanoid(kin_humanoid)
```

### Example 4: Using Axis-Angle Rotation

```python
import numpy as np
import magnum as mn

# Create a quaternion from axis-angle
axis = mn.Vector3(1, 0, 0)  # X-axis
angle = np.radians(45)       # 45 degrees
quat = joint_ctrl._axis_angle_to_quat(axis, angle)

# Apply to joint
joint_ctrl.set_joint_rotation("left_shoulder", quat)
joint_ctrl.apply_to_humanoid(kin_humanoid)
```

## JointController API Reference

### Pre-built Gestures

| Method | Parameters | Description |
|--------|------------|-------------|
| `raise_left_arm(angle_degrees=90)` | `angle_degrees`: float | Raise left arm sideways |
| `raise_right_arm(angle_degrees=90)` | `angle_degrees`: float | Raise right arm sideways |
| `raise_both_arms(angle_degrees=90)` | `angle_degrees`: float | Raise both arms |
| `wave_hand()` | None | Right hand wave gesture |
| `arms_forward(angle_degrees=45)` | `angle_degrees`: float | Extend arms forward |
| `bend_left_elbow(angle_degrees=90)` | `angle_degrees`: float | Bend left elbow |
| `bend_right_elbow(angle_degrees=90)` | `angle_degrees`: float | Bend right elbow |
| `reset_arms()` | None | Reset arms to standing pose |

### Core Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_current_joints()` | None | `List[float]` | Get all 68 joint values (17 joints × 4) |
| `set_joint_rotation(joint_name, quaternion)` | `joint_name`: str<br>`quaternion`: tuple(4) | `bool` | Set specific joint rotation |
| `apply_to_humanoid(humanoid)` | `humanoid`: KinematicHumanoid | None | Apply modified joints to humanoid |

### Static Utilities

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `_axis_angle_to_quat(axis, angle)` | `axis`: Vector3<br>`angle`: float (radians) | tuple(4) | Convert axis-angle to quaternion |

## Understanding Quaternions

Quaternions represent 3D rotations as 4 values: **(x, y, z, w)**

### Common Rotations

```python
# Identity (no rotation)
identity = (0, 0, 0, 1)

# 90° around X-axis (pitch)
pitch_90 = (0.707, 0, 0, 0.707)

# 90° around Y-axis (yaw)
yaw_90 = (0, 0.707, 0, 0.707)

# 90° around Z-axis (roll)
roll_90 = (0, 0, 0.707, 0.707)
```

### Creating Quaternions from Angles

```python
import numpy as np
import magnum as mn

# Method 1: Axis-angle (recommended)
axis = mn.Vector3(0, 1, 0)  # Y-axis
angle = np.radians(45)       # 45 degrees
quat = mn.Quaternion.rotation(mn.Rad(angle), axis.normalized())
quat_tuple = (quat.vector.x, quat.vector.y, quat.vector.z, quat.scalar)

# Method 2: Use JointController helper
quat_tuple = JointController._axis_angle_to_quat(axis, angle)
```

## Advanced Examples

### Example 1: Create Custom Pose

```python
def surrender_pose(joint_ctrl):
    """Hands up in surrender pose"""
    joint_ctrl.raise_both_arms(150)      # Arms very high
    joint_ctrl.bend_left_elbow(120)      # Elbows bent significantly
    joint_ctrl.bend_right_elbow(120)
    joint_ctrl.apply_to_humanoid(kin_humanoid)
    print("🙌 Surrender pose!")

surrender_pose(joint_ctrl)
```

### Example 2: Animated Gesture

```python
import time

def wave_animation(joint_ctrl, kin_humanoid, sim):
    """Animated waving motion"""
    for angle in [90, 100, 90, 100, 90]:
        joint_ctrl.raise_right_arm(angle)
        joint_ctrl.bend_right_elbow(90)
        joint_ctrl.apply_to_humanoid(kin_humanoid)
        sim.step_physics(1.0 / 60.0)
        time.sleep(0.3)

wave_animation(joint_ctrl, kin_humanoid, sim)
```

### Example 3: T-Pose

```python
def t_pose(joint_ctrl):
    """Classic T-pose for 3D models"""
    joint_ctrl.raise_both_arms(90)  # Arms straight out to sides
    # Elbows and wrists straight (default)
    joint_ctrl.apply_to_humanoid(kin_humanoid)
    print("🤸 T-pose activated!")

t_pose(joint_ctrl)
```

### Example 4: Direct Joint Access

```python
# Get all current joints
joints = joint_ctrl.get_current_joints()  # List of 68 floats

# Modify specific joint manually
# left_shoulder is joint index 16, so starts at position 16*4 = 64
joints[64:68] = [0, 0.707, 0, 0.707]  # 90° around Y-axis

# Apply manually
joint_ctrl.modified_joints = joints
joint_ctrl.apply_to_humanoid(kin_humanoid)
```

## Joint Data Format

Each joint is stored as **4 consecutive floats** (quaternion):

```
Joint Array (68 values total for 17 joints):
[
  # Joint 0 (pelvis) - indices 0-3
  quat0_x, quat0_y, quat0_z, quat0_w,

  # Joint 1 (left_hip) - indices 4-7
  quat1_x, quat1_y, quat1_z, quat1_w,

  # ... continues for all 17 joints ...

  # Joint 16 (left_shoulder) - indices 64-67
  quat16_x, quat16_y, quat16_z, quat16_w,
]
```

## Low-Level Control

For advanced users who want direct control:

```python
# Access the humanoid directly
current_joints, current_transform = kin_humanoid.get_joint_transform()

# Modify joints (68-element list)
modified_joints = list(current_joints)
modified_joints[64:68] = [0, 0.707, 0, 0.707]  # left_shoulder

# Apply
kin_humanoid.set_joint_transform(
    modified_joints,
    offset_transform,  # mn.Matrix4
    base_transform     # mn.Matrix4
)
```

## Common Pitfalls

### 1. Forgot to Call `apply_to_humanoid()`

```python
# ❌ WRONG - joint changes won't be visible
joint_ctrl.raise_left_arm(90)

# ✅ CORRECT - apply changes to humanoid
joint_ctrl.raise_left_arm(90)
joint_ctrl.apply_to_humanoid(kin_humanoid)
```

### 2. Invalid Quaternion Values

```python
# ❌ WRONG - not a unit quaternion
bad_quat = (1, 1, 1, 1)  # Length != 1

# ✅ CORRECT - use normalized quaternion or axis-angle helper
good_quat = JointController._axis_angle_to_quat(
    mn.Vector3(0, 1, 0),
    np.radians(45)
)
```

### 3. Joint Name Typo

```python
# ❌ WRONG - joint name doesn't exist
joint_ctrl.set_joint_rotation("left_arm", quat)  # ❌ No "left_arm"

# ✅ CORRECT - use exact joint name
joint_ctrl.set_joint_rotation("left_shoulder", quat)
```

## Integration with Motion Controller

The joint control works alongside the motion controller:

```python
# Walk forward while waving
joint_ctrl.wave_hand()
joint_ctrl.apply_to_humanoid(kin_humanoid)

# Now walk (motion controller)
humanoid_controller.apply_movement(
    forward_speed=1.2,  # m/s
    rot_speed=0.0,      # rad/s
    humanoid=kin_humanoid
)
```

**Note:** Walking animations may override arm poses. For custom poses during walking, you may need to blend or modify the walking motion data.

## Troubleshooting

### Arms snap back after moving

**Problem:** Joint poses are overridden by walking motion.

**Solution:** Either:
1. Stop walking before applying pose
2. Modify the walking motion data to preserve arm positions
3. Apply joint control every frame after motion update

### Joints in weird positions

**Problem:** Quaternion values are incorrect.

**Solution:** Use the helper methods or `_axis_angle_to_quat()` instead of manually creating quaternions.

### Can't control hands/fingers

**Problem:** Basic humanoid model has fixed wrists.

**Solution:** The SMPL-X model supports finger joints (joints 40-69 in the full model), but they may not be active in all configurations. Check your URDF file.

## Full Working Example

```python
#!/usr/bin/env python3
"""Example: Humanoid poses and gestures"""

import habitat
import numpy as np
from humanoid_ultimate_control_smooth import (
    configure_humanoid,
    JointController,
    MOTION_DATA_PATH
)

def main():
    config = habitat.get_config("benchmark/rearrange/play/play.yaml")

    with habitat.Env(config=config) as env:
        env.reset()
        sim = env._sim

        # Initialize humanoid
        kin_humanoid, humanoid_controller = configure_humanoid(
            sim, MOTION_DATA_PATH
        )

        # Initialize joint controller
        joint_ctrl = JointController(humanoid_controller)

        # Perform gestures
        print("1. Raising right arm...")
        joint_ctrl.raise_right_arm(90)
        joint_ctrl.apply_to_humanoid(kin_humanoid)
        sim.step_physics(1.0)

        print("2. Waving...")
        joint_ctrl.wave_hand()
        joint_ctrl.apply_to_humanoid(kin_humanoid)
        sim.step_physics(1.0)

        print("3. T-pose...")
        joint_ctrl.raise_both_arms(90)
        joint_ctrl.apply_to_humanoid(kin_humanoid)
        sim.step_physics(1.0)

        print("4. Reset...")
        joint_ctrl.reset_arms()
        joint_ctrl.apply_to_humanoid(kin_humanoid)

        print("Done!")

if __name__ == "__main__":
    main()
```

## Resources

- **Habitat-Lab Docs:** https://aihabitat.org/docs/habitat-lab/
- **Quaternion Math:** https://eater.net/quaternions
- **SMPL-X Model:** https://smpl-x.is.tue.mpg.de/

## Summary

The `JointController` provides an easy-to-use interface for controlling the humanoid's joints:

1. **Simple gestures** with pre-built methods (raise_arm, wave_hand, etc.)
2. **Custom rotations** with quaternions
3. **Full joint access** for advanced control
4. **Keyboard controls** (1-9, 0) for interactive testing

This enables creating realistic humanoid poses, gestures, and animations programmatically!
