#!/usr/bin/env python3
"""
LLM-Driven Scene Builder for Habitat-Lab

This script shows how to use natural language -> JSON/Python to create custom scenes.
You can integrate with any LLM (GPT-4, Claude, LLaMA, etc.) to generate scene specifications.

Example workflow:
1. User: "Create a kitchen with an apple on the table and the fridge open"
2. LLM generates JSON scene spec
3. This script loads it into Habitat-Lab
"""

import json
import magnum as mn
import numpy as np
from typing import Dict, List, Any, Optional
import habitat
import habitat_sim
from habitat.sims.habitat_simulator.object_state_machine import (
    set_state_of_obj,
    get_state_of_obj,
)


# ============================================================================
# SCENE SPECIFICATION FORMAT (for LLM to generate)
# ============================================================================

EXAMPLE_SCENE_SPEC = {
    "scene": "data/scene_datasets/hssd-hab/scenes/103997919_171031233.scene_instance.json",
    "objects": [
        {
            "handle": "002_master_chef_can",  # Object template name
            "position": [0.5, 1.0, 2.0],  # [x, y, z] in meters
            "rotation": [0, 0, 0, 1],  # Quaternion [x, y, z, w]
            "state": {
                "is_clean": True,
                "is_powered_on": False
            }
        },
        {
            "handle": "004_sugar_box",
            "position": [0.3, 1.0, 2.1],
            "rotation": [0, 0, 0, 1],
            "receptacle": "kitchen_table|surface"  # Place on specific receptacle
        },
        {
            "handle": "003_cracker_box",
            "position": [-0.5, 0.5, 1.8],
            "rotation": [0, 0.707, 0, 0.707]  # 90 degree rotation
        }
    ],
    "articulated_objects": [
        {
            "name": "fridge",  # Instance name in scene
            "link_states": {
                0: 1.2,  # Link 0 (door) open to 1.2 radians (~69 degrees)
            }
        },
        {
            "name": "kitchen_counter_:0000",
            "link_states": {
                1: 0.8,  # Drawer 1 open 0.8 radians
                2: 0.0   # Drawer 2 closed
            }
        }
    ],
    "humanoid": {
        "position": [0.0, 0.0, 0.0],
        "rotation_y": 0.0  # Facing direction in radians
    }
}


# ============================================================================
# SCENE BUILDER CLASS
# ============================================================================

class HabitatSceneBuilder:
    """
    Builds custom Habitat scenes from JSON specifications.
    Can be driven by LLM outputs for natural language scene creation.
    """

    def __init__(self, sim: habitat_sim.Simulator):
        """
        Args:
            sim: An initialized Habitat-Sim Simulator instance
        """
        self.sim = sim
        self.rigid_obj_mgr = sim.get_rigid_object_manager()
        self.ao_mgr = sim.get_articulated_object_manager()

        # Track spawned objects for cleanup
        self.spawned_object_ids = []

    def clear_scene(self):
        """Remove all dynamically spawned objects."""
        for obj_id in self.spawned_object_ids:
            self.rigid_obj_mgr.remove_object_by_id(obj_id)
        self.spawned_object_ids = []

    def spawn_object(
        self,
        handle: str,
        position: List[float],
        rotation: Optional[List[float]] = None,
        state: Optional[Dict[str, Any]] = None
    ) -> habitat_sim.physics.ManagedRigidObject:
        """
        Spawn a rigid object at a specific location with optional state.

        Args:
            handle: Object template handle (e.g., "002_master_chef_can")
            position: [x, y, z] position in world coordinates
            rotation: [x, y, z, w] quaternion rotation (optional)
            state: Dictionary of object states (e.g., {"is_clean": True})

        Returns:
            The spawned ManagedRigidObject instance
        """
        # Add object to scene
        obj = self.rigid_obj_mgr.add_object_by_template_handle(handle)

        if obj is None:
            print(f"[ERROR] Failed to spawn object: {handle}")
            available = self.rigid_obj_mgr.get_template_handles()[:5]
            print(f"        Available objects: {available}...")
            return None

        # Set position
        obj.translation = mn.Vector3(*position)

        # Set rotation (quaternion)
        if rotation is not None:
            quat = mn.Quaternion(
                mn.Vector3(rotation[0], rotation[1], rotation[2]),
                rotation[3]
            )
            obj.rotation = quat

        # Set object states (is_clean, is_powered_on, etc.)
        if state is not None:
            for state_name, state_value in state.items():
                set_state_of_obj(obj, state_name, state_value)

        # Track for cleanup
        self.spawned_object_ids.append(obj.object_id)

        print(f"[SUCCESS] Spawned {handle} at {position}")
        return obj

    def set_articulated_object_state(
        self,
        object_name: str,
        link_states: Dict[int, float]
    ):
        """
        Set articulated object joint positions (e.g., open fridge door).

        Args:
            object_name: Name of articulated object in scene
            link_states: {link_index: joint_position} mapping

        Example:
            set_articulated_object_state("fridge", {0: 1.5})  # Open door
        """
        obj = self.ao_mgr.get_object_by_handle(object_name)

        if obj is None:
            print(f"[ERROR] Articulated object not found: {object_name}")
            print(f"        Available: {self.ao_mgr.get_object_handles()}")
            return

        # Set each link position
        for link_idx, position in link_states.items():
            if link_idx < len(obj.joint_positions):
                # Get current positions
                joint_positions = obj.joint_positions
                joint_positions[link_idx] = position
                obj.joint_positions = joint_positions
                print(f"[SUCCESS] Set {object_name} link {link_idx} to {position}")
            else:
                print(f"[WARNING] Invalid link index {link_idx} for {object_name}")

    def spawn_on_receptacle(
        self,
        handle: str,
        receptacle_name: str,
        offset: Optional[List[float]] = None
    ) -> Optional[habitat_sim.physics.ManagedRigidObject]:
        """
        Spawn object on a specific receptacle (e.g., "table|surface").

        NOTE: This requires the receptacle system to be initialized.
        For simple placement, use spawn_object() with calculated positions.
        """
        print(f"[WARNING] Receptacle placement requires full receptacle system setup")
        print(f"          Use spawn_object() with calculated positions for now")
        return None

    def build_from_spec(self, scene_spec: Dict[str, Any]):
        """
        Build entire scene from JSON specification.

        Args:
            scene_spec: Dictionary with "objects" and "articulated_objects" keys
        """
        print("\n[BUILDER] Building scene from specification...")

        # Clear existing objects
        self.clear_scene()

        # Spawn rigid objects
        if "objects" in scene_spec:
            for obj_spec in scene_spec["objects"]:
                self.spawn_object(
                    handle=obj_spec["handle"],
                    position=obj_spec["position"],
                    rotation=obj_spec.get("rotation"),
                    state=obj_spec.get("state")
                )

        # Set articulated object states
        if "articulated_objects" in scene_spec:
            for ao_spec in scene_spec["articulated_objects"]:
                self.set_articulated_object_state(
                    object_name=ao_spec["name"],
                    link_states=ao_spec["link_states"]
                )

        # Step physics to settle objects
        for _ in range(10):
            self.sim.step_physics(1.0 / 60.0)

        print("[BUILDER] Scene built successfully!\n")


# ============================================================================
# LLM INTEGRATION EXAMPLE
# ============================================================================

def llm_prompt_to_scene_spec(user_prompt: str) -> Dict[str, Any]:
    """
    This function would call your LLM (GPT-4, Claude, etc.) to convert
    natural language to scene specification JSON.

    Example prompts:
    - "Put an apple on the kitchen table"
    - "Open the fridge door halfway"
    - "Place three cans in a row on the counter"

    For now, returns example spec. Replace with actual LLM call.
    """
    print(f"[LLM] Processing: '{user_prompt}'")
    print("      (In production, this would call GPT-4/Claude API)")

    # MOCK LLM RESPONSE - Replace with actual LLM call
    if "apple" in user_prompt.lower() and "table" in user_prompt.lower():
        return {
            "objects": [
                {
                    "handle": "apple",  # Or "003_apple" depending on your dataset
                    "position": [0.5, 1.0, 2.0],
                    "rotation": [0, 0, 0, 1]
                }
            ],
            "articulated_objects": []
        }
    elif "fridge" in user_prompt.lower() and "open" in user_prompt.lower():
        return {
            "objects": [],
            "articulated_objects": [
                {
                    "name": "fridge_:0000",
                    "link_states": {0: 1.5}  # Open door
                }
            ]
        }
    else:
        # Return example spec
        return EXAMPLE_SCENE_SPEC


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Demonstrates how to use the scene builder with or without LLM.
    """
    # Setup Habitat environment
    config = habitat.get_config("benchmark/rearrange/play/play.yaml")

    with habitat.config.read_write(config):
        config.habitat.environment.max_episode_steps = 0
        config.habitat.simulator.debug_render = True

    with habitat.Env(config=config) as env:
        obs = env.reset()
        sim = env._sim

        # Initialize scene builder
        builder = HabitatSceneBuilder(sim)

        print("\n" + "="*80)
        print("LLM-Driven Scene Builder for Habitat-Lab")
        print("="*80)

        # Example 1: Manual JSON specification
        print("\n[EXAMPLE 1] Manual Scene Specification")
        manual_spec = {
            "objects": [
                {
                    "handle": "002_master_chef_can",
                    "position": [0.5, 1.0, 2.0],
                    "rotation": [0, 0, 0, 1],
                    "state": {"is_clean": True}
                }
            ],
            "articulated_objects": []
        }
        builder.build_from_spec(manual_spec)

        # Example 2: LLM-driven (mock)
        print("\n[EXAMPLE 2] LLM Natural Language")
        user_prompt = "Put an apple on the kitchen table"
        scene_spec = llm_prompt_to_scene_spec(user_prompt)
        builder.build_from_spec(scene_spec)

        # Example 3: Articulated object control
        print("\n[EXAMPLE 3] Opening Fridge Door")
        ao_handles = sim.get_articulated_object_manager().get_object_handles()
        print(f"            Available articulated objects: {ao_handles}")

        # Find fridge and open it
        for handle in ao_handles:
            if "fridge" in handle.lower():
                builder.set_articulated_object_state(handle, {0: 1.2})
                break

        print("\n[COMPLETE] Demo complete! Scene is ready.")
        print("           Add visualization code to see results.")


if __name__ == "__main__":
    main()
