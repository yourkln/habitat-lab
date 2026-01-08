#!/usr/bin/env python3
"""
Complete LLM Scene Builder Demo

Combines Claude AI with Habitat-Lab scene builder for natural language scene creation.

Usage:
    1. Set your API key: export ANTHROPIC_API_KEY='your-key-here'
    2. Run: python complete_llm_scene_demo.py
    3. Enter natural language scene descriptions
    4. Watch the scene build in Habitat-Lab

Requirements:
    pip install anthropic habitat-sim habitat-lab opencv-python
"""

import os
import time
import habitat
import habitat_sim
from llm_scene_builder import HabitatSceneBuilder
from llm_integration_claude import ClaudeSceneGenerator


def main():
    """
    Full pipeline: Natural language -> Claude -> Scene specification -> Habitat visualization
    """
    print("\n" + "="*80)
    print("Complete LLM-Driven Scene Builder for Habitat-Lab")
    print("="*80)

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n[ERROR] ANTHROPIC_API_KEY not set.")
        print("        Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("\nRunning in DEMO mode with pre-defined scenes instead...\n")
        use_real_llm = False
    else:
        use_real_llm = True

    # Initialize Habitat environment
    print("\n[HABITAT] Initializing environment...")
    config = habitat.get_config("benchmark/rearrange/play/play.yaml")

    with habitat.config.read_write(config):
        config.habitat.environment.max_episode_steps = 0
        config.habitat.simulator.debug_render = True

    with habitat.Env(config=config) as env:
        obs = env.reset()
        sim = env._sim

        # Initialize scene builder
        builder = HabitatSceneBuilder(sim)

        # Initialize Claude generator (if API key available)
        if use_real_llm:
            generator = ClaudeSceneGenerator()
        else:
            generator = None

        print("[HABITAT] Environment ready")
        print(f"          Scene: {sim.curr_scene_name}")
        print(f"          Available objects: {len(sim.get_rigid_object_manager().get_template_handles())}")
        print(f"          Articulated objects: {sim.get_articulated_object_manager().get_object_handles()}")

        # Main interaction loop
        print("\n" + "="*80)
        print("Interactive Scene Builder")
        print("="*80)
        print("Enter natural language descriptions to build scenes.")
        print("Commands:")
        print("  'clear'  - Remove all spawned objects")
        print("  'info'   - Show available objects")
        print("  'quit'   - Exit")
        print("="*80 + "\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    break

                elif user_input.lower() == 'clear':
                    builder.clear_scene()
                    print("[CLEARED] All spawned objects removed")
                    continue

                elif user_input.lower() == 'info':
                    rom = sim.get_rigid_object_manager()
                    templates = rom.get_template_handles()
                    print(f"\n[INFO] Available objects ({len(templates)} total):")
                    for i, template in enumerate(templates[:20], 1):
                        print(f"       {i}. {template}")
                    if len(templates) > 20:
                        print(f"       ... and {len(templates) - 20} more")

                    ao_handles = sim.get_articulated_object_manager().get_object_handles()
                    print(f"\n[INFO] Articulated objects ({len(ao_handles)} total):")
                    for i, handle in enumerate(ao_handles, 1):
                        print(f"       {i}. {handle}")
                    continue

                # Generate scene specification
                if use_real_llm:
                    print("\n[CLAUDE] Generating scene specification...")
                    scene_spec = generator.generate_scene(user_input)
                else:
                    # Demo mode - use predefined specs
                    print("\n[DEMO] Using predefined scene (no API key)")
                    scene_spec = get_demo_scene(user_input)

                if not scene_spec:
                    print("[ERROR] Failed to generate scene specification")
                    continue

                # Display specification
                print("\n[SPEC] Scene specification:")
                import json
                print(json.dumps(scene_spec, indent=2))

                # Build scene in Habitat
                print("\n[BUILD] Applying to Habitat environment...")
                builder.build_from_spec(scene_spec)

                # Step physics to visualize
                print("[PHYSICS] Settling objects...")
                for _ in range(30):
                    sim.step_physics(1.0 / 60.0)

                print("[SUCCESS] Scene built! Check your visualization.")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback
                traceback.print_exc()


def get_demo_scene(user_input: str):
    """
    Returns demo scene specs when no API key is available.
    """
    user_lower = user_input.lower()

    if "apple" in user_lower or "banana" in user_lower:
        return {
            "objects": [
                {
                    "handle": "011_banana",
                    "position": [0.5, 1.0, 2.0],
                    "rotation": [0, 0, 0, 1]
                }
            ],
            "articulated_objects": []
        }
    elif "can" in user_lower or "soup" in user_lower:
        return {
            "objects": [
                {
                    "handle": "002_master_chef_can",
                    "position": [0.5, 1.0, 2.0],
                    "rotation": [0, 0, 0, 1]
                },
                {
                    "handle": "005_tomato_soup_can",
                    "position": [0.7, 1.0, 2.0],
                    "rotation": [0, 0, 0, 1]
                }
            ],
            "articulated_objects": []
        }
    elif "fridge" in user_lower:
        return {
            "objects": [],
            "articulated_objects": [
                {
                    "name": "fridge_:0000",
                    "link_states": {0: 1.2}
                }
            ]
        }
    else:
        # Default: place a single can
        return {
            "objects": [
                {
                    "handle": "002_master_chef_can",
                    "position": [0.5, 1.0, 2.0],
                    "rotation": [0, 0, 0, 1]
                }
            ],
            "articulated_objects": []
        }


if __name__ == "__main__":
    main()
