#!/usr/bin/env python3
"""
Claude LLM Integration for Habitat-Lab Scene Builder

Shows how to connect Claude AI to generate scene specifications from natural language.

Requirements:
    pip install anthropic
"""

import json
import os
from typing import Dict, Any
import anthropic

# ============================================================================
# CLAUDE SYSTEM PROMPT
# ============================================================================

SCENE_BUILDER_SYSTEM_PROMPT = """You are a scene specification generator for Habitat-Lab 3D simulator.

Your task: Convert natural language scene descriptions into JSON specifications.

Output format:
{
    "objects": [
        {
            "handle": "object_template_name",
            "position": [x, y, z],
            "rotation": [x, y, z, w],
            "state": {
                "is_clean": true/false,
                "is_powered_on": true/false
            }
        }
    ],
    "articulated_objects": [
        {
            "name": "object_instance_name",
            "link_states": {
                0: 1.5
            }
        }
    ]
}

Available object templates:
- "002_master_chef_can"
- "003_cracker_box"
- "004_sugar_box"
- "005_tomato_soup_can"
- "006_mustard_bottle"
- "007_tuna_fish_can"
- "008_pudding_box"
- "009_gelatin_box"
- "010_potted_meat_can"
- "011_banana"
- "apple", "orange"
- "bowl", "plate", "cup", "fork", "spoon", "knife"

Position guidelines:
- Floor: y = 0.0
- Table: y = 0.8 to 1.0
- Counter: y = 0.9 to 1.1
- Shelves: y = 1.2, 1.5, 1.8

Articulated objects:
- Fridge door: link 0, open = 1.2 to 1.5 radians
- Cabinet door: link 0, open = 0.8 to 1.2 radians
- Drawer: link varies, open = 0.3 to 0.8 radians

Rules:
1. Only output valid JSON
2. Use reasonable spacing (0.3-0.5m apart)
3. Default rotation: [0, 0, 0, 1]
4. Y-axis is up
5. No overlapping or floating objects
"""


# ============================================================================
# CLAUDE INTEGRATION
# ============================================================================

class ClaudeSceneGenerator:
    """Handles communication with Claude API for scene generation."""

    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Claude scene generator.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter"
            )

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_scene(
        self,
        user_prompt: str,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate scene specification from natural language.

        Args:
            user_prompt: Natural language scene description
            temperature: Claude temperature (0.0 = deterministic, 1.0 = creative)

        Returns:
            Scene specification dictionary
        """
        print(f"\n[Claude] Processing prompt: '{user_prompt}'")

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SCENE_BUILDER_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )

            # Extract text response
            content = message.content[0].text

            # Parse JSON
            scene_spec = json.loads(content)

            print(f"[Claude] Successfully generated scene specification")
            print(f"         Objects: {len(scene_spec.get('objects', []))}")
            print(f"         Articulated: {len(scene_spec.get('articulated_objects', []))}")

            return scene_spec

        except json.JSONDecodeError as e:
            print(f"[ERROR] Claude did not return valid JSON: {e}")
            print(f"        Raw output: {content}")
            return {}
        except Exception as e:
            print(f"[ERROR] Claude API call failed: {e}")
            return {}

    def chat_mode(self):
        """
        Interactive chat mode for testing scene generation.
        """
        print("\n" + "="*80)
        print("Claude Scene Generator - Chat Mode")
        print("="*80)
        print("Enter scene descriptions in natural language.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Exiting chat mode.")
                    break

                if not user_input:
                    continue

                scene_spec = self.generate_scene(user_input)

                if scene_spec:
                    print("\nGenerated JSON:")
                    print(json.dumps(scene_spec, indent=2))
                    print()

            except KeyboardInterrupt:
                print("\nExiting chat mode.")
                break
            except Exception as e:
                print(f"[ERROR] {e}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def main():
    """
    Demonstrates Claude integration for scene generation.
    """
    print("\n" + "="*80)
    print("Claude Scene Generator for Habitat-Lab")
    print("="*80)

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[ERROR] ANTHROPIC_API_KEY not set.")
        print("        Export it: export ANTHROPIC_API_KEY='your-key-here'")
        print("        Or pass it: generator = ClaudeSceneGenerator(api_key='...')")
        return

    # Initialize generator
    generator = ClaudeSceneGenerator()

    # Example 1: Simple scene
    print("\n" + "-"*80)
    print("Example 1: Simple object placement")
    print("-"*80)
    prompt1 = "Place an apple on the kitchen table"
    spec1 = generator.generate_scene(prompt1)
    if spec1:
        print("Result:", json.dumps(spec1, indent=2))

    # Example 2: Multiple objects
    print("\n" + "-"*80)
    print("Example 2: Multiple objects")
    print("-"*80)
    prompt2 = "Put three soup cans in a row on the counter, spaced 30cm apart"
    spec2 = generator.generate_scene(prompt2)
    if spec2:
        print("Result:", json.dumps(spec2, indent=2))

    # Example 3: Articulated objects
    print("\n" + "-"*80)
    print("Example 3: Opening articulated objects")
    print("-"*80)
    prompt3 = "Open the fridge door halfway"
    spec3 = generator.generate_scene(prompt3)
    if spec3:
        print("Result:", json.dumps(spec3, indent=2))

    # Example 4: Complex scene
    print("\n" + "-"*80)
    print("Example 4: Complex scene")
    print("-"*80)
    prompt4 = "Create a dining table setup with a plate in the center, fork on the left, knife on the right, and a cup in the top right corner"
    spec4 = generator.generate_scene(prompt4)
    if spec4:
        print("Result:", json.dumps(spec4, indent=2))

    # Interactive mode
    print("\n" + "-"*80)
    print("Starting interactive chat mode...")
    print("-"*80)
    generator.chat_mode()


if __name__ == "__main__":
    main()
