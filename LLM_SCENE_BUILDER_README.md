# LLM-Driven Scene Builder for Habitat-Lab

Generate 3D scenes in Habitat-Lab using natural language through Claude AI.

## Overview

This project enables you to create and manipulate 3D scenes in Habitat-Lab using natural language descriptions. An LLM (Claude) translates your descriptions into scene specifications that are then executed in the simulator.

### What You Can Do

1. **Spawn Objects**: Place specific objects at exact positions
2. **Control Object States**: Set properties like "clean/dirty", "powered on/off"
3. **Manipulate Articulated Objects**: Open/close doors, drawers, cabinets
4. **Create Complex Scenes**: Arrange multiple objects with spatial relationships

## Files

- `llm_scene_builder.py` - Core scene building infrastructure
- `llm_integration_claude.py` - Claude API integration
- `complete_llm_scene_demo.py` - Full interactive demo
- `humanoid_ultimate_control_smooth.py` - Humanoid control (existing)

## Installation

```bash
# Install dependencies
pip install anthropic habitat-sim habitat-lab opencv-python

# Set your Claude API key
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

## Quick Start

### 1. Basic Usage (Python API)

```python
from llm_scene_builder import HabitatSceneBuilder
from llm_integration_claude import ClaudeSceneGenerator

# Initialize Habitat
import habitat
config = habitat.get_config("benchmark/rearrange/play/play.yaml")
env = habitat.Env(config=config)
obs = env.reset()
sim = env._sim

# Initialize builders
builder = HabitatSceneBuilder(sim)
generator = ClaudeSceneGenerator()

# Generate scene from natural language
scene_spec = generator.generate_scene("Place an apple on the table")

# Build the scene
builder.build_from_spec(scene_spec)
```

### 2. Interactive Demo

```bash
# Run the complete demo
python complete_llm_scene_demo.py

# Example prompts:
# - "Place three soup cans in a row on the counter"
# - "Put an apple and a banana on the table"
# - "Open the fridge door halfway"
# - "Create a dining setup with plate, fork, and knife"
```

## Scene Specification Format

The LLM generates JSON specifications that look like this:

```json
{
  "objects": [
    {
      "handle": "002_master_chef_can",
      "position": [0.5, 1.0, 2.0],
      "rotation": [0, 0, 0, 1],
      "state": {
        "is_clean": true,
        "is_powered_on": false
      }
    }
  ],
  "articulated_objects": [
    {
      "name": "fridge_:0000",
      "link_states": {
        "0": 1.2
      }
    }
  ]
}
```

## Core Components

### HabitatSceneBuilder

The main class for manipulating scenes:

```python
builder = HabitatSceneBuilder(sim)

# Spawn a rigid object
obj = builder.spawn_object(
    handle="002_master_chef_can",
    position=[0.5, 1.0, 2.0],
    rotation=[0, 0, 0, 1],
    state={"is_clean": True}
)

# Control articulated objects (fridge, drawers, etc.)
builder.set_articulated_object_state(
    object_name="fridge_:0000",
    link_states={0: 1.5}  # Open door to 1.5 radians
)

# Clear all spawned objects
builder.clear_scene()

# Build from specification
builder.build_from_spec(scene_spec)
```

### ClaudeSceneGenerator

Handles LLM communication:

```python
generator = ClaudeSceneGenerator(api_key="your-key")

# Generate scene from natural language
scene_spec = generator.generate_scene(
    "Place three objects on the table",
    temperature=0.3  # Lower = more deterministic
)

# Interactive chat mode
generator.chat_mode()
```

## Available Objects

Common object templates in Habitat-Lab:

### Food Items
- `002_master_chef_can`
- `003_cracker_box`
- `004_sugar_box`
- `005_tomato_soup_can`
- `006_mustard_bottle`
- `007_tuna_fish_can`
- `008_pudding_box`
- `009_gelatin_box`
- `010_potted_meat_can`
- `011_banana`
- `apple`, `orange`

### Dishware
- `bowl`
- `plate`
- `cup`
- `fork`
- `spoon`
- `knife`

### Get Full List

```python
sim.get_rigid_object_manager().get_template_handles()
```

## Articulated Objects

Objects with movable parts (doors, drawers):

```python
# Get all articulated objects in scene
ao_handles = sim.get_articulated_object_manager().get_object_handles()
# Example: ['fridge_:0000', 'kitchen_counter_:0001', ...]

# Common link states:
# - Fridge door: link 0, open range 0.0 to 1.5 radians
# - Cabinet door: link 0, open range 0.0 to 1.2 radians
# - Drawer: link varies, open range 0.0 to 0.8 radians
```

## Object States

Habitat's object state machine supports metadata properties:

```python
from habitat.sims.habitat_simulator.object_state_machine import (
    set_state_of_obj,
    get_state_of_obj
)

# Set state
set_state_of_obj(obj, "is_clean", True)
set_state_of_obj(obj, "is_powered_on", False)

# Get state
is_clean = get_state_of_obj(obj, "is_clean")
```

Available states:
- `is_clean` (boolean)
- `is_powered_on` (boolean)
- Custom states can be added

## Coordinate System

Habitat uses a right-handed coordinate system:

- **X-axis**: Left/Right
- **Y-axis**: Up/Down (vertical)
- **Z-axis**: Forward/Back

### Common Heights

- Floor: `y = 0.0`
- Table: `y = 0.8 to 1.0`
- Counter: `y = 0.9 to 1.1`
- Shelves: `y = 1.2, 1.5, 1.8`

### Rotation

Rotations use quaternions `[x, y, z, w]`:
- Identity (no rotation): `[0, 0, 0, 1]`
- 90° around Y: `[0, 0.707, 0, 0.707]`
- 180° around Y: `[0, 1, 0, 0]`

## Example Prompts

### Simple Placement
```
"Place an apple on the kitchen table"
"Put a can on the counter"
```

### Multiple Objects
```
"Place three soup cans in a row on the counter, 30cm apart"
"Create a fruit bowl with an apple, banana, and orange"
```

### Articulated Objects
```
"Open the fridge door halfway"
"Open all drawers in the kitchen cabinet"
"Close all doors and drawers"
```

### Complex Scenes
```
"Set up a dining table with a plate in the center, fork on left, knife on right"
"Arrange breakfast items: cereal box, bowl, and milk on the counter"
"Create a messy kitchen with items scattered on surfaces"
```

## Integration with Humanoid Control

Combine with the humanoid controller:

```python
from llm_scene_builder import HabitatSceneBuilder
from llm_integration_claude import ClaudeSceneGenerator

# ... setup scene with LLM ...

# Then control humanoid to interact with scene
from humanoid_ultimate_control_smooth import (
    configure_humanoid,
    ManualHumanoidController
)

humanoid, controller = configure_humanoid(sim, MOTION_DATA_PATH)

# Humanoid can now pick/place objects in LLM-generated scene
```

## Advanced Usage

### Custom Object Templates

Add your own object templates:

```python
# Register custom object
sim.get_object_template_manager().register_template(
    "path/to/your_object.object_config.json"
)

# Use in scene
builder.spawn_object(
    handle="your_object",
    position=[0, 0, 0]
)
```

### Receptacle-Based Placement

For automatic surface finding:

```python
from habitat.datasets.rearrange.samplers.receptacle import find_receptacles

# Find all receptacles in scene
receptacles = find_receptacles(sim)

# Use receptacle names in LLM prompt
# "Place object on kitchen_table|surface"
```

### Batch Scene Generation

Generate multiple scenes:

```python
prompts = [
    "Clean kitchen with organized items",
    "Messy kitchen after cooking",
    "Breakfast setup on table"
]

for prompt in prompts:
    scene_spec = generator.generate_scene(prompt)
    builder.build_from_spec(scene_spec)
    # ... save observations ...
    builder.clear_scene()
```

## Troubleshooting

### "Object not found" Error

```python
# List available objects
templates = sim.get_rigid_object_manager().get_template_handles()
print(templates)

# Use exact handle from list
```

### "Articulated object not found"

```python
# List all articulated objects
handles = sim.get_articulated_object_manager().get_object_handles()
print(handles)

# Use exact handle (including :0000 suffix)
```

### Objects Falling Through Floor

```python
# Increase settling time
for _ in range(50):  # More physics steps
    sim.step_physics(1.0 / 60.0)

# Or adjust position height
position[1] += 0.1  # Raise object 10cm
```

### LLM Returns Invalid JSON

```python
# Increase temperature for less strict formatting
scene_spec = generator.generate_scene(prompt, temperature=0.1)

# Or use try-except with retry
try:
    scene_spec = generator.generate_scene(prompt)
except json.JSONDecodeError:
    # Retry or use fallback
    pass
```

## Performance Tips

1. **Reuse simulator**: Don't create new sim for each scene
2. **Batch operations**: Group multiple spawns together
3. **Limit physics steps**: Use minimum needed for stability
4. **Clear unused objects**: Call `builder.clear_scene()` regularly

## API Reference

### HabitatSceneBuilder Methods

- `spawn_object(handle, position, rotation=None, state=None)` - Add rigid object
- `set_articulated_object_state(name, link_states)` - Control doors/drawers
- `clear_scene()` - Remove all spawned objects
- `build_from_spec(scene_spec)` - Build from JSON specification

### ClaudeSceneGenerator Methods

- `generate_scene(prompt, temperature=0.3)` - Generate from natural language
- `chat_mode()` - Interactive command-line interface

## License

Same as Habitat-Lab (MIT License)

## Support

For issues:
1. Check Habitat-Lab documentation: https://aihabitat.org/docs/habitat-lab/
2. Verify API key is set: `echo $ANTHROPIC_API_KEY`
3. Ensure dependencies installed: `pip list | grep habitat`

## Contributing

To extend this system:

1. Add new object state types in `object_state_machine.py`
2. Implement receptacle-based placement
3. Add more articulated object types
4. Create custom LLM prompts for specific use cases
