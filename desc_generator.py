import pickle
import argparse
import os

# Define descriptions for different tasks
task_descriptions = {
    "StackCube-v1": [
        "pick up a red cube and stack it on top of a green cube and let go of the cube without it falling."
    ],
    "PushCube-v1": [
        "push the cube to the red white target on the table"
    ],
    "PickCube-v1": [
        "grasp the red cube and move it to the target goal position, then release the cube in a stable position without it falling."
    ],
    "PokeCube-v1": [
        "Pick a peg and then use it to poke a red cube and then push it to a red white target goal position."
    ],
    "PegInsertionSide-v1": [
        "Pick up a orange-white peg and insert the orange end into the box with a hole in it."
    ]
}

# Argument parsing
parser = argparse.ArgumentParser(description="Generate task descriptions and save to specified directory.")
parser.add_argument("--task", type=str, required=True, help="Task name, e.g., 'PushCube-v1'")
parser.add_argument("--save_dir", type=str, required=True, help="Directory where the description file will be saved")
args = parser.parse_args()

# Get the description for the task
desc = task_descriptions.get(args.task, [])
if not desc:
    raise ValueError(f"No description found for task '{args.task}'")

# Ensure the save directory exists
os.makedirs(args.save_dir, exist_ok=True)

# Save the description to a file
save_path = os.path.join(args.save_dir, 'desc.pkl')
with open(save_path, 'wb') as file:
    pickle.dump(desc, file)

print(f"Description for '{args.task}' saved to '{save_path}'")
