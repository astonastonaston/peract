import pickle

# desc = ["close the gripper",
#         "move the gripper to the back of the cube on the table",
#         "push the cube to the center of the red white target on the table"]
# desc = ["push the cube to the center of the red white target"]
# desc = ["push the cube on the table to the center of the red white target on the table"]
desc = ["A simple task where the objective is to reach a cube and then push the cube to a red white goal region in front of it"] # long 1
#         The cube xy position is randomized on top of a table in the region [0.1, 0.1] times [-0.1, -0.1]. It is placed flat on the table. \
#         The target goal region is marked by a red white circular target. The position of the target is fixed to be the cube xy position + [0.1 + goal radius, 0]. \
#         The success condition is that the cube xy position is within goal_radius (default 0.1) of the target xy position by euclidean distance.\
#         "] 
# desc = ["\
#     Task Description: A simple task where the objective is to reach a cube and then push and move the cube to a goal region in front of it. \
#     Randomizations: The cube’s xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table. The target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.1 + goal_radius, 0]. \
#     Success Conditions: The cube’s xy position is within goal_radius (default 0.1) of the target’s xy position by euclidean distance.\
#     "] # long 1

# Save the description to a file
with open('demos/PushCube-v1/motionplanning/desc.pkl', 'wb') as file:
    pickle.dump(desc, file)
