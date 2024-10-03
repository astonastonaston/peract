import pickle

# desc = ["close the gripper",
#         "move the gripper to the back of the cube",
#         "push the cube to the center of the red white target"]
desc = ["push the cube on the table to the center of the red white target on the table"]

# Save the description to a file
with open('demos/PushCube-v1/motionplanning/desc.pkl', 'wb') as file:
    pickle.dump(desc, file)
