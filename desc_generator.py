import pickle

desc = ["push the cube to the center of the red white target"]

# Save the description to a file
with open('demos/PushCube-v1/motionplanning/desc.pkl', 'wb') as file:
    pickle.dump(desc, file)
