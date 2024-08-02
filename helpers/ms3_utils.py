import h5py
from mani_skill.utils.io_utils import load_json

def get_ms_demos(traj_path, json_path):
    # Load associated h5 file
    h5_data = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = load_json(json_path)
    return h5_data, json_data
