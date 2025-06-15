import json

sensordata_path = r"/home/ghr/yktang/RoboVerse/roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka/demo_0000/sensordata.json"
metadata_path = r"/home/ghr/yktang/RoboVerse/roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka/demo_0000/metadata.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
with open(sensordata_path, 'r') as f:
    data = json.load(f)

for key, value in data.items():
    if hasattr(value, 'shape'):
        print(f"shape {key}: {value.shape}")
    elif hasattr(value, '__len__'):
        print(f"len {key}: {len(value)}")
    else:
        print(f"{key}: {value}")

for key, value in metadata.items():
    if hasattr(value, 'shape'):
        print(f"shape {key}: {value.shape}")
    elif hasattr(value, '__len__'):
        print(f"len {key}: {len(value)}")
    else:
        print(f"{key}: {value}")
