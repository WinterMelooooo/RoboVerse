import json

import torch

path1 = "/home/ghr/yktang/RoboVerse/info/outputs/DP/2025.04.20/04.53.50_CloseBoxFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/150.ckpt"
path2 = "/home/ghr/yktang/RoboVerse/info/outputs/DP/2025.04.20/04.53.50_CloseBoxFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/200.ckpt"
ckpt1 = torch.load(path1, map_location="cpu")
ckpt2 = torch.load(path2, map_location="cpu")
sd1 = ckpt1["state_dicts"]["model"]
sd2 = ckpt2["state_dicts"]["model"]

keys1 = {k for k in sd1.keys() if k.startswith("obs_encoder")}
keys2 = {k for k in sd2.keys() if k.startswith("obs_encoder")}
common_keys = sorted(keys1 & keys2)

data1 = {}
data2 = {}
diff = {}

for k in common_keys:
    t1 = sd1[k]
    t2 = sd2[k]
    v1 = t1.cpu().tolist()
    v2 = t2.cpu().tolist()

    if not torch.equal(t1, t2):
        print("not equal")
        diff[k] = [v1, v2]
    else:
        diff[k] = ["equal"]
        print("not equal")
