import os
import json
import pickle as pkl
import imageio as iio
import numpy as np
import torch
from metasim.types import EnvState
from metasim.utils.io_util import write_16bit_depth_video
from roboverse_learn.collect_real_world.vis_calibration import draw_world_frame_axes
def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    return (depth - depth.min()) / (depth.max() - depth.min())


def save_single_demo(save_dir: str, demo):
    """Save a demo to a directory.

    Args:
        save_dir: The directory to save the demo.
        demo: The demo to save.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get the main robot name (assuming first robot in first state)
    robot_name = "franka"
    # Get the main camera name (assuming first camera in first state)
    camera_name = "camera0"
    # Convert and prepare data for saving
    rgbs = []
    depths = []
    jsondata = {
        "depth_min": [],
        "depth_max": [],
        "cam_intr": [],
        "cam_extr": [],
        "joint_qpos_target": [],
        "joint_qpos": [],
        "robot_root_state": [],
    }
    robot_root_state = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Process each timestep
    for i, state in enumerate(demo):
        # Extract robot state
        robot_state = state["robots"][robot_name]
        camera_state = state["cameras"][camera_name]

        if "rgb" in camera_state:
            rgb = camera_state["rgb"].cpu().numpy()
            rgbs.append(rgb)

        if "depth" in camera_state:
            depth = camera_state["depth"].cpu().numpy()
            depths.append(_normalize_depth(depth))
            jsondata["depth_min"].append(depth.min().item())
            jsondata["depth_max"].append(depth.max().item())
        # print(f"camera_state.keys(): {camera_state.keys()}")  # Debugging line
        # Extract camera data
        jsondata["cam_intr"].append(camera_state["intrinsics"].tolist() if "intrinsics" in camera_state else [])
        jsondata["cam_extr"].append(camera_state["extrinsics"].tolist() if "extrinsics" in camera_state else [])

        # Extract robot data
        jsondata["joint_qpos"].append(robot_state["dof_pos"])

        # For targets, handle them in the same way as the original function
        ## XXX
        if next(iter(demo[0]["robots"].values())).get("dof_pos_target", None) is not None:
            if i < len(demo) - 1:
                next_robot_state = demo[i + 1]["robots"][robot_name]
                target_dof_pos = next_robot_state["dof_pos_target"]
            else:
                # For the last timestep, use the same target as the current state
                target_dof_pos = robot_state["dof_pos_target"]
        else:
            raise ValueError(
                f"The demo does not contain 'dof_pos_target' in robot states. what we've got is: {next(iter(demo[0]['robots'].values())).keys()}"
            )
            # if i < len(demo) - 1:
            #     next_robot_state = demo[i + 1]["robots"][robot_name]
            #     target_dof_pos = next_robot_state["dof_pos"]
            # else:
            #     target_dof_pos = robot_state["dof_pos"]

        jsondata["joint_qpos_target"].append(target_dof_pos)


        # Extract root and body state
        jsondata["robot_root_state"].append(
            robot_root_state
        )

    # Save video files
    if rgbs:
        #print(f"rgb shape: {rgbs[0].shape}, dtype: {rgbs[0].dtype}, min: {rgbs[0].min()}, max: {rgbs[0].max()}")
        iio.mimsave(os.path.join(save_dir, "rgb.mp4"), rgbs, fps=30, quality=10, macro_block_size=1)
        rgb = rgbs[0]
        intr = jsondata["cam_intr"][0]
        extr = jsondata["cam_extr"][0]
        draw_world_frame_axes(
            rgb,
            np.array(intr).reshape(3, 3),
            np.array(extr).reshape(4, 4),
            save_path=os.path.join(save_dir, "vis_calibration.png")
        )
        iio.imwrite(os.path.join(save_dir, "demo_rgb.png"), rgb)
    if depths:
        #print(f"depth shape: {depths[0].shape}, dtype: {depths[0].dtype}, min: {depths[0].min()}, max: {depths[0].max()}")
        #write_16bit_depth_video(os.path.join(save_dir, "depth_uint16.mkv"), (depths * 65535).astype(np.uint16), fps=30)
        iio.mimsave(
            os.path.join(save_dir, "depth_uint8.mp4"),
            [(depth * 255).astype(np.uint8) for depth in depths],
            fps=30,
            quality=10,
            macro_block_size=1
        )
        depth = (depths[0] * 255).astype(np.uint8)
        iio.imwrite(os.path.join(save_dir, "demo_depth.png"), depth)
    # Save metadata
    json.dump(jsondata, open(os.path.join(save_dir, "metadata.json"), "w"))
    pkl.dump(jsondata, open(os.path.join(save_dir, "metadata.pkl"), "wb"))

    # Mark as finished
    with open(os.path.join(save_dir, "status.txt"), "w+") as f:
        f.write("success")
