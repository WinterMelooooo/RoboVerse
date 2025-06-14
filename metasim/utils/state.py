"""Tensorized state of the simulation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch
from loguru import logger as log

from metasim.types import EnvState
from metasim.utils.math import convert_camera_frame_orientation_convention

try:
    from metasim.sim.base import BaseSimHandler
except:
    pass


@dataclass
class ContactForceState:
    """State of a single contact force sensor."""

    force: torch.Tensor
    """Contact force. Shape is (num_envs, 3)."""


SensorState = ContactForceState


@dataclass
class ObjectState:
    """State of a single object."""

    root_state: torch.Tensor
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    body_names: list[str] | None = None
    """Body names. This is only available for articulation objects."""
    body_state: torch.Tensor | None = None
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13). This is only available for articulation objects."""
    joint_pos: torch.Tensor | None = None
    """Joint positions. Shape is (num_envs, num_joints). This is only available for articulation objects."""
    joint_vel: torch.Tensor | None = None
    """Joint velocities. Shape is (num_envs, num_joints). This is only available for articulation objects."""


@dataclass
class RobotState:
    """State of a single robot."""

    root_state: torch.Tensor
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    body_names: list[str]
    """Body names."""
    body_state: torch.Tensor
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13)."""
    joint_pos: torch.Tensor
    """Joint positions. Shape is (num_envs, num_joints)."""
    joint_vel: torch.Tensor
    """Joint velocities. Shape is (num_envs, num_joints)."""
    joint_pos_target: torch.Tensor
    """Joint positions target. Shape is (num_envs, num_joints)."""
    joint_vel_target: torch.Tensor
    """Joint velocities target. Shape is (num_envs, num_joints)."""
    joint_effort_target: torch.Tensor
    """Joint effort targets. Shape is (num_envs, num_joints)."""


@dataclass
class CameraState:
    """State of a single camera."""

    ## Images
    rgb: torch.Tensor | None
    """RGB image. Shape is (num_envs, H, W, 3)."""
    depth: torch.Tensor | None
    """Depth image. Shape is (num_envs, H, W)."""
    instance_id_seg: torch.Tensor | None = None
    """Instance id segmentation for each pixel. Shape is (num_envs, H, W)."""
    instance_id_seg_id2label: dict[int, str] | None = None
    """Instance id segmentation id to label mapping. Keys are instance ids, values are labels. Go together with :attr:`instance_id_seg`."""
    instance_seg: torch.Tensor | None = None
    """Instance segmentation for each pixel. Shape is (num_envs, H, W).

    .. warning::
        This is experimental and subject to change.
    """
    instance_seg_id2label: dict[int, str] | None = None
    """Instance segmentation id to label mapping. Keys are instance ids, values are labels. Go together with :attr:`instance_seg`.

    .. warning::
        This is experimental and subject to change.
    """

    ## Camera parameters
    pos: torch.Tensor | None = None  # TODO: remove N
    """Position of the camera. Shape is (num_envs, 3)."""
    quat_world: torch.Tensor | None = None  # TODO: remove N
    """Quaternion ``(w, x, y, z)`` of the camera, following the world frame convention. Shape is (num_envs, 4).

    Note:
        World frame convention follows the camera aligned with forward axis +X and up axis +Z.
    """
    intrinsics: torch.Tensor | None = None  # TODO: remove N
    """Intrinsics matrix of the camera. Shape is (num_envs, 3, 3)."""

    @property
    def quat_ros(self) -> torch.Tensor:
        """Quaternion ``(w, x, y, z)`` of the camera, following the ROS convention. Shape is (num_envs, 4).

        Note:
            ROS convention follows the camera aligned with forward axis +Z and up axis -Y.
        """
        return convert_camera_frame_orientation_convention(self.quat_world, origin="world", target="ros")

    @property
    def quat_opengl(self) -> torch.Tensor:
        """Quaternion ``(w, x, y, z)`` of the camera, following the OpenGL convention. Shape is (num_envs, 4).

        Note:
            OpenGL convention follows the camera aligned with forward axis -Z and up axis +Y.
        """
        return convert_camera_frame_orientation_convention(self.quat_world, origin="world", target="opengl")

    @property
    def extrinsics(self) -> torch.Tensor:
        # quat_world shape: (num_envs, 4) with components (w, x, y, z)
        # pos shape: (num_envs, 3)
        DEFAULT_CAM_POS = [1.5, 0.0, 1.5]
        if self.pos.shape[0] > 1:
            # ugly but works, since for random.level >= 1, we only use num_envs=1
            self.pos = self.pos.new_tensor(DEFAULT_CAM_POS).unsqueeze(0).repeat(self.pos.size(0), 1)
        w, x, y, z = (
            self.quat_world[:, 0],
            self.quat_world[:, 1],
            self.quat_world[:, 2],
            self.quat_world[:, 3],
        )
        w = self.quat_world[:, 0]
        x = -self.quat_world[:, 1]
        y = -self.quat_world[:, 2]
        z = -self.quat_world[:, 3]

        # 1. 计算旋转矩阵 R (shape: num_envs x 3 x 3)
        R = torch.empty(
            self.quat_world.shape[0],
            3,
            3,
            device=self.quat_world.device,
            dtype=self.quat_world.dtype,
        )
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        #   new_row0 = -old_row1
        #   new_row1 = -old_row2
        #   new_row2 =  old_row0
        R = torch.stack(
            [
                -R[:, 1, :],  # x_cam 对应原来的 row1，再取反
                -R[:, 2, :],  # y_cam 对应原来的 row2，再取反
                R[:, 0, :],  # z_cam 对应原来的 row0
            ],
            dim=1,
        )
        # 2. 计算平移向量 t = -R @ pos
        # 将 pos 扩展为 (num_envs, 3, 1) 以便矩阵乘法
        pos_vec = self.pos.unsqueeze(-1)  # shape: (num_envs, 3, 1)
        t = -torch.bmm(R, pos_vec).squeeze(-1)  # shape: (num_envs, 3)

        # 3. 组装齐次变换矩阵 (num_envs x 4 x 4)
        extrinsics = torch.eye(4, device=self.quat_world.device, dtype=self.quat_world.dtype)
        extrinsics = extrinsics.unsqueeze(0).repeat(self.quat_world.shape[0], 1, 1)  # (num_envs, 4, 4)
        extrinsics[:, :3, :3] = R
        extrinsics[:, :3, 3] = t
        # extrinsics[:, 3, :] = tensor([0,0,0,1])  # 已经是默认的 [0,0,0,1] 行
        extrinsics[torch.abs(extrinsics) < 1e-6] = 0.0
        return extrinsics


@dataclass
class TensorState:
    """Tensorized state of the simulation."""

    objects: dict[str, ObjectState]
    """States of all objects."""
    robots: dict[str, RobotState]
    """States of all robots."""
    cameras: dict[str, CameraState]
    """States of all cameras."""
    sensors: dict[str, SensorState]
    """States of all sensors."""


def join_tensor_states(tensor_states: list[TensorState]) -> TensorState:
    """Join a list of tensor states with num_envs = 1 into a single tensor state."""
    rst = TensorState(objects={}, robots={}, cameras={}, sensors={})

    if not tensor_states:
        return rst

    # Get all unique keys from each category
    all_object_keys = set()
    all_robot_keys = set()
    all_camera_keys = set()
    all_sensor_keys = set()

    for state in tensor_states:
        all_object_keys.update(state.objects.keys())
        all_robot_keys.update(state.robots.keys())
        all_camera_keys.update(state.cameras.keys())
        all_sensor_keys.update(state.sensors.keys())

    # Join objects
    for key in all_object_keys:
        object_states = [state.objects[key] for state in tensor_states if key in state.objects]
        if object_states:
            rst.objects[key] = ObjectState(
                root_state=torch.cat([obj.root_state for obj in object_states], dim=0),
                body_names=object_states[0].body_names,
                body_state=torch.cat([obj.body_state for obj in object_states], dim=0)
                if object_states[0].body_state is not None
                else None,
                joint_pos=torch.cat([obj.joint_pos for obj in object_states], dim=0)
                if object_states[0].joint_pos is not None
                else None,
                joint_vel=torch.cat([obj.joint_vel for obj in object_states], dim=0)
                if object_states[0].joint_vel is not None
                else None,
            )

    # Join robots
    for key in all_robot_keys:
        robot_states = [state.robots[key] for state in tensor_states if key in state.robots]
        if robot_states:
            rst.robots[key] = RobotState(
                root_state=torch.cat([robot.root_state for robot in robot_states], dim=0),
                body_names=robot_states[0].body_names,
                body_state=torch.cat([robot.body_state for robot in robot_states], dim=0),
                joint_pos=torch.cat([robot.joint_pos for robot in robot_states], dim=0),
                joint_vel=torch.cat([robot.joint_vel for robot in robot_states], dim=0),
                joint_pos_target=torch.cat([robot.joint_pos_target for robot in robot_states], dim=0)
                if robot_states[0].joint_pos_target is not None
                else None,
                joint_vel_target=torch.cat([robot.joint_vel_target for robot in robot_states], dim=0)
                if robot_states[0].joint_vel_target is not None
                else None,
                joint_effort_target=torch.cat([robot.joint_effort_target for robot in robot_states], dim=0)
                if robot_states[0].joint_effort_target is not None
                else None,
            )

    # Join cameras
    for key in all_camera_keys:
        camera_states = [state.cameras[key] for state in tensor_states if key in state.cameras]
        if camera_states:
            rst.cameras[key] = CameraState(
                rgb=torch.cat([cam.rgb for cam in camera_states], dim=0) if camera_states[0].rgb is not None else None,
                depth=torch.cat([cam.depth for cam in camera_states], dim=0)
                if camera_states[0].depth is not None
                else None,
                pos=torch.cat([cam.pos for cam in camera_states], dim=0) if camera_states[0].pos is not None else None,
                quat_world=torch.cat([cam.quat_world for cam in camera_states], dim=0)
                if camera_states[0].quat_world is not None
                else None,
                intrinsics=torch.cat([cam.intrinsics for cam in camera_states], dim=0)
                if camera_states[0].intrinsics is not None
                else None,
            )

    # Join sensors (assuming similar structure to objects)
    for key in all_sensor_keys:
        sensor_states = [state.sensors[key] for state in tensor_states if key in state.sensors]
        if sensor_states:
            # Note: SensorState structure is not defined, so this is a placeholder
            rst.sensors[key] = sensor_states[0]  # This would need to be implemented based on SensorState structure

    return rst


def _dof_tensor_to_dict(dof_tensor: torch.Tensor, joint_names: list[str]) -> dict[str, float]:
    """Convert a DOF tensor to a dictionary of joint positions."""
    joint_names = sorted(joint_names)
    return {jn: dof_tensor[i].item() for i, jn in enumerate(joint_names)}


def _body_tensor_to_dict(body_tensor: torch.Tensor, body_names: list[str]) -> dict[str, float]:
    """Convert a body tensor to a dictionary of body positions."""
    body_names = sorted(body_names)
    return {
        bn: {
            "pos": body_tensor[i][:3].cpu(),
            "rot": body_tensor[i][3:7].cpu(),
            "vel": body_tensor[i][7:10].cpu(),
            "ang_vel": body_tensor[i][10:13].cpu(),
        }
        for i, bn in enumerate(body_names)
    }


def state_tensor_to_nested(handler: BaseSimHandler, tensor_state: TensorState) -> list[EnvState]:
    """Convert a tensor state to a list of env states. All the tensors will be converted to cpu for compatibility."""
    log.warning(
        "Users please ignore this message, we are working on it. For developers: You are using the very inefficient function to convert the tensorized states to old nested states. Please consider not using this function and optimize your code when number of environments is large."
    )

    num_envs = next(iter(chain(tensor_state.objects.values(), tensor_state.robots.values()))).root_state.shape[0]
    env_states = []
    for env_id in range(num_envs):
        object_states = {}
        for obj_name, obj_state in tensor_state.objects.items():
            object_states[obj_name] = {
                "pos": obj_state.root_state[env_id, :3].cpu(),
                "rot": obj_state.root_state[env_id, 3:7].cpu(),
                "vel": obj_state.root_state[env_id, 7:10].cpu(),
                "ang_vel": obj_state.root_state[env_id, 10:13].cpu(),
            }
            if obj_state.body_state is not None:
                bns = handler.get_body_names(obj_name)
                object_states[obj_name]["body"] = _body_tensor_to_dict(obj_state.body_state[env_id], bns)
            if obj_state.joint_pos is not None:
                jns = handler.get_joint_names(obj_name)
                object_states[obj_name]["dof_pos"] = _dof_tensor_to_dict(obj_state.joint_pos[env_id], jns)
            if obj_state.joint_vel is not None:
                jns = handler.get_joint_names(obj_name)
                object_states[obj_name]["dof_vel"] = _dof_tensor_to_dict(obj_state.joint_vel[env_id], jns)

        robot_states = {}
        for robot_name, robot_state in tensor_state.robots.items():
            jns = handler.get_joint_names(robot_name)
            robot_states[robot_name] = {
                "pos": robot_state.root_state[env_id, :3].cpu(),
                "rot": robot_state.root_state[env_id, 3:7].cpu(),
                "vel": robot_state.root_state[env_id, 7:10].cpu(),
                "ang_vel": robot_state.root_state[env_id, 10:13].cpu(),
            }
            robot_states[robot_name]["dof_pos"] = _dof_tensor_to_dict(robot_state.joint_pos[env_id], jns)
            robot_states[robot_name]["dof_vel"] = _dof_tensor_to_dict(robot_state.joint_vel[env_id], jns)
            robot_states[robot_name]["dof_pos_target"] = (
                _dof_tensor_to_dict(robot_state.joint_pos_target[env_id], jns)
                if robot_state.joint_pos_target is not None
                else None
            )
            robot_states[robot_name]["dof_vel_target"] = (
                _dof_tensor_to_dict(robot_state.joint_vel_target[env_id], jns)
                if robot_state.joint_vel_target is not None
                else None
            )
            robot_states[robot_name]["dof_torque"] = (
                _dof_tensor_to_dict(robot_state.joint_effort_target[env_id], jns)
                if robot_state.joint_effort_target is not None
                else None
            )
            if robot_state.body_state is not None:
                bns = handler.get_body_names(robot_name)
                robot_states[robot_name]["body"] = _body_tensor_to_dict(robot_state.body_state[env_id], bns)

        camera_states = {}
        for camera_name, camera_state in tensor_state.cameras.items():
            camera_states[camera_name] = {
                "rgb": camera_state.rgb[env_id].cpu(),
                "depth": camera_state.depth[env_id].cpu(),
                "pos": camera_state.pos[env_id].cpu(),
                "quat_world": camera_state.quat_world[env_id].cpu(),
                "cam_extr": camera_state.extrinsics[env_id].cpu(),
                "cam_intr": camera_state.intrinsics[env_id].cpu(),
            }

        sensor_states = {}
        for sensor_name, sensor_state in tensor_state.sensors.items():
            sensor_states[sensor_name] = {
                "force": sensor_state.force[env_id].cpu(),
            }

        env_state = {
            "objects": object_states,
            "robots": robot_states,
            "cameras": camera_states,
            "sensors": sensor_states,
        }
        env_states.append(env_state)

    return env_states


def _alloc_state_tensors(n_env: int, n_body: int | None = None, n_jnt: int | None = None, device="gpu"):
    root = torch.zeros((n_env, 13), device=device)

    n_body = n_body or 0
    body = torch.zeros((n_env, n_body, 13), device=device) if n_body else None

    n_jnt = n_jnt or 0
    jpos = torch.zeros((n_env, n_jnt), device=device) if n_jnt else None
    jvel = torch.zeros_like(jpos) if jpos is not None else None
    return root, body, jpos, jvel


def list_state_to_tensor(
    handler: BaseSimHandler,
    env_states: list[dict],
    device: torch.device | str = "cpu",
) -> TensorState:
    """Convert nested python list-states to a batched TensorState."""
    obj_names = sorted({n for es in env_states for n in es["objects"].keys()})
    robot_names = sorted({n for es in env_states for n in es["robots"].keys()})
    cam_names = sorted({n for es in env_states if "cameras" in es for n in es["cameras"].keys()})

    n_env = len(env_states)
    dev = device

    objects: dict[str, ObjectState] = {}
    robots: dict[str, RobotState] = {}
    cameras: dict[str, CameraState] = {}

    # -------- objects --------------------------------------------------
    for name in obj_names:
        bnames = handler.get_body_names(name)
        jnames = handler.get_joint_names(name)

        root, body, jpos, jvel = _alloc_state_tensors(n_env, len(bnames) or None, len(jnames) or None, dev)

        for e, es in enumerate(env_states):
            if name not in es["objects"]:
                continue
            s = es["objects"][name]

            vel = s.get("vel", torch.zeros(3, device=dev))
            ang_vel = s.get("ang_vel", torch.zeros(3, device=dev))

            root[e, :3] = s["pos"]
            root[e, 3:7] = s["rot"]
            root[e, 7:10] = vel
            root[e, 10:13] = ang_vel

            if body is not None and "body" in s:
                for i, bn in enumerate(sorted(bnames)):
                    if bn not in s["body"]:
                        continue
                    bi = s["body"][bn]
                    body[e, i, :3], body[e, i, 3:7] = bi["pos"], bi["rot"]
                    body[e, i, 7:10], body[e, i, 10:13] = bi["vel"], bi["ang_vel"]

            if jpos is not None and "dof_pos" in s:
                for i, jn in enumerate(sorted(jnames)):
                    if jn in s["dof_pos"]:
                        jpos[e, i] = s["dof_pos"][jn]
            if jvel is not None and "dof_vel" in s:
                for i, jn in enumerate(sorted(jnames)):
                    if jn in s["dof_vel"]:
                        jvel[e, i] = s["dof_vel"][jn]

        objects[name] = ObjectState(root_state=root, body_state=body, joint_pos=jpos, joint_vel=jvel)

    # -------- robots ---------------------------------------------------
    for name in robot_names:
        jnames = handler.get_joint_names(name)
        bnames = handler.get_body_names(name)

        root, body, jpos, jvel = _alloc_state_tensors(n_env, len(bnames) or None, len(jnames) or None, dev)
        jpos_t, jvel_t, jeff_t = (
            torch.zeros_like(jpos) if jpos is not None else None,
            torch.zeros_like(jvel) if jvel is not None else None,
            torch.zeros_like(jvel) if jvel is not None else None,
        )

        for e, es in enumerate(env_states):
            if name not in es["robots"]:
                continue
            s = es["robots"][name]

            pos = s["pos"]
            rot = s["rot"]
            vel = s.get("vel", torch.zeros(3, device=dev))
            ang_vel = s.get("ang_vel", torch.zeros(3, device=dev))

            root[e, :3] = pos
            root[e, 3:7] = rot
            root[e, 7:10] = vel
            root[e, 10:13] = ang_vel
            for i, jn in enumerate(sorted(jnames)):
                if "dof_pos" in s and jn in s["dof_pos"]:
                    jpos[e, i] = s["dof_pos"][jn]
                if "dof_vel" in s and jn in s["dof_vel"]:
                    jvel[e, i] = s["dof_vel"][jn]
                if "dof_pos_target" in s and jn in s["dof_pos_target"]:
                    jpos_t[e, i] = s["dof_pos_target"][jn]
                if "dof_vel_target" in s and jn in s["dof_vel_target"]:
                    jvel_t[e, i] = s["dof_vel_target"][jn]
                if "dof_torque" in s and jn in s["dof_torque"]:
                    jeff_t[e, i] = s["dof_torque"][jn]

            if body is not None and "body" in s:
                for i, bn in enumerate(sorted(bnames)):
                    if bn not in s["body"]:
                        continue
                    bi = s["body"][bn]
                    body[e, i, :3], body[e, i, 3:7], body[e, i, 7:10], body[e, i, 10:13] = (
                        bi["pos"],
                        bi["rot"],
                        bi["vel"],
                        bi["ang_vel"],
                    )

        robots[name] = RobotState(
            root_state=root,
            body_names=bnames,
            body_state=body,
            joint_pos=jpos,
            joint_vel=jvel,
            joint_pos_target=jpos_t,
            joint_vel_target=jvel_t,
            joint_effort_target=jeff_t,
        )

    # -------- cameras ---------------------------------------------
    for cam in cam_names:
        rgb = torch.stack(
            [es["cameras"][cam]["rgb"] for es in env_states if "cameras" in es and cam in es["cameras"]], dim=0
        ).to(dev)
        depth = torch.stack(
            [es["cameras"][cam]["depth"] for es in env_states if "cameras" in es and cam in es["cameras"]], dim=0
        ).to(dev)
        cameras[cam] = CameraState(rgb=rgb, depth=depth)

    return TensorState(
        objects=objects,
        robots=robots,
        cameras=cameras,
        sensors={},
    )
