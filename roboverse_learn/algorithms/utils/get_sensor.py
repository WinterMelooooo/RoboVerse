from metasim.cfg.sensors import ContactForceSensorCfg

Task_2_Desired_Object = {
    "PickCube": "cube",
    "StackCube": "cube",
    "CloseBox": ("box_base", "box_lid")
}


def get_touch_sensors(task_name, robot_name, task_cfg, set_source_link=False) -> ContactForceSensorCfg:
    links = get_robot_links(robot_name)
    source_link = get_desired_object(task_name, task_cfg=task_cfg) if set_source_link else None
    sensors = []
    for link in links:
        sensor = ContactForceSensorCfg(
            name=f"{robot_name}_{link[-1]}_touch_sensor",
            base_link=link,
            source_link=source_link,
        )
        sensors.append(sensor)
    return sensors


def get_robot_links(robot_name):
    if robot_name.lower() == "Franka".lower():
        return [(robot_name.lower(), "panda_leftfinger"), (robot_name.lower(), "panda_rightfinger")]
    else:
        raise NotImplementedError(f"get_robot_links is not implemented for robot {robot_name}.")


def get_desired_object(task_name, task_cfg):
    objs = task_cfg.objects
    desired_object_name = Task_2_Desired_Object[task_name]
    if isinstance(desired_object_name, tuple):
        return desired_object_name
    filter_objs = [obj for obj in objs if obj.name == desired_object_name]
    assert len(filter_objs) == 1, f"Expected one object with name {desired_object_name}, but found {len(filter_objs)}."
    return filter_objs[0].name
