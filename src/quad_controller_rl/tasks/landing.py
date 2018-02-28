import numpy as np
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from gym import spaces
from quad_controller_rl.tasks.base_task import BaseTask


class Landing(BaseTask):
    # Task is to lift off the ground
    __name__ = "Landing"
    def __init__(self):
        cube_size = 300.0
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2, 0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([cube_size / 2, cube_size / 2, cube_size, 1.0, 1.0, 1.0, 1.0]))
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([max_force, max_force, max_force, max_torque, max_torque, max_torque]))
        self.max_duration = 5.0
        self.target_z = 0.1

    def reset(self):
        return Pose(
            position=Point(0.0, 0.0, np.random.normal(10.0, 1.0)),
            orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
        ), Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0)
        )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        state = np.array([
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        done = False
        if -2.0 < linear_acceleration.z < 0:
            reward = 1
        else:
            reward = -abs(self.target_z - pose.position.z) - abs(linear_acceleration.z)
        if pose.position.z <= self.target_z:
            reward += 10.0
            done = True
        if timestamp > self.max_duration:
            reward -= 10.0
            done = True
        action = self.agent.step(state, reward, done)
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)
            return Wrench(force=Vector3(action[0], action[1], action[2]),
                          torque=Vector3(action[3], action[4], action[5])), done
        else:
            return Wrench(), done
