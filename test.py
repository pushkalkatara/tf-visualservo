#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


r"""
This is an example of how to add new actions to habitat-api


We will use the strafe action outline in the habitat_sim example
"""

import attr
import numpy as np
import utils.ros_utils as ros_utils
import cv2
import model

import tensorflow as tf
import tensorflow.contrib.slim as slim

import habitat
import habitat_sim
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat.tasks.nav.nav import SimulatorTaskAction

path = '/home/pushkalkatara/rrc/weight/model.ckpt-0'

@attr.s(auto_attribs=True, slots=True)
class NoisyStrafeActuationSpec:
    move_amount: float
    # Classic strafing is to move perpendicular (90 deg) to the forward direction
    strafe_angle: float = 90.0
    noise_amount: float = 0.05


def _strafe_impl(
    scene_node: habitat_sim.SceneNode,
    move_amount: float,
    strafe_angle: float,
    noise_amount: float,
):
    forward_ax = (
        np.array(scene_node.absolute_transformation().rotation_scaling())
        @ habitat_sim.geo.FRONT
    )
    strafe_angle = np.deg2rad(strafe_angle)
    strafe_angle = np.random.uniform(
        (1 - noise_amount) * strafe_angle, (1 + noise_amount) * strafe_angle
    )

    rotation = habitat_sim.utils.quat_from_angle_axis(
        np.deg2rad(strafe_angle), habitat_sim.geo.UP
    )
    move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)

    move_amount = np.random.uniform(
        (1 - noise_amount) * move_amount, (1 + noise_amount) * move_amount
    )
    scene_node.translate_local(move_ax * move_amount)


@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyStrafeLeft(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: NoisyStrafeActuationSpec,
    ):
        print(f"strafing left with noise_amount={actuation_spec.noise_amount}")
        _strafe_impl(
            scene_node,
            actuation_spec.move_amount,
            actuation_spec.strafe_angle,
            actuation_spec.noise_amount,
        )


@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyStrafeRight(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: NoisyStrafeActuationSpec,
    ):
        print(
            f"strafing right with noise_amount={actuation_spec.noise_amount}"
        )
        _strafe_impl(
            scene_node,
            actuation_spec.move_amount,
            -actuation_spec.strafe_angle,
            actuation_spec.noise_amount,
        )


@habitat.registry.register_action_space_configuration
class NoNoiseStrafe(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.STRAFE_LEFT] = habitat_sim.ActionSpec(
            "noisy_strafe_left",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.0),
        )
        config[HabitatSimActions.STRAFE_RIGHT] = habitat_sim.ActionSpec(
            "noisy_strafe_right",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.0),
        )

        return config


@habitat.registry.register_action_space_configuration
class NoiseStrafe(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.STRAFE_LEFT] = habitat_sim.ActionSpec(
            "noisy_strafe_left",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.05),
        )
        config[HabitatSimActions.STRAFE_RIGHT] = habitat_sim.ActionSpec(
            "noisy_strafe_right",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.05),
        )

        return config


@habitat.registry.register_task_action
class StrafeLeft(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "strafe_left"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.STRAFE_LEFT)


@habitat.registry.register_task_action
class StrafeRight(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "strafe_right"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.STRAFE_RIGHT)

def calc_error(img1, img2, curr_pose):
    m = model.Model()

    posexyz, poseq = None, None
    error = 0
    prev = {'x':0, 'y':0, 'z':0}
    cmd = {'x':None, 'y':None, 'z':None}

    with tf.Session() as sess:
        m.saver.restore(sess, path)
        posexyz, poseq = sess.run([m.fc_pose_xyz, m.fc_pose_wpqr],
                                            feed_dict={m.x1:img1.reshape(1,384,512,3),
                                            m.x2:img2.reshape(1,384,512,3)})

    lmda1 = 0.7
    lmda2 = 0.07
    thresh = 0.1

    quaternion = (poseq[0][1], poseq[0][2], poseq[0][3], poseq[0][0])
    q_inv = np.conjugate(quaternion)

    r_mat = ros_utils.quaternion_matrix(q_inv)
    trans = -posexyz[0]/1000

    angle, direction, point = ros_utils.rotation_from_matrix(r_mat)
    tvel = -1 * lmda1 * np.dot(np.transpose(r_mat[0:3, 0:3]), trans)
    q_vel = ros_utils.quaternion_about_axis(-1 * lmda2 * angle, direction)
    euler = ros_utils.euler_from_quaternion(q_vel)


    prev['x'] = curr_pose['x'] - prev['x']
    prev['y'] = curr_pose['y'] - prev['y']
    prev['z'] = curr_pose['z'] - prev['z']

    cmd['x'] = np.sign(tvel[2])*min(abs(tvel[2]),thresh)
    cmd['y'] = -np.sign(tvel[0])*min(abs(tvel[0]),thresh)
    cmd['z'] = -np.sign(tvel[1])*min(abs(tvel[1]),thresh)


    cv2.imshow('Final Image', img2)
    cv2.imwrite('Final Image.jpg', img2)
    cv2.waitKey(30)
    cv2.imshow('Current Image', img1)
    cv2.imwrite('Current Image.jpg', img1)
    cv2.waitKey(30)


    error = abs(prev['x'] - cmd['x']) + \
                abs(prev['y'] - cmd['y']) + \
                abs(prev['z'] - cmd['z'])

    thr = 5e-2
    print("Error: ", error)
    if error >= thr:
        print("Position: ", prev)
        print("Error: ", error)
        print("Linear: ", cmd)


def main():
    HabitatSimActions.extend_action_space("STRAFE_LEFT")
    HabitatSimActions.extend_action_space("STRAFE_RIGHT")

    config = habitat.get_config(config_paths="habitat-configs/pointnav.yaml")
    config.defrost()

    config.TASK.POSSIBLE_ACTIONS = config.TASK.POSSIBLE_ACTIONS + [
        "STRAFE_LEFT",
        "STRAFE_RIGHT",
    ]
    config.TASK.ACTIONS.STRAFE_LEFT = habitat.config.Config()
    config.TASK.ACTIONS.STRAFE_LEFT.TYPE = "StrafeLeft"
    config.TASK.ACTIONS.STRAFE_RIGHT = habitat.config.Config()
    config.TASK.ACTIONS.STRAFE_RIGHT.TYPE = "StrafeRight"
    config.SIMULATOR.ACTION_SPACE_CONFIG = "NoNoiseStrafe"
    config.freeze()

    env = habitat.Env(config=config)
    o1 = env.reset()
    print(env._sim.get_agent_state())
    curr_pose = env._sim.get_agent_state().position
    curr_pose = {'x':curr_pose[0], 'y':curr_pose[1], 'z':curr_pose[2]}
    img1 = cv2.resize(o1['rgb'], (512,384), interpolation = cv2.INTER_LINEAR)

    env.step("STRAFE_LEFT")
    env.step("STRAFE_LEFT")
    env.step("STRAFE_LEFT")
    env.step("STRAFE_LEFT")
    o2 = env.step("STRAFE_LEFT")

    print(env._sim.get_agent_state())
    img2 = cv2.resize(o2['rgb'], (512,384), interpolation = cv2.INTER_LINEAR)

    calc_error(img1, img2, curr_pose)
    env.step("STRAFE_RIGHT")
    env.step("STRAFE_RIGHT")
    env.step("STRAFE_RIGHT")
    o3 = env.step("STRAFE_RIGHT")
    curr_pose = env._sim.get_agent_state().position
    curr_pose = {'x':curr_pose[0], 'y':curr_pose[1], 'z':curr_pose[2]}

    img3 = cv2.resize(o3['rgb'], (512,384), interpolation = cv2.INTER_LINEAR)
    calc_error(img1, img3, curr_pose)






    env.close()

if __name__ == "__main__":
    main()
