import asyncio
import time
from pathlib import Path

import lcm
import numpy as np
from vuer.schemas import CoordsMarker, DefaultScene, Urdf, group
from vuer.server import Vuer, VuerSession

from go1_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from go1_gym_deploy.modules.base.state_estimator import JOINT_IDX_MAPPING, BasicStateEstimator

lc = "udpm://239.255.76.67:7667?ttl=255"
lc_obj = lcm.LCM(lc)

ISAAC_DOF_NAMES = [
    'FL_hip_joint',
    'FL_thigh_joint',
    'FL_calf_joint',
    'FR_hip_joint',
    'FR_thigh_joint',
    'FR_calf_joint',
    'RL_hip_joint',
    'RL_thigh_joint',
    'RL_calf_joint',
    'RR_hip_joint',
    'RR_thigh_joint',
    'RR_calf_joint',
]

UNITREE_DOF_NAMES = np.array([ISAAC_DOF_NAMES])[:, JOINT_IDX_MAPPING][0].tolist()

STARTING_POSITION_UNITREE = [-0.3,
                             1.2,
                             -2.721,
                             0.3,
                             1.2,
                             -2.721,
                             -0.3,
                             1.2,
                             -2.721,
                             0.3,
                             1.2,
                             -2.721]


def setup():
    command_for_robot = pd_tau_targets_lcmt()
    joint_pos_target = STARTING_POSITION_UNITREE
    joint_vel_target = np.zeros(12)

    command_for_robot.q_des = joint_pos_target
    command_for_robot.qd_des = joint_vel_target
    command_for_robot.kp = np.zeros(12)
    command_for_robot.kd = np.zeros(12)
    command_for_robot.tau_ff = np.zeros(12)
    command_for_robot.se_contactState = np.zeros(4)
    command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
    command_for_robot.id = 0

    lc_obj.publish("pd_plustau_targets", command_for_robot.encode())


def echo(anchor_leg, stiffness=5, damping=0.6):
    assert anchor_leg in ['FL', 'FR', 'RL', 'RR']

    opposite_leg_index = {
        'FL': 'FR',
        'FR': 'FL',
        'RL': 'RR',
        'RR': 'RL',
    }

    opposite_leg = opposite_leg_index[anchor_leg]

    joint_pos_target = STARTING_POSITION_UNITREE

    anchor_hip = se.get_dof_pos()[ISAAC_DOF_NAMES.index(f'{anchor_leg}_hip_joint')]
    anchor_thigh = se.get_dof_pos()[ISAAC_DOF_NAMES.index(f'{anchor_leg}_thigh_joint')]
    anchor_calf = se.get_dof_pos()[ISAAC_DOF_NAMES.index(f'{anchor_leg}_calf_joint')]

    joint_pos_target[UNITREE_DOF_NAMES.index(f'{opposite_leg}_hip_joint')] = anchor_hip
    joint_pos_target[UNITREE_DOF_NAMES.index(f'{opposite_leg}_thigh_joint')] = anchor_thigh
    joint_pos_target[UNITREE_DOF_NAMES.index(f'{opposite_leg}_calf_joint')] = anchor_calf

    kp = np.zeros(12)
    kp[UNITREE_DOF_NAMES.index(f'{opposite_leg}_hip_joint')] = stiffness
    kp[UNITREE_DOF_NAMES.index(f'{opposite_leg}_thigh_joint')] = stiffness
    kp[UNITREE_DOF_NAMES.index(f'{opposite_leg}_calf_joint')] = stiffness

    kd = np.zeros(12)
    kd[UNITREE_DOF_NAMES.index(f'{opposite_leg}_hip_joint')] = damping
    kd[UNITREE_DOF_NAMES.index(f'{opposite_leg}_thigh_joint')] = damping
    kd[UNITREE_DOF_NAMES.index(f'{opposite_leg}_calf_joint')] = damping

    command_for_robot = pd_tau_targets_lcmt()

    command_for_robot.q_des = joint_pos_target
    command_for_robot.qd_des = np.zeros(12)
    command_for_robot.kp = kp
    command_for_robot.kd = np.zeros(12)
    command_for_robot.tau_ff = np.zeros(12)
    command_for_robot.se_contactState = np.zeros(4)
    command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
    command_for_robot.id = 0
    lc.publish("pd_plustau_targets", command_for_robot.encode())


if __name__ == '__main__':

    setup()

    se = BasicStateEstimator(lc, "cpu")
    se.spin_process()

    app = Vuer(
        static_root=Path(__file__).parent / "/Users/alanyu/urop/parkour/main_street/assets/robots/gabe_go1")

    # app = Vuer(
    #     static_root="/home/escher/mit/parkour/main_street/assets/robots/gabe_go1"
    # )
    print('here')


    @app.spawn
    async def main(session: VuerSession):
        print('wtf')
        session.set @ DefaultScene()
        # session.set @ group(group(group(
        #     Urdf(
        #         src="http://localhost:8012/static/urdf/go1.urdf",
        #         jointValues={name: pos for name, pos in
        #                      zip(ISAAC_DOF_NAMES, np.zeros(12).tolist())},
        #         key="go1",
        #     ),
        #     CoordsMarker(
        #         position=[0, 0, 0],
        #         key="coords",
        #     ),
        #     key="roll",
        #     rotation=[0, 0, 0],
        #     scale=0.4,
        # ),
        #     rotation=[0, 0, 0], key="pitch"))
        # print('huh')
        while True:
            # roll, pitch, yaw = se.euler.cpu().numpy().tolist()
            # print(roll, pitch, yaw)
            roll, pitch, yaw = se.data["euler"][0].tolist()
            droll, dpitch, dyaw = se.data["deuler"][0].tolist()
            dofs = se.data["joint_pos"][0].tolist()
            # print(dofs)
            # print(roll, pitch, yaw)
            session.upsert @ group(group(group(
                Urdf(
                    src="http://localhost:8012/static/urdf/go1.urdf",
                    jointValues={name: pos for name, pos in
                                 zip(UNITREE_DOF_NAMES, dofs)},
                    key="go1",
                ),
                CoordsMarker(
                    position=[0, 0, 0],
                    key="coords",
                ),
                key="roll",
                rotation=[roll, 0, 0],
                scale=0.4,
            ),
                rotation=[0, pitch, 0], key="pitch"), rotation=[0, 0, yaw],
                key="outer")

            await asyncio.sleep(0.005)


    app.run()
