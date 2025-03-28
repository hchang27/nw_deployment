import mujoco
import mujoco.viewer
import numpy as np

from go1_gym_deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from go1_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from go1_gym_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt
from go1_gym_deploy.modules.base.state_estimator import get_rpy_from_quaternion
from main_street import MAIN_ST_ROOT_DIR


class Go1MujocoLCMEmulator:
    FOOT_NAMES = ["FR", "FL", "RR", "RL"]

    def __init__(self, fix_base=False, joystick_callback=None, viewer=True):
        self.joint_idxs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

        # Start MuJoCo
        self.go1 = mujoco.MjModel.from_xml_path(f"{MAIN_ST_ROOT_DIR}/assets/robots/gabe_go1/xml/go1.xml")
        self.go1_data = mujoco.MjData(self.go1)

        self.go1.opt.timestep = 0.002
        self.go1_data.qpos = [
            0,
            0,
            0.19,  # pos
            0,
            0,
            0,
            0,  # quat
            0,
            1.2,
            -2.5,  # FR
            0,
            1.2,
            -2.5,  # FL
            0,
            1.2,
            -2.5,  # RR
            0,
            1.2,
            -2.5,  # RL
        ]

        self.viewer = None
        if viewer:
            if joystick_callback is not None:
                self.viewer = mujoco.viewer.launch_passive(self.go1, self.go1_data, key_callback=joystick_callback)
            else:
                self.viewer = mujoco.viewer.launch_passive(self.go1, self.go1_data)

            self.viewer._hide_overlay = True
            self.viewer._render_every_frame = True

        self.state_estimator_cb = None
        self.legdata_state_cb = None
        self.battery_state_cb = None
        self.rc_command_cb = None

        self.geom_ids = {}
        for foot_name in self.FOOT_NAMES:
            geom_id = self.go1_data.geom(foot_name).id
            self.geom_ids[geom_id] = foot_name

    def subscribe(self, channel, callback):
        if channel == "state_estimator_data":
            self.state_estimator_cb = callback
            self.send_imudata()
        elif channel == "leg_control_data":
            self.legdata_state_cb = callback
            self.send_legdata()
        elif channel == "battery_state":
            self.battery_state_cb = callback
        elif channel == "rc_command":
            self.rc_command_cb = callback

    def publish(self, channel, data):
        msg = pd_tau_targets_lcmt.decode(data)

        joint_pos_targets = np.array(msg.q_des)  # [self.joint_idxs]
        p_gains = msg.kp
        d_gains = msg.kd

        self.step(joint_pos_targets, p_gains, d_gains)

        self.send_imudata()
        self.send_legdata()

    def send_imudata(self):
        imu_msg = state_estimator_lcmt()
        imu_msg.rpy = self.get_rpy()
        imu_msg.p = self.get_pos()

        contact_forces = self._get_contacts()

        imu_msg.contact_estimate = [np.linalg.norm(contact_forces[x]) for x in contact_forces]
        self.state_estimator_cb("state_estimator_data", imu_msg.encode())

    def send_legdata(self):
        legdata_msg = leg_control_data_lcmt()
        legdata_msg.q = np.array(self.get_joint_pos())
        legdata_msg.qd = np.array(self.get_joint_vel())
        legdata_msg.tau_est = np.array(self.go1_data.ctrl)

        self.legdata_state_cb("leg_control_data", legdata_msg.encode())

    def step(self, joint_pos_targets, p_gains, d_gains):
        # joint_pos_targets_fixed = joint_pos_targets[self.joint_idxs]
        # for i in range(10):
        #     joint_pos = self.get_joint_pos()
        #     joint_vel = self.get_joint_vel()
        #
        #     torques = p_gains * (np.array(joint_pos_targets_fixed) - joint_pos) - (d_gains) * (joint_vel)
        #     self.go1_data.ctrl = torques
        #
        #     mujoco.mj_step(self.go1, self.go1_data)
        self.go1_data.ctrl = np.array(joint_pos_targets)
        for _ in range(4):
            mujoco.mj_step(self.go1, self.go1_data)
            if self.viewer is not None:
                self.viewer.sync()

    def get_joint_pos(self):
        return self.go1_data.qpos[7:]

    def get_joint_vel(self):
        return self.go1_data.qvel[6:]

    def get_rpy(self):
        quat = np.array(self.go1_data.qpos[3:7])
        rpy = get_rpy_from_quaternion(quat)
        return rpy

    def get_pos(self):
        pos = self.go1_data.qpos[:3]
        return pos

    def _get_contacts(self):
        contact_forces = {k: np.zeros(3) for k in self.geom_ids.values()}
        force_buf = np.zeros(6, dtype=np.float64)
        for i in range(len(self.go1_data.contact)):
            contact = self.go1_data.contact[i]

            geom1_id = contact.geom1
            geom2_id = contact.geom2

            geom1_name = self.geom_ids.get(geom1_id, None)
            geom2_name = self.geom_ids.get(geom2_id, None)

            if geom1_name is None and geom2_name is None:
                continue

            mujoco.mj_contactForce(self.go1, self.go1_data, i, force_buf)

            if geom1_name in contact_forces:
                contact_forces[geom1_name] += force_buf[:3]
            if geom2_name in contact_forces:
                contact_forces[geom2_name] += force_buf[:3]

        # Warning: this is in ZYX or something
        # print("Forces", contact_forces)
        return contact_forces


if __name__ == "__main__":
    import time

    env = Go1MujocoLCMEmulator()
    time.sleep(10)
