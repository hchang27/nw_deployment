import subprocess
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple

import lcm
import numpy as np
import select
import torch

from go1_gym_deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from go1_gym_deploy.lcm_types.rc_command_lcmt import rc_command_lcmt
from go1_gym_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt

JOINT_IDX_MAPPING = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
CONTACT_IDX_MAPPING = [1, 0, 3, 2]

SE_LOG_SIZES = {
    "command": (1, 4),
    "euler": (1, 3),
    "deuler": (1, 3),
    "joint_pos": (1, 12),
    "joint_vel": (1, 12),
    "contact_state": (1, 4),
    "trigger": (2,),
}

from go1_gym_deploy.utils import get_rotation_matrix_from_rpy


def get_rpy_from_quaternion_t(q):
    w, x, y, z = q
    r = torch.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    p = torch.arcsin(2 * (w * y - z * x))
    y = torch.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return torch.tensor([r, p, y], device=q.device)


def get_rpy_from_quaternion(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    p = np.arcsin(2 * (w * y - z * x))
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([r, p, y])


class BasicStateEstimator:
    def __init__(self, lc: lcm.LCM, device: str):
        # reverse legs, invertible with itself

        self.device = device

        self.lc = lc

        self.joint_pos = torch.zeros(12, device=self.device)
        self.joint_vel = torch.zeros(12, device=self.device)
        self.tau_est = torch.zeros(12, device=self.device)
        self.euler = torch.zeros(3, device=self.device)
        self.R = torch.eye(3, device=self.device)
        self.buf_idx = 0

        self.smoothing_length = 1
        self.deuler_history = torch.zeros((self.smoothing_length, 3), device=self.device)
        self.dt_history = torch.zeros((self.smoothing_length, 1), device=self.device)
        self.euler_prev = torch.zeros(3, device=self.device)
        self.timuprev = time.time()

        self.deuler = torch.zeros(3, device=self.device)

        self.body_ang_vel = torch.zeros(3, device=self.device)
        self.smoothing_ratio = 0.9

        self.contact_state = torch.ones(4, device=self.device)

        self.mode = 0
        self.ctrlmode_left = 0
        self.ctrlmode_right = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0

        # Triggers, need to commmunicate via self.data

        self.right_lower_right_switch = 0
        self.right_lower_right_switch_pressed = 0

        # keys: command, euler, deuler, joint_pos, joint_vel, contact_state,

        self.data = dict()
        self.buffers = []

        self.init_time = time.time()
        self.received_first_legdata = False

    def __enter__(self):
        # self.lc: lcm.LCM = lcm.LCM(self.lc_addr)

        imu = self.lc.subscribe(
            "state_estimator_data",
            self._imu_cb,
        )
        leg = self.lc.subscribe(
            "leg_control_data",
            self._legdata_cb,
        )
        cmd = self.lc.subscribe(
            "rc_command",
            self._rc_command_cb,
        )

    def __exit__(self, *args):
        print("closing state estimator! ")
        self.release()

    open = __enter__
    close = __exit__

    def release(self):
        for b in self.buffers:
            b.unlink()

        self.data.clear()

    def get_body_angular_vel(self):
        return self.deuler

    def get_gravity_vector(self):
        grav = torch.dot(self.R.T, torch.tensor([0, 0, -1], device=self.device))
        return grav

    def get_contact_state(self):
        return self.contact_state[CONTACT_IDX_MAPPING]

    def get_rpy(self):
        return self.euler

    def get_command(self):
        # if self.left_upper_switch_pressed:
        #     self.ctrlmode_left = (self.ctrlmode_left + 1) % 3
        #     self.left_upper_switch_pressed = False
        # if self.right_upper_switch_pressed:
        #     self.ctrlmode_right = (self.ctrlmode_right + 1) % 3
        #     self.right_upper_switch_pressed = False

        self.left_stick, self.right_stick = self.data["command"][:2], self.data["command"][2:]

        cmd_x = max(self.left_stick[1], 0.0)
        cmd_y = self.left_stick[0]
        cmd_yaw = -1.0 * self.right_stick[0]

        return [cmd_x, cmd_y, cmd_yaw]

    def get_buttons(self):
        return np.array(
            [
                self.left_lower_left_switch,
                self.left_upper_switch,
                self.right_lower_right_switch,
                self.right_upper_switch,
            ]
        )

    def get_dof_pos(self):
        """return Joint angles in the isaacgym order"""
        return self.joint_pos[JOINT_IDX_MAPPING]

    def get_dof_vel(self):
        return self.joint_vel[JOINT_IDX_MAPPING]

    def get_tau_est(self):
        return self.tau_est[JOINT_IDX_MAPPING]

    def get_yaw(self):
        return self.euler[2]

    def _legdata_cb(self, channel, data):
        if not self.received_first_legdata:
            self.received_first_legdata = True
            print(f"First legdata: {time.time() - self.init_time}")

        msg = leg_control_data_lcmt.decode(data)
        self.joint_pos = torch.tensor(msg.q, device=self.device)
        self.joint_vel = torch.tensor(msg.qd, device=self.device)
        self.tau_est = torch.tensor(msg.tau_est, device=self.device)

        self._update_key("joint_pos", self.joint_pos)
        self._update_key("joint_vel", self.joint_vel)

        # print('leg', self.joint_pos)

    def _imu_cb(self, channel, data):
        msg = state_estimator_lcmt.decode(data)

        self.euler = torch.tensor(msg.rpy, device=self.device)
        # print('recieved imu data', self.euler)

        self.deuler[:3] = torch.tensor(msg.omegaBody, device=self.device)
        # print('recieved imu data', msg.omegaBody)

        self.R = get_rotation_matrix_from_rpy(self.euler)

        self.contact_state = 1.0 * (torch.tensor(msg.contact_estimate, device=self.device) > 2)

        self.deuler_history[self.buf_idx % self.smoothing_length, :] = self.euler - self.euler_prev
        self.dt_history[self.buf_idx % self.smoothing_length] = time.time() - self.timuprev
        # print('updated dt_history')
        # print(self.get_body_angular_vel(self.body_ang_vel))

        self.timuprev = time.time()

        self.buf_idx += 1
        self.euler_prev = torch.tensor(msg.rpy, device=self.device)

        self._update_key("euler", self.euler)
        # print('euler', self.euler)
        self._update_key("deuler", self.deuler)
        self._update_key("contact_state", self.contact_state)

    def _sensor_cb(self, channel, data):
        pass

    def _rc_command_cb(self, channel, data):
        msg = rc_command_lcmt.decode(data)

        self.left_upper_switch_pressed = (msg.left_upper_switch and not self.left_upper_switch) or self.left_upper_switch_pressed
        self.left_lower_left_switch_pressed = (
            msg.left_lower_left_switch and not self.left_lower_left_switch
        ) or self.left_lower_left_switch_pressed
        self.left_lower_right_switch_pressed = (
            msg.left_lower_right_switch and not self.left_lower_right_switch
        ) or self.left_lower_right_switch_pressed
        self.right_upper_switch_pressed = (msg.right_upper_switch and not self.right_upper_switch) or self.right_upper_switch_pressed
        self.right_lower_left_switch_pressed = (
            msg.right_lower_left_switch and not self.right_lower_left_switch
        ) or self.right_lower_left_switch_pressed

        # self.right_lower_right_switch_pressed = \
        #     (msg.right_lower_right_switch and not self.right_lower_right_switch) \
        #     or self.right_lower_right_switch_pressed

        self.right_lower_right_switch_pressed = msg.right_lower_right_switch

        # print("right_lower_right_switch", msg.right_lower_right_switch)
        self.mode = msg.mode
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch

        self._update_key("command", np.array([*self.left_stick, *self.right_stick]))
        self._update_key("trigger", np.array([self.right_lower_right_switch, self.right_lower_right_switch_pressed]))

    def _update_key(self, key, value):
        if key not in self.data:
            self.data[key] = value
        else:
            np.copyto(self.data[key], value)

    @staticmethod
    def create_buffer(shape: Tuple[int], dtype=np.uint8, name="zed"):
        size = np.prod(shape) * np.dtype(dtype).itemsize
        try:
            shm = SharedMemory(create=True, size=size, name=name)
            print(
                "created a shared memory",
                name,
                "with size",
                size,
                "from item size",
                np.dtype(dtype).itemsize,
                "dtype",
                dtype,
                "shape",
                shape,
            )
        except FileExistsError:
            shm = SharedMemory(name=name, size=size)
            print(
                "connected to a shared memory",
                name,
                "with size",
                size,
                "from item size",
                np.dtype(dtype).itemsize,
                "dtype",
                dtype,
                "shape",
                shape,
            )

        # shm.close()  # The server will open this again
        return shm

    def share_buffers(self):
        for key, shape in SE_LOG_SIZES.items():
            shm = self.create_buffer(shape, dtype=np.float32, name=f"se-{key}")
            self.buffers.append(shm)

            np_buffer = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
            self.data[key] = np_buffer

        # set the triggers
        self._update_key("trigger", np.array([0.0, 0.0]))

    def poll(self, cb=None):
        try:
            with self:
                while True:
                    timeout = 0.01
                    rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                    if rfds:
                        self.lc.handle()
                    else:
                        # print('timeout')
                        continue
        except KeyboardInterrupt:
            exit()

    def spin(self):
        # set to be a daemon thread so that it closes when the main thread exists. - Ge
        self.run_thread = threading.Thread(target=self.poll, daemon=True)
        self.run_thread.start()

    def spin_process(self):
        import sys
        print('yo')
        self.share_buffers()

        print("Here", self.data)

        args = [sys.executable, __file__]
        p = subprocess.Popen(args)
        return p


def entry_point():
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    se = BasicStateEstimator(lc, "cpu")
    se.share_buffers()
    se.poll()


if __name__ == "__main__":
    entry_point()
