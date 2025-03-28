from go1_gym_deploy.modules.base.state_estimator import BasicStateEstimator


class LucidSimStateEstimator(BasicStateEstimator):
    def get_command(self):
        self.left_stick, self.right_stick = self.data["command"][0, :2], self.data["command"][0, 2:]

        cmd_x = 0.8
        cmd_y = 0
        cmd_yaw = -0.5 * self.right_stick[0]

        return [cmd_x, cmd_y, cmd_yaw]


if __name__ == "__main__":
    import lcm

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    se = LucidSimStateEstimator(lc, device="cpu")
    print("created")
    se.poll()
