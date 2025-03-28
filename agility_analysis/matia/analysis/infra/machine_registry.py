import time


def kill_docker_instance(instance_id):
    from jaynes.shell import run

    out, err = run(f"docker kill {instance_id}")
    print(out)


class CSAIL_MACHINES:
    def __init__(self, ):
        self.get_machine_list()
        self.get_instance_list()

    def get_machine_list(self):
        from ml_logger import logger
        logger.configure(prefix="/geyang/csail_machines/csail_machines")

        csv = logger.load_csv("machine_list.csv")
        self.machine_list = csv.sort_values(by="memory.free", ascending=False)
        self.machine_list.reset_index(drop=True, inplace=True)

    def get_instance_list(self):
        from ml_logger import logger
        logger.configure(prefix="/geyang/csail_machines/running_instances")

        self.instance_list = logger.load_csv("instance_list.csv")
        self.instance_list.set_index('id', inplace=True)
        print(self.instance_list)

    def pop(self, order="memory.free"):
        row = self.machine_list.iloc[0]
        self.machine_list = self.machine_list.drop(0)
        self.machine_list.reset_index(drop=True, inplace=True)
        return row['hostname'].split('.')[0], str(row['device_id'])

    def kill_instance(self, instance_id):
        import jaynes

        instance = self.instance_list.iloc[instance_id]
        jaynes.config("ssh", launch=dict(ip=instance['hostname'], block=True))
        jaynes.run(kill_docker_instance, instance_id=instance['id'])


def train(seed=10):
    # from dm_control import suite
    # env = suite.load('cartpole', 'swingup')
    import gym
    import torch

    print(torch)

    env = gym.make("dmc:Cartpole-balance-v1")
    print(env)

    time.sleep(10)
    print(seed)
    print('done')


if __name__ == '__main__':
    import jaynes
    from agility_analysis import instr, RUN

    machines = CSAIL_MACHINES()

    for i in range(50):
        hostname, RUN.CUDA_VISIBLE_DEVICES = machines.pop()
        jaynes.config("visiongpu-docker", launch=dict(ip=hostname))
        thunk = instr(train, seed=i * 100)
        jaynes.run(thunk)

    jaynes.listen()
