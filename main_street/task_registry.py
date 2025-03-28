from typing import Tuple

from cxx.env import VecEnv
from cxx.runners import OnPolicyRunner
from main_street.config import RunArgs
from main_street.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

        print("registering task:", name, "with env_cfg:", env_cfg, "and train_cfg:", train_cfg)

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        # print('WARNING: deprecation warning, should not load config from name.')
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name'

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        from main_street.utils.helpers import update_cfg_from_args, class_to_dict, set_seed, parse_sim_params

        # if no args passed get command line arguments
        if args is None:
            args = RunArgs
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)

        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        from pprint import pprint

        pprint(sim_params)
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless,
        )
        return env, env_cfg

    def make_alg_runner(
        self,
        env,
        name=None,
        args=None,
        train_cfg=None,
        # init_wandb=True,
        load_checkpoint=None,
        # log_root="default",
        **kwargs,
    ) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym command line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example).
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_MAIN_ST>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        from main_street.utils.helpers import update_cfg_from_args, class_to_dict

        # if no args passed get command line arguments
        # if args is None:
        #     args = RunArgs
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")

        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        train_cfg_dict = class_to_dict(train_cfg)

        runner = OnPolicyRunner(
            env,
            train_cfg_dict,
            # log_dir,
            # init_wandb=init_wandb,
            device=args.rl_device,
            **kwargs,
        )   
        if not load_checkpoint:
            return runner, train_cfg

        if load_checkpoint.endswith(".pt"):
            resume_path = load_checkpoint
            runner.load(resume_path, load_from_logger=load_checkpoint is not None)
        else:
            resume_path = f"{load_checkpoint}/checkpoints/model_last.pt"
            runner.load(resume_path, load_from_logger=load_checkpoint is not None)

        if train_cfg.policy.init_noise_std is not None and train_cfg.policy.init_noise_std > 0.0:
            # if not train_cfg.policy.continue_from_last_std:
            print("WARNING: resetting std of the policy. This should NOT happen during evaluation.")
            print("adding init noise std:", train_cfg.policy.init_noise_std)
            runner.alg.actor_critic.reset_std(train_cfg.policy.init_noise_std, 12, device=runner.device)

        return runner, train_cfg

        # if "return_log_dir" in kwargs:
        #     return runner, train_cfg, os.path.dirname(resume_path)
        # else:
        #     return runner, train_cfg


# make global task registry
task_registry = TaskRegistry()
