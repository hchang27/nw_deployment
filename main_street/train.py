from main_street.config import RunArgs


def train(**deps):
    from isaacgym import gymapi

    assert gymapi, "force import first to avoid torch error."
    from ml_logger import logger

    from main_street import MAIN_ST_ENVS_DIR
    from main_street.config import RunArgs
    from main_street.envs import task_registry
    from main_street.envs.go1.go1_config import Go1RoughCfg

    RunArgs._update(deps)

    RunArgs.headless = True

    RunArgs.validate_args()
    #
    # try:  # RUN.prefix is a template, will raise error.
    #     RUN.update(deps)
    #     logger.configure(RUN.prefix)
    # except KeyError:
    #     pass
    #
    print("Dashboard:", logger.get_dash_url())

    logger.job_started(run=vars(RunArgs))

    # Go1RoughCfg.domain_rand._update(deps)
    Go1RoughCfg.commands.max_ranges._update(deps)

    # print("Domain rand:", Go1RoughCfg.domain_rand.action_curr_step, Go1RoughCfg.domain_rand.action_delay_view)
    print("Max ranges:", Go1RoughCfg.commands.max_ranges.lin_vel_x)

    # fmt: off
    logger.log_text("""
        keys:
        - task.name
        - algo.name
        charts:
        - yKeys: ["Train/mean_reward/mean"]
          xKey: iterations
        - glob: "**/*.mp4"
          type: video
        """, ".charts.yml", True, True)
    # fmt: on

    if RunArgs.debug:
        print("deprecate debug flag", RunArgs.debug)
        # mode = "disabled"
        RunArgs.rows = 8
        RunArgs.cols = 6
        RunArgs.num_envs = 16

    logger.upload_file(MAIN_ST_ENVS_DIR + "/go1/go1_config.py")
    logger.upload_file(MAIN_ST_ENVS_DIR + "/base/legged_robot_config.py")
    logger.upload_file(MAIN_ST_ENVS_DIR + "/base/legged_robot.py")

    env, env_cfg = task_registry.make_env(name=RunArgs.task, args=RunArgs)

    # todo: when loading a checkpoint, also set the environment steps

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=RunArgs.task,
        args=RunArgs,
        load_checkpoint=RunArgs.load_checkpoint,
    )
    # remove this if you want to resume previous checkpoint
    ppo_runner.curr_step = 0
    # this is to set the environment counter via the runner step. However,
    # ideally we just remove the counter from the environment.
    env.global_counter = ppo_runner.curr_step * 24

    ppo_runner.learn(num_steps=train_cfg.runner.max_iterations + 1)


if __name__ == "__main__":
    RunArgs.debug = True

    RunArgs.headless = True
    RunArgs.max_iterations = 25_000
    RunArgs.task = "go1"

    # RunArgs.resume = True
    # RunArgs.use_camera = True
    # RunArgs.load_checkpoint = "/lucidsim/lucidsim/parkour/baselines/launch_gabe_go1/go1/200"
    # RunArgs.delay = True

    # RunArgs.debug = True

    RunArgs.seed = 300

    train()

    exit()

    def depth_student_example():
        from agility_analysis import RUN
        from main_street.config import RunArgs

        RunArgs.task = "go1"
        RunArgs.seed = 900

        RunArgs.exptid = "000-00-debug"
        RunArgs.max_iterations = 25_000

        RunArgs.use_camera = True
        RunArgs.headless = False
        # RunArgs.resume = True
        # RunArgs.delay = True

        RunArgs.num_envs = 192 * 2

        RunArgs.load_checkpoint = "/lucid-sim/lucid-sim/baselines/launch_gains/2024-03-19/03.17.23/go1/300/20/0.5"

        RunArgs.device = "cuda:1"

        RunArgs.debug = True

        # RunArgs.validate_args()

        RUN.job_name = f"{RunArgs.task}/{RunArgs.seed}"

        train()

    depth_student_example()
