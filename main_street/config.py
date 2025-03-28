from params_proto import Flag, ParamsProto, Proto


def parse_device_str(device_str):
    # defaults
    device = "cpu"
    device_id = 0

    if device_str == "cpu" or device_str == "cuda":
        device = device_str
        device_id = 0
    else:
        device_args = device_str.split(":")
        assert len(device_args) == 2 and device_args[0] == "cuda", f'Invalid device string "{device_str}"'
        device, device_id_s = device_args
        try:
            device_id = int(device_id_s)
        except ValueError:
            raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
    return device, device_id


class RunArgs(ParamsProto, prefix='run'):
    task = Proto(default="go1", dtype=str, help="Resume training or start testing from a checkpoint. Overrides config file if provided.")

    # this is now deprecated, replaced by init_std_noise.
    # continue_from_last_std = True
    # """This is used only once, and is the same as the resume flag."""

    experiment_name = Proto(dtype=str, help="Name of the experiment to run or load. Overrides config file if provided.")
    run_name = Proto(dtype=str, help="Name of the run. Overrides config file if provided.")
    load_run = Proto(
        dtype=str, help="Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."
    )
    # now deprecated, replaced by load_checkpoint.
    # checkpoint = Proto(
    #     default=-1,
    #     dtype=int,
    #     help="Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
    # )

    # headless = Proto(default=True, help="Force display off at all times")
    horovod = Proto(default=False, help="Use horovod for multi-gpu training")
    rl_device = Proto(default="cuda:0", dtype=str, help="Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)")
    num_envs = Proto(dtype=int, help="Number of environments to create. Overrides config file if provided.")
    seed = Proto(100, help="Random seed. Overrides config file if provided.")
    max_iterations = Proto(25_000, help="Maximum number of training iterations. Overrides config file if provided.")
    device = Proto(default="cuda:0", dtype=str, help="Device for sim, rl, and graphics")

    rows = Proto(dtype=int, help="num_rows.")
    cols = Proto(dtype=int, help="num_cols")
    print("Here is rows and cols", rows, cols)

    debug = Proto(default=False, help="Disable wandb logging")
    print("deprecate debug flag. Here is rows and cols", debug)

    proj_name = Proto(default="parkour_new", dtype=str, help="run folder name.")

    teacher = Proto(dtype=str, help="Name of the teacher policy to use when distilling")
    # exptid = Proto(dtype=str, help="exptid")
    # resumeid = Proto(dtype=str, help="exptid")
    # daggerid = Proto(dtype=str, help="name of dagger run")
    use_camera = Proto(default=False, help="render camera for distillation")

    mask_obs = Proto(default=False, help="Mask observation when playing")
    use_jit = Proto(default=False, help="Load jit script when playing")
    use_latent = Proto(default=False, help="Load depth latent when playing")
    draw = Proto(default=False, help="draw debug plot when playing")
    save = Proto(default=False, help="save data for evaluation")

    task_both = Proto(default=False, help="Both climbing and hitting policies")
    nodelay = Proto(default=False, help="Add action delay")
    delay = Proto(default=False, help="Add action delay")
    hitid = Proto(dtype=str, default=None, help="exptid fot hitting policy")

    web = Proto(default=False, help="if use web viewer")
    no_wandb = Proto(default=False, help="no wandb")

    headless = Flag("Run headless without creating a viewer window, used to indicate training (as opposed to eval)")
    no_graphics = Proto(
        default=False, help="Disable graphics context creation, no viewer window is created, and no headless rendering is available"
    )

    sim_device = Proto(default="cuda:0", dtype=str, help="Physics Device in PyTorch-like syntax")
    pipeline = Proto(default="gpu", dtype=str, help="Tensor API pipeline (cpu/gpu)")
    graphics_device_id = Proto(default=0, dtype=int, help="Graphics Device ID")

    flex = Proto(default=False, help="Use FleX for physics")
    physx = Proto(default=True, help="Use PhysX for physics")

    num_threads = Proto(default=0, dtype=int, help="Number of cores used by PhysX")
    subscenes = Proto(default=0, dtype=int, help="Number of PhysX subscenes to simulate in parallel")
    slices = Proto(dtype=int, help="Number of client threads that process env slices")

    distill_direction = Proto(default=False, help="Include direction distillation")

    load_checkpoint = Proto(help="If present, will override loading from exptid. Loads the policy from the server.")

    @staticmethod
    def validate_args():
        print("remove line 138 if this is ran twice.")

        assert RunArgs.flex ^ RunArgs.physx, "Only one of Flex or PhysX can be used for physics"
        if RunArgs.device is not None:
            RunArgs.sim_device = RunArgs.device
            RunArgs.rl_device = RunArgs.device

        RunArgs.sim_device_type, RunArgs.compute_device_id = parse_device_str(RunArgs.sim_device)
        pipeline = RunArgs.pipeline.lower()

        assert pipeline == "cpu" or pipeline in ("gpu", "cuda"), f"Invalid pipeline '{RunArgs.pipeline}'. Should be either cpu or gpu."
        RunArgs.use_gpu_pipeline = pipeline in ("gpu", "cuda")

        from isaacgym import gymapi

        RunArgs.physics_engine = gymapi.SIM_PHYSX if RunArgs.physx else gymapi.SIM_FLEX

        if RunArgs.no_graphics:
            RunArgs.headless = True

        if RunArgs.slices is None:
            RunArgs.slices = RunArgs.subscenes

        RunArgs.sim_device_id = RunArgs.compute_device_id
        RunArgs.sim_device = RunArgs.sim_device_type

        if RunArgs.sim_device == "cuda":
            RunArgs.sim_device += f":{RunArgs.sim_device_id}"

        RunArgs.use_gpu = RunArgs.sim_device_type == "cuda"


import platform
#
# # Only run this if it is not a mac
if platform.system() != "Darwin":  # Darwin is the name for the macOS operating system
    print("this can simply be removed, if it is ran twice.")
    RunArgs.validate_args()
