from pathlib import Path

if __name__ == "__main__":
    from params_proto.hyper import Sweep

    from lucidsim_old.traj_generation.traj_gen import TrajGenerator

    with Sweep(TrajGenerator) as sweep:
        TrajGenerator.rollout_range = [0, 20]
        TrajGenerator.add_noise_prob = 1.0
        TrajGenerator.sampling_noise = 0.1
        with sweep.product:
            TrajGenerator.terrain_type = ["parkour_flat", "parkour_hurdle", "parkour_gap", "parkour_step"]


    @sweep.each
    def tail(TG: TrajGenerator):
        TG.prefix = f"lucidsim/scenes/{TG.terrain_type}/noisy"


    sweep.save(f"{Path(__file__).stem}.jsonl")
