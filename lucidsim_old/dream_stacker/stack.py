"""
Use optical flow to warp our dreams, so they are consistent within some interval  

Note: due to the stack sizing, it may not match the trajectory length exactly, so the first few timesteps will be dropped during training.  
"""

import os

from params_proto import ParamsProto


class DreamStack(ParamsProto):
    root = "http://luma01.csail.mit.edu:4000"

    prefix = "scenes/experiments/real2real"

    scene: str = "ball/scene_00001"
    rollout = 1

    device = "cuda:0"

    # how many frames to make consistent
    stack_size = 5

    # optical flow batch size, lower to comply with memory limits 
    chunk_size = 4
    save_videos = True

    def __post_init__(self, _deps=None):
        from ml_logger import ML_Logger
        print(f"Running scene {self.scene}, rollout number {self.rollout}...")
        self.logger = ML_Logger(root=self.root, prefix=os.path.join(self.prefix, self.scene))
        self.rollout = int(self.rollout)

    def __call__(self, ):
        from agility_analysis.matia.pose_estimation.loaders.of_pose_loader import OFPoseLoader
        from agility_analysis.matia.pose_estimation.loaders.pose_loader import PoseLoader
        from agility_analysis.matia.pose_estimation.optical_flow.utils import consistent_dream
        rollout_range = [self.rollout, self.rollout + 1]
        rgb_loader = OFPoseLoader(self.root, self.prefix, [self.scene], rollout_range, "depth", self.device)

        dream_loader = PoseLoader(self.root, self.prefix, [self.scene], rollout_range, "augmented",
                                  self.device)

        with self.logger.Prefix(f"stacked_lucid_dreams/imagen_{self.rollout:04d}"):
            consistent_dream(rgb_loader, dream_loader, self.logger, self.stack_size, chunk_size=self.chunk_size,
                             save_video=self.save_videos)


if __name__ == '__main__':
    dream_stacker = DreamStack()
    dream_stacker()
