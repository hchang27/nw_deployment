import statistics
import time
import warnings
from collections import deque
from copy import deepcopy, copy

import torch
from params_proto import ParamsProto

from cxx.algorithms import PPO
from cxx.env import VecEnv
from cxx.modules import ActorCritic, Estimator, DepthOnlyFCBackbone, RecurrentDepthBackbone
from main_street.config import RunArgs


class RunnerArgs(ParamsProto, prefix="runner"):
    log_every = 100
    save_intermediate_checkpoints = True


class OnPolicyRunner:
    def __init__(
        self,
        env: VecEnv,
        train_cfg,
        # log_dir=None,
        # init_wandb=True,
        device="cpu",
        **kwargs,
    ):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.device = device
        self.env = env

        print("Using MLP and Priviliged Env encoder ActorCritic structure")
        actor_critic: ActorCritic = ActorCritic(
            self.env.cfg.env.n_proprio,
            self.env.cfg.env.n_scan,
            self.env.num_obs,
            self.env.cfg.env.n_priv_latent,
            self.env.cfg.env.n_priv,
            self.env.cfg.env.history_len,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        estimator = Estimator(
            input_dim=env.cfg.env.n_proprio, output_dim=env.cfg.env.n_priv, hidden_dims=self.estimator_cfg["hidden_dims"]
        ).to(self.device)
        # Depth encoder
        self.if_depth = self.depth_encoder_cfg["if_depth"]
        if self.if_depth:
            depth_backbone = DepthOnlyFCBackbone(
                env.cfg.env.n_proprio,
                self.policy_cfg["scan_encoder_dims"][-1],
                self.depth_encoder_cfg["hidden_dims"],
                env.cfg.depth.resized[::-1],
            )
            depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg.env.n_proprio).to(self.device)
            depth_actor = deepcopy(actor_critic.actor)
        else:
            depth_encoder = None
            depth_actor = None
        # self.depth_encoder = depth_encoder
        # self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=self.depth_encoder_cfg["learning_rate"])
        # self.depth_encoder_params = self.depth_encoder_cfg
        # self.depth_encoder_criterion = nn.MSELoss()
        # Create algorithm
        alg_class: type["PPO"] = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(
            actor_critic=actor_critic,
            estimator=estimator,
            estimator_params=self.estimator_cfg,
            depth_encoder=depth_encoder,
            depth_encoder_params=self.depth_encoder_cfg,
            depth_actor=depth_actor,
            device=self.device,
            **self.alg_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        self.learn = self.learn_RL if not self.if_depth else self.learn_vision

        # Log
        # self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.curr_step = 0

    def learn_RL(self, num_steps, init_at_random_ep_len=True):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # tot_iter = self.curr_step + num_learning_iterations
        # todo: remove, this is not used.
        self.start_learning_iteration = copy(self.curr_step)

        for self.curr_step in range(self.curr_step, num_steps):
            start = time.time()
            hist_encoding = self.curr_step % self.dagger_update_freq == 0

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(
                        actions
                    )  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    total_rew = self.alg.process_env_step(rewards, dones, infos)

                    # fixme: remove these or confirm that these are useful.
                    # print("deprecation warning: these were conditioned on the log_dir, should just log anyways. can turn off.")
                    # if self.log_dir is not None:
                    # Book keeping
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    cur_reward_sum += total_rew
                    cur_reward_explr_sum += 0
                    cur_reward_entropy_sum += 0
                    cur_episode_length += 1

                    new_ids = (dones > 0).nonzero(as_tuple=False)

                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                    rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                    cur_reward_sum[new_ids] = 0
                    cur_reward_explr_sum[new_ids] = 0
                    cur_reward_entropy_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_estimator_loss,
                mean_disc_loss,
                mean_disc_acc,
                mean_priv_reg_loss,
                priv_reg_coef,
            ) = self.alg.update()
            if hist_encoding:
                print("Updating dagger...")
                mean_hist_latent_loss = self.alg.update_dagger()

            stop = time.time()
            learn_time = stop - start
            # if self.log_dir is not None:
            self.log(locals())

            if self.curr_step < 2500:
                if self.curr_step % self.save_interval == 0:
                    self.save(fname=f"checkpoints/model_{self.curr_step}.pt")
            elif self.curr_step < 5000:
                if self.curr_step % (2 * self.save_interval) == 0:
                    self.save(fname=f"checkpoints/model_{self.curr_step}.pt")
            else:
                if self.curr_step % (5 * self.save_interval) == 0:
                    self.save(fname=f"checkpoints/model_{self.curr_step}.pt")
            ep_infos.clear()

        # self.current_learning_iteration += num_learning_iterations
        self.save(f"model_{self.curr_step}.pt")

    def learn_vision(self, num_steps, **_):
        # tot_iter = self.curr_step + num_learning_iterations
        self.start_learning_iteration = copy(self.curr_step)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        obs = self.env.get_observations()
        infos = {}
        if self.if_depth and not self.env.cfg.depth.ignore_vision:
            infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1]
        else:
            infos["depth"] = None
        infos["delta_yaw_ok"] = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()

        num_pretrain_iter = 0
        for self.curr_step in range(self.curr_step, num_steps):
            start = time.time()
            depth_latent_buffer = []
            scandots_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            yaw_buffer_student = []
            yaw_buffer_teacher = []
            delta_yaw_ok_buffer = []
            for i in range(self.depth_encoder_cfg["num_steps_per_env"]):
                if infos["depth"] != None:
                    with torch.no_grad():
                        scandots_latent = self.alg.actor_critic.actor.infer_scandots_latent(obs)
                    scandots_latent_buffer.append(scandots_latent)
                    obs_prop_depth = obs[:, : self.env.cfg.env.n_proprio].clone()
                    obs_prop_depth[:, 6:8] = 0
                    depth_latent_and_yaw = self.alg.depth_encoder(
                        infos["depth"].clone(), obs_prop_depth
                    )  # clone is crucial to avoid in-place operation

                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = 1.5 * depth_latent_and_yaw[:, -2:]

                    depth_latent_buffer.append(depth_latent)
                    yaw_buffer_student.append(yaw)
                    yaw_buffer_teacher.append(obs[:, 6:8])

                with torch.no_grad():
                    actions_teacher = self.alg.actor_critic.act_inference(obs, hist_encoding=True, scandots_latent=None)
                    actions_teacher_buffer.append(actions_teacher)

                obs_student = obs.clone()
                # obs_student[:, 6:8] = yaw.detach()
                if RunArgs.distill_direction:
                    obs_student[infos["delta_yaw_ok"], 6:8] = yaw.detach()[infos["delta_yaw_ok"]]
                delta_yaw_ok_buffer.append(torch.nonzero(infos["delta_yaw_ok"]).size(0) / infos["delta_yaw_ok"].numel())
                actions_student = self.alg.depth_actor(obs_student, hist_encoding=True, scandots_latent=depth_latent)
                actions_student_buffer.append(actions_student)

                # detach actions before feeding the env
                if self.curr_step < num_pretrain_iter:
                    obs, privileged_obs, rewards, dones, infos = self.env.step(
                        actions_teacher.detach()
                    )  # obs has changed to next_obs !! if done obs has been reset
                else:
                    obs, privileged_obs, rewards, dones, infos = self.env.step(
                        actions_student.detach()
                    )  # obs has changed to next_obs !! if done obs has been reset
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = (
                    obs.to(self.device),
                    critic_obs.to(self.device),
                    rewards.to(self.device),
                    dones.to(self.device),
                )

                #  remove or confirm these are useful
                # if self.log_dir is not None:
                # Book keeping
                if "episode" in infos:
                    ep_infos.append(infos["episode"])
                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
            if self.if_depth and not self.env.cfg.depth.ignore_vision:
                scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)
                depth_latent_buffer = torch.cat(depth_latent_buffer, dim=0)
            depth_encoder_loss = 0
            # depth_encoder_loss = self.alg.update_depth_encoder(depth_latent_buffer, scandots_latent_buffer)

            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            yaw_buffer_student = torch.cat(yaw_buffer_student, dim=0)
            yaw_buffer_teacher = torch.cat(yaw_buffer_teacher, dim=0)
            depth_actor_loss, yaw_loss = self.alg.update_depth_actor(
                actions_student_buffer, actions_teacher_buffer, yaw_buffer_student, yaw_buffer_teacher
            )

            # depth_encoder_loss, depth_actor_loss = self.alg.update_depth_both(depth_latent_buffer, scandots_latent_buffer, actions_student_buffer, actions_teacher_buffer)
            stop = time.time()
            learn_time = stop - start

            if self.if_depth and not self.env.cfg.depth.ignore_vision:
                self.alg.depth_encoder.detach_hidden_states()

            # if self.log_dir is not None:
            self.log_vision(locals())

            if (
                (self.curr_step < 2500 and self.curr_step % self.save_interval == 0)
                or (self.curr_step < 5000 and self.curr_step % (2 * self.save_interval) == 0)
                or (self.curr_step >= 5000 and self.curr_step % (5 * self.save_interval) == 0)
            ):
                self.save(f"model_{self.curr_step}.pt")

            ep_infos.clear()

    def log_vision(self, locs, width=80, pad=35):
        from ml_logger import logger

        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        wandb_dict = {}
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict["Episode_rew/" + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        wandb_dict["Loss_depth/delta_yaw_ok_percent"] = locs["delta_yaw_ok_percentage"]
        wandb_dict["Loss_depth/depth_encoder"] = locs["depth_encoder_loss"]
        wandb_dict["Loss_depth/depth_actor"] = locs["depth_actor_loss"]
        wandb_dict["Loss_depth/yaw"] = locs["yaw_loss"]
        wandb_dict["Policy/mean_noise_std"] = mean_std.item()
        wandb_dict["Perf/total_fps"] = fps
        wandb_dict["Perf/collection time"] = locs["collection_time"]
        wandb_dict["Perf/learning_time"] = locs["learn_time"]

        log_keys = ["Episode_rew/rew_tracking_goal_vel", "Episode_rew/rew_tracking_yaw", "Loss_depth/depth_actor"]

        if len(locs["rewbuffer"]) > 0:
            wandb_dict["Train/mean_reward"] = statistics.mean(locs["rewbuffer"])
            wandb_dict["Train/mean_episode_length"] = statistics.mean(locs["lenbuffer"])
            log_keys.append("Train/mean_reward")

        logger.store_metrics(**{k: wandb_dict[k] for k in log_keys}, step=self.curr_step)

        if logger.every(RunnerArgs.log_every, "iteration", start_on=1):
            logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": self.curr_step})
            logger.job_running()

        str = f" \033[1m Learning iteration {self.curr_step}/{self.curr_step + locs['num_steps']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'Depth encoder loss:':>{pad}} {locs['depth_encoder_loss']:.4f}\n"""
                f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n"""
                f"""{'Yaw loss:':>{pad}} {locs['yaw_loss']:.4f}\n"""
                f"""{'Delta yaw ok percentage:':>{pad}} {locs['delta_yaw_ok_percentage']:.4f}\n"""
            )
        else:
            log_string = f"""{'#' * width}\n"""

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = self.curr_step - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs["num_steps"] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n"""
        )
        print(log_string)

    def log(self, locs, width=80, pad=35):
        from ml_logger import logger

        assert logger.prefix

        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        wandb_dict = {}
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict["Episode_rew/" + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        print("remove wandb logging, remove log_dir.")
        wandb_dict["Loss/value_function"] = ["mean_value_loss"]
        wandb_dict["Loss/surrogate"] = locs["mean_surrogate_loss"]
        wandb_dict["Loss/estimator"] = locs["mean_estimator_loss"]
        wandb_dict["Loss/hist_latent_loss"] = locs["mean_hist_latent_loss"]
        wandb_dict["Loss/priv_reg_loss"] = locs["mean_priv_reg_loss"]
        wandb_dict["Loss/priv_ref_lambda"] = locs["priv_reg_coef"]
        # wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict["Loss/learning_rate"] = self.alg.learning_rate
        wandb_dict["Loss/discriminator"] = locs["mean_disc_loss"]
        wandb_dict["Loss/discriminator_accuracy"] = locs["mean_disc_acc"]

        wandb_dict["Policy/mean_noise_std"] = mean_std.item()
        wandb_dict["Perf/total_fps"] = fps
        wandb_dict["Perf/collection time"] = locs["collection_time"]
        wandb_dict["Perf/learning_time"] = locs["learn_time"]

        log_keys = ["Episode_rew/rew_tracking_goal_vel", "Episode_rew/rew_tracking_yaw"]

        if len(locs["rewbuffer"]) > 0:
            wandb_dict["Train/mean_reward"] = statistics.mean(locs["rewbuffer"])
            wandb_dict["Train/mean_reward_explr"] = statistics.mean(locs["rew_explr_buffer"])
            wandb_dict["Train/mean_reward_task"] = wandb_dict["Train/mean_reward"] - wandb_dict["Train/mean_reward_explr"]
            wandb_dict["Train/mean_reward_entropy"] = statistics.mean(locs["rew_entropy_buffer"])
            wandb_dict["Train/mean_episode_length"] = statistics.mean(locs["lenbuffer"])
            log_keys.append("Train/mean_reward")

        logger.store_metrics(**{k: wandb_dict[k] for k in log_keys}, step=self.curr_step)

        if logger.every(RunnerArgs.log_every, "iteration", start_on=1):
            logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": self.curr_step})
            logger.job_running()

        str = f" \033[1m Learning iteration {self.curr_step}/{self.curr_step + locs['num_steps']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['rewbuffer']) - statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = self.curr_step - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs["num_steps"] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n"""
        )
        try:
            logger.print(log_string)
        except:
            print("wtf")

    def save(self, fname="checkpoints/model_last.pt", infos=None):
        from ml_logger import logger

        state_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "estimator_state_dict": self.alg.estimator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.curr_step,
            "infos": infos,
        }
        if self.if_depth:
            state_dict["depth_encoder_state_dict"] = self.alg.depth_encoder.state_dict()
            state_dict["depth_actor_state_dict"] = self.alg.depth_actor.state_dict()

        # model_fname = os.path.basename(path)

        if RunnerArgs.save_intermediate_checkpoints:
            # torch.save(state_dict, path)
            logger.torch_save(state_dict, fname)
            logger.duplicate(fname, "checkpoints/model_last.pt")
        else:
            logger.torch_save(state_dict, path=fname)

    def load(self, path, load_from_logger=False, load_optimizer=True):
        print("*" * 80)
        print(f"Loading model from {path}...")
        if load_from_logger:
            from ml_logger import logger
            loaded_dict = logger.torch_load(path, map_location=self.device)
        else:
            loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.estimator.load_state_dict(loaded_dict["estimator_state_dict"])
        if self.if_depth:
            if "depth_encoder_state_dict" not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict["depth_encoder_state_dict"])
            if "depth_actor_state_dict" in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict["depth_actor_state_dict"])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # this is for continuing training
            self.curr_step = loaded_dict["iter"]

        print("*" * 80)
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_depth_actor_inference_policy(self, device=None):
        self.alg.depth_actor.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.depth_actor.to(device)
        return self.alg.depth_actor

    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic

    def get_estimator_inference_policy(self, device=None):
        self.alg.estimator.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference

    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder

    def get_disc_inference_policy(self, device=None):
        self.alg.discriminator.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.discriminator.to(device)
        return self.alg.discriminator.inference
