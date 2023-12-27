import argparse
import gymnasium as gym
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

from collections.abc import Callable
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def make_env_generator(
        gym_id: str,
        idx: int,
        capture_video: bool,
        run_name: str,
        num_steps: int
) -> Callable[[], gym.Env]:
    activate_render = capture_video and idx == 0
    render_mode = "rgb_array" if activate_render else None

    def generate_env() -> gym.Env:
        env = gym.wrappers.RecordEpisodeStatistics(
            gym.make(gym_id, render_mode=render_mode, max_episode_steps=num_steps),
            deque_size=num_steps
        )
        if activate_render:
            env = gym.wrappers.RecordVideo(
                env, video_folder="videos", name_prefix=run_name, episode_trigger=lambda n: n % 100 == 0
            )
        return env

    return generate_env


class Agent(nn.Module):

    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        total_num_features_in = np.prod(envs.single_observation_space.shape)
        # NOTE: The critic must return the value of an observation (the value of a state)
        self.critic = nn.Sequential(
            self.__layer_init(nn.Linear(total_num_features_in, 64)),
            nn.Tanh(),
            self.__layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.__layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            self.__layer_init(nn.Linear(total_num_features_in, 64)),
            nn.Tanh(),
            self.__layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # NOTE: The very small std on this output layer ensures that output layer parameters all have
            # similar scalar values (because the random number cannot vary too much with a small std) and
            # as a result, the probability of taking each action will be similar (at least at the beginning
            # of the training)
            self.__layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        )

    # NOTE: Layer initialization strategy typical of the PPO implementation
    @staticmethod
    def __layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation)

    def get_action_and_value(
            self,
            observation: torch.Tensor,
            action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(observation)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(observation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A Proximal Policy Optimization implementation, following the tutorial by Costa Huang "
                    "(https://www.youtube.com/watch?v=MEt6rrxH8W4)",
        epilog="Author: Michele De Stefano"
    )
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="The name of the experiment. Default: %(default)s")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
                        help="The gym environemnt ID to use. Default: %(default)s")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="The learning rate of the optimizer. Default: %(default)s")
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1,
                        help="The minimum learning rate ratio. When using learning rate annealing, the minimum learning"
                             " rate is computed as min_learning_rate_ratio * learning_rate, and it it reached at the"
                             " last update. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed of the experiment. Default: %(default)s")
    parser.add_argument("--total-timesteps", type=int, default=25000,
                        help="Total training timesteps. Default: %(default)s")
    parser.add_argument("--torch-not-deterministic", action="store_true",
                        help="When provided, use disables the deterministic mode of PyTorch")
    parser.add_argument("--no-cuda", action="store_true",
                        help="When provided, uses CPU instead of the GPU")
    parser.add_argument("--capture-video", action="store_true",
                        help="When provided, activates capturing video recording of the agent behavior.")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Total number of parallel environments to use. Default: %(default)s")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="The number of steps to run in each environment per policy rollout. Default: %(default)s")
    parser.add_argument("--no-lr-annealing", action="store_true",
                        help="When provided, keeps the learning rate constant for all the training.")
    parser.add_argument("--no-gae", action="store_true",
                        help="PPO uses General Advantage Estimation (GAE) by default. Specifying this flag disables"
                             " GAE computation.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Rewards discount factor. Default: %(default)s")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="The lambda parameter for the General Advantage Estimation. Default: %(default)s")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="The number of minibatches used for training. Default: %(default)s")
    parser.add_argument("--num-update-epochs", type=int, default=4,
                        help="The number of epochs to run before updating the policy. Default: %(default)s")
    parser.add_argument("--no-advantage-normalization", action="store_true",
                        help="When provided, it disables advantage normalization, which is done by default by PPO")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="The surrogate objective clipping coefficient. Default: %(default)s")
    parser.add_argument("--no-value-loss-clip", action="store_true",
                        help="When provided, disables value-loss clipping, which is done by default by PPO")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy loss coefficient. Default: %(default)s")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value loss coefficient. Default: %(default)s")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum norm for the gradient. Used for gradient clipping. Default: %(default)s")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="A target KL-divergence threshold. When provided, it enables early-stopping if the"
                             " threshold is exceeded. If you want to use it, a good value is 0.015.")

    arguments = parser.parse_args()
    arguments.batch_size = int(arguments.num_envs * arguments.num_steps)
    arguments.minibatch_size = int(arguments.batch_size // arguments.num_minibatches)
    return arguments


def main() -> None:
    args = parse_args()
    print(args)
    run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = not args.torch_not_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([
        make_env_generator(args.gym_id, i, args.capture_video, run_name, num_steps=args.num_steps)
        for i in range(args.num_envs)
    ])

    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    seed_list = [args.seed + i for i in range(args.num_envs)]
    obs_batch, _ = envs.reset(seed=seed_list)

    agent = Agent(envs=envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1.e-5)

    # ALGO LOGIC: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros_like(logprobs).to(device)
    dones = torch.zeros_like(logprobs, dtype=torch.bool).to(device)
    values = torch.zeros_like(logprobs).to(device)

    global_step = 0
    start_time = time.time()
    # Batch of initial observations
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    # Initial termination conditions (all false)
    next_done = torch.zeros(args.num_envs).to(device)
    # The following is a division with ceiling, so I ensure at least total_timesteps are performed
    num_updates = -(args.total_timesteps // -args.batch_size)

    anneal_lr = not args.no_lr_annealing

    lr_anneal_frac = (1 - args.min_learning_rate_ratio) / (num_updates - 1) if anneal_lr else 0.0

    # Training loop
    for update in range(1, num_updates + 1):
        cur_lr = args.learning_rate * (1.0 - lr_anneal_frac * (update - 1.0))
        optimizer.param_groups[0]["lr"] = cur_lr

        # Run a whole rollout: we interact with the environment by using our agent for num_steps
        # NOTE: Because we use a vectorized environment, we always run a fixed total number of steps
        # During these steps, some episodes will end and restart
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # DON'T MODIFY: EXECUTE THE GAME AND LOG DATA
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.tensor(next_obs).to(device)
            next_done = torch.tensor(terminated | truncated).to(device)

            # The following code logs some quantities to tensorboard
            if len(infos) > 0 and np.any(infos["_final_info"]):
                # At least one environment has terminated an episode
                valid_returns = []
                valid_lengths = []
                for has_info, cur_info in zip(infos["_final_info"], infos["final_info"]):
                    if not has_info:
                        continue
                    valid_returns += [cur_info["episode"]["r"][-1]]
                    valid_lengths += [cur_info["episode"]["l"][-1]]
                avg_episodic_return = np.mean(valid_returns)
                avg_episodic_length = np.mean(valid_lengths)
                print(f"global_step: {global_step}; avg. episodic_return: {avg_episodic_return}; "
                      f"avg. length: {avg_episodic_length}")
                writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
                writer.add_scalar("charts/avg_episodic_length", avg_episodic_length, global_step)

        # bootstrap rewards if not done and prepare variables for the objective function computation
        compute_gae = not args.no_gae
        with torch.no_grad():
            # NOTE: with the following instructions when an environment is not done (after the num_steps)
            # done above) we estimate the value of the next obs. as the end-rollout value. This is done
            # both when we compute/don't compute GAE
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if compute_gae:
                # Recall that rewards is num_steps x num_envs
                advantages = torch.zeros_like(rewards).to(device)
                last_gae_lambda = 0
                for t in reversed(range(args.num_steps)):
                    # NOTE: read after the "else" of "if compute_gae" to understand the function of next_nonterminal
                    if t == args.num_steps - 1:
                        next_nonterminal = ~next_done
                        next_values = next_value
                    else:
                        next_nonterminal = ~dones[t + 1]
                        next_values = values[t + 1]
                    # The following is a partial advantage computation
                    delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                    # The following is basically a "discounted advantage computation". This is specific to PPO
                    # gae_lambda is a factor that modifies gamma for this discount computation
                    advantages[t] = last_gae_lambda = \
                        delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae_lambda
                # NOTE: The following is an alternative way of computing returns: we first compute advantages and values
                returns = advantages + values
            else:
                # Normal advantage calculation (no GAE)
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_nonterminal = ~next_done
                        next_return = next_value
                    else:
                        next_nonterminal = ~dones[t + 1]
                        next_return = returns[t + 1]
                    # NOTE: The following is the standard way of computing returns and advantages
                    # The return at state t is the reward if state t is a terminal state, otherwise
                    # it is the discounted reward. Notice how next_nonterminal is a 1-0 mask (actually True-False)
                    # that sets to 0 the discounted return of the next state if the current state is a terminal state
                    # (the name next_nonterminal is misleading: contains True if the current state is nonterminal;
                    # the "next" prefix only indicates that next_nonterminal has to be applied to next_return)
                    returns[t] = rewards[t] + args.gamma * next_nonterminal * next_return
                advantages = returns - values

        # At this point we have acquired experiences from the environment. We prepare the variables for the
        # computation of the objective function to optimize

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value networks: here we need gradient tracking, so we don't disable it in general
        # We are going to disable gradient tracking only for monitoring code
        batch_inds = np.arange(args.batch_size)
        # MONITORING variable: measures how often the clipped objective is actually triggered
        clipfracs = []
        # Run some epochs through the prepared data history, and update network pars
        for epoch in range(args.num_update_epochs):
            np.random.shuffle(batch_inds)
            # We proceed in mini-batches of the batched data instead of using the whole batch in one shot.
            # This is an implementation choice of PPO. In the following, I use "mb" prefix to indicate
            # quantities related to a mini-batch
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = batch_inds[start:end]

                # Here is where I run through the history and re-evaluate each observation, with the
                # action that was taken during the interaction with the environment
                _, new_logprob, entropy, new_values = \
                    agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                log_ratio = new_logprob - b_logprobs[mb_inds]
                ratio = log_ratio.exp()

                # MONITORING: compute KL-divergence and clip activation flags
                with torch.no_grad():
                    # calculate approx KL-divergence (http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                do_advantage_normalization = not args.no_advantage_normalization

                mb_advantages = b_advantages[mb_inds]
                if do_advantage_normalization:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1.e-8)

                # Policy-Gradient loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                # NOTE: In the following I take the MAX (and not the MIN like in the PPO paper) because I am
                # minimizing a loss function, which is -(objective function). In the paper, everything is
                # written for maximizing an objective function, while here we are minimizing -(objective)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                clip_value_loss = not args.no_value_loss_clip

                # Value loss
                new_values = new_values.view(-1) # Recall that these arrive from the critic evaluation
                if clip_value_loss:
                    v_loss_unclipped = (new_values - b_returns[mb_inds])**2
                    v_clipped = b_values[mb_inds] + \
                                torch.clamp(new_values - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[mb_inds])**2).mean()

                entropy_loss = entropy.mean()

                # Compose the loss function with the previously computed pieces
                # The logic is to minimize the policy and value losses, while maximizing the entropy loss
                # Maximizing entropy should encourage more exploration
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                # Perform the optimization step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # Do early stopping if the monitored KL-divergence becomes too large (when a threshold is specified)
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # MONITORING: explained variance on the estimates of the value function (tells if the value function is
        # a good indicator of the actual returns)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # We now use tensorboard to record all the metrics (try not modifying the following)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        steps_per_second = round(global_step / (time.time() - start_time))
        print(f"Steps per second: {steps_per_second}")
        writer.add_scalar("charts/steps_per_second", steps_per_second, global_step)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
