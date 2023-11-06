# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC

# The only difference from the original implementation:
# default pytorch weight initialization,
# without custom rlkit init & uniform init for last layers.
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import os
import random
import uuid

import d4rl
import gym
import numpy as np
import pyrallis
import torch
from torch.distributions import Normal
from torch.distributions.dirichlet import Dirichlet
import torch.nn as nn
from tqdm import trange
import wandb

Q_MAX_VALUE = 1000
@dataclass
class TrainConfig:
    # wandb params
    project: str = "CORL"
    group: str = "APE-V"
    name: str = "APE-V"
    env_name: str = "halfcheetah-medium-v2" # Hopper-v3
    task_data_type: str = "low"
    task_train_num: int = 100
    data_dir: str = "/input"
    diri: float = 0.01
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    num_agents: int = 20
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 1_000_000
    batch_size: int = 256
    num_epochs: int = 2000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 20
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    seed: int = 0
    # train_seed: int = 0
    # eval_seed: int = 0
    log_every: int = 500
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{self.seed}-{str(uuid.uuid4())[:4]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils.
TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        # name=f"{config['env_name']}_{config['task_data_type']}_{config['task_train_num']}_{config['seed']}",
        id=str(uuid.uuid4()),
    )
    
    wandb.run.save()


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(deterministic_torch)


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        raise NotImplementedError

class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias

class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, belief_dim: int, max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        hidden = self.trunk(torch.cat([state, belief], dim=-1))
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        # log_sigma = torch.clip(log_sigma, -5, 2)
        log_sigma = torch.clamp(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))


        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, belief: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        belief = torch.tensor(belief, device=device, dtype=torch.float32)
        action = self(state, belief, deterministic=deterministic)[0].cpu().numpy()
        return action
    
    def _get_next_relative_belief(
        self, 
        apev,
        state: np.ndarray, 
        belief: np.ndarray, 
        action: np.ndarray,
        reward: np.ndarray, 
        next_state: np.ndarray, 
        done: bool,
        device: str,
    )-> np.ndarray:
        state = torch.tensor(np.array([state]), device=device, dtype=torch.float32)
        belief = torch.tensor(np.array([belief]), device=device, dtype=torch.float32)
        action = torch.tensor(np.array([action]), device=device, dtype=torch.float32)
        reward = torch.tensor(np.array([reward]), device=device, dtype=torch.float32)
        next_state = torch.tensor(np.array([next_state]), device=device, dtype=torch.float32)
        done = torch.tensor(np.array([done]), device=device, dtype=torch.float32)
        with torch.no_grad():        
            td_err = apev._critic_loss(state, belief, action, reward, next_state, belief, done, need_td_err=True)            
            # [ensemble_size, batch_size]
            # log_likelihood = torch.clip(-td_err, -5, 0)
            # likelihood = torch.exp(-td_err) # TODO CHECK CLIP 
        log_next_belief = torch.log(belief) - td_err - torch.logsumexp((-td_err + torch.log(belief)), dim=1, keepdim=True)
        next_belief = torch.exp(log_next_belief) 
        if torch.isnan(belief).any() or torch.isnan(next_belief).any():
            import pdb; pdb.set_trace()
        
        return next_belief.view(-1).cpu().numpy()

# num_agents, num_critics
class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_agents: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + num_agents + action_dim, hidden_dim, num_critics * num_agents),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics * num_agents),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics * num_agents),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics * num_agents),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_agents = num_agents
        self.num_critics = num_critics
    
    def forward(self, state: torch.Tensor, belief: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, belief, action], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.num_critics * self.num_agents, dim=0
        )
        # [num_agents * num_critics, batch_size] --> [num_agents, num_critics, batch_siz]
        q_values = self.critic(state_action).squeeze(-1).reshape(self.num_agents, self.num_critics, -1)
        return q_values


class APEV:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critic: VectorizedCritic,
        critic_optimizer: torch.optim.Optimizer,
        belief_dim : int, # num_agent
        num_critics: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        diri: float = 0.01,
        alpha_learning_rate: float = 1e-4,
        device: str = "cpu",  # noqa
    ):
        self.device = device
        self.actor = actor
        self.critic = critic
        self.num_critics = num_critics
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.num_agents = self.belief_dim = belief_dim
        self.tau = tau
        self.gamma = gamma
        self.dist = Dirichlet(torch.tensor([diri]*self.belief_dim))
        
        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

    def _alpha_loss(self, state: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_log_prob = self.actor(state, belief, need_log_prob=True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss

    def _get_belief(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor
        ) -> torch.Tensor:

        bs = state.shape[0]
        belief = self.dist.sample([bs]).to(self.device) # [batch_size, ensemble_size]
        with torch.no_grad():        
            td_err = self._critic_loss(state, belief, action, reward, next_state, belief, done, need_td_err=True)            

        log_next_belief = torch.log(belief) - td_err - torch.logsumexp((-td_err + torch.log(belief)), dim=1, keepdim=True)
        next_belief = torch.exp(log_next_belief) 
        if torch.isnan(belief).any() or torch.isnan(next_belief).any():
            import pdb; pdb.set_trace()
        return belief, next_belief

    def _actor_loss(
        self,
        state: torch.Tensor, 
        belief: torch.Tensor,
        ) -> Tuple[torch.Tensor, float, float]:
        
        action, action_log_prob = self.actor(state, belief, need_log_prob=True)
        
        # [num_agents, num_critics, batch_size] --> [num_agents, batch_size] --> batch_size, num_agents
        q_value_min = self.critic(state, belief, action).min(1).values.T
        assert q_value_min.shape[1] == self.belief_dim
        # [batch_size, ensemble_size] --> batch_size
        filtered_q_value = (belief * q_value_min).sum(-1)
        loss = (self.alpha * action_log_prob - filtered_q_value).mean()
        batch_entropy = -action_log_prob.mean().item()
        return loss, batch_entropy

    def _critic_loss(
        self,
        state: torch.Tensor, 
        belief: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor, 
        next_state: torch.Tensor,
        next_belief: torch.Tensor,
        done: torch.Tensor,
        need_td_err: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, next_belief, need_log_prob=True)
            # [num_agents, num_critics, batch_size] 
            qs_next = self.target_critic(next_state, next_belief, next_action)
            # [num_agents, batch_size]
            q_next = qs_next.min(1).values
            q_next = q_next - self.alpha * next_action_log_prob
            # [num_agents, 1, batch_size]
            # reward (batch_size, 1) --> (1, 1, batch_size)
            # [num_agents, 1, batch_size]
            q_target = reward.T[None, ...] + self.gamma * (1 - done.T[None, ...]) * q_next.unsqueeze(1)

        # num_agents, num_critic, batch_size
        q_value = self.critic(state, belief, action)
        if need_td_err: 
            #num_agents, batch_size --> batch_size, num_agents 
            return ((q_value.min(1).values - q_target.squeeze(1))**2).T 
        # [num_agent, num_critic, batch_size] - [batch_size, 1, ensemble_size]        
        td_err = ((q_value - q_target) ** 2)
        loss = td_err.sum(dim=1).sum(dim=0).mean()
        return loss

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = [arr.to(self.device) for arr in batch]
        # Usually updates are done in the following order: critic -> actor -> alpha
        # But we found that EDAC paper uses reverse (which gives better results)
                
        # Belief update
        belief, next_belief = self._get_belief(state, action, reward, next_state, done)
        
        # Alpha update
        alpha_loss = self._alpha_loss(state, belief)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()
        
        # Critic update
        critic_loss = self._critic_loss(state, belief, action, reward, next_state, next_belief, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss, actor_batch_entropy = self._actor_loss(state, belief)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)
            q_random_std = self.critic(state, belief, random_actions).std(0).mean().item()

        update_info = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            # "q_policy_std": q_policy_std,
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_random_std": q_random_std,
            "belief_max": belief.max().item(),
            "next_belif_max": next_belief.max().item()
        }
        return update_info
    
    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])

@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: Actor, apev: APEV, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    last_belief_max_value = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        # TODO
        belief = np.ones(apev.belief_dim) / apev.belief_dim 
        episode_reward = 0.0
        while not done:
            action = actor.act(state, belief, device)
            next_state, reward, done, _ = env.step(action)
            belief = actor._get_next_relative_belief(apev, state, belief, action, reward, next_state, done, device)
            state = next_state
            episode_reward += reward
        last_belief_max_value.append(belief.max().item())
        episode_rewards.append(episode_reward)

    actor.train()
    return np.array(episode_rewards), np.array(last_belief_max_value)

def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


@pyrallis.wrap()
def train(config: TrainConfig):
    print(config)
    set_seed(config.seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    # data, evaluation, env setup
    if config.env_name in ["Hopper-v3", "Walker2d-v3", "HalfCheetah-v3"]:
        import neorl
        env = neorl.make(config.env_name)
        train, val = env.get_dataset(data_type=config.task_data_type, train_num=config.task_train_num, need_val=False, path=config.data_dir)
        # train, val = env.get_dataset(data_type=config.task_data_type, train_num=config.task_train_num, need_val=False)
        dataset = dict()
        dataset['observations'] = train['obs'].copy() 
        dataset['actions'] = train['action'].copy()
        dataset['next_observations'] = train['next_obs'].copy()
        dataset['rewards'] = train['reward'].squeeze().copy()
        dataset['terminals'] = train['done'].squeeze().copy()
        eval_env = wrap_env(env)
    else: 
        env = gym.make(config.env_name)
        # d4rl_dataset = d4rl.qlearning_dataset(eval_env)
        # dataset = d4rl.qlearning_dataset(eval_env)
        eval_env = wrap_env(env)
        dataset = d4rl.qlearning_dataset(eval_env)
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    if config.normalize_reward:
        modify_reward(dataset, config.env_name)

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=config.device,
    )
    buffer.load_d4rl_dataset(dataset)

    # Actor & Critic setup
    # belief_dim = config.num_critics - 1 
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.num_agents, config.max_action)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.num_agents, config.num_critics
    )
    critic.to(config.device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )

    trainer = APEV(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        belief_dim=config.num_agents,
        num_critics=config.num_critics,
        gamma=config.gamma,
        tau=config.tau,
        diri=config.diri,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    total_updates = 0.0
    for epoch in trange(config.num_epochs, desc="Training"):
        # training
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.sample(config.batch_size)
            update_info = trainer.update(batch)

            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, **update_info})

            total_updates += 1

        # evaluation
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns, eval_belief = eval_actor(
                env=eval_env,
                actor=actor,
                apev=trainer,
                n_episodes=config.eval_episodes,
                seed=config.seed,
                device=config.device,
            )
            eval_log = {
                "eval/reward_mean": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "eval/lastb_max_max": eval_belief.max(),
                "eval/lastb_max_mean": eval_belief.mean(),
                "eval/lastb_max_min": eval_belief.min(),
                "epoch": epoch,
            }
            if hasattr(eval_env, "get_normalized_score"):
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
                eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                eval_log["eval/normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log)
            print(eval_log)

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"{epoch}.pt"),
                )

    wandb.finish()

if __name__ == "__main__":
    train()