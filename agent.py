from typing import Optional, Union, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import gym
from tqdm import tqdm
import pybullet as pb
import matplotlib.pyplot as plt
import os

from equivariant_delta_pred import EquivariantDeltaNetwork
from utils import ReplayBuffer, plot_curves  # , plot_predictions, plot_curves
from grasping_env import HandoverGraspingEnv


class DQNAgent:
    def __init__(self,
                 env: HandoverGraspingEnv,
                 gamma: float,
                 learning_rate: float,
                 buffer_size: int,
                 batch_size: int,
                 initial_epsilon: float,
                 final_epsilon: float,
                 update_method: str = 'standard',
                 exploration_fraction: float = 0.9,
                 target_network_update_freq: int = 1000,
                 seed: int = 0,
                 device: Union[str, torch.device] = 'cpu',
                 ) -> None:
        self.env = env

        self.gamma = gamma
        self.batch_size = batch_size
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.exploration_fraction = exploration_fraction
        self.target_network_update_freq = target_network_update_freq
        self.update_method = update_method

        self.buffer = ReplayBuffer(buffer_size,
                                   env.observation_space.shape,
                                   env.action_space.shape)

        self.device = device
        img_shape = (3, self.env.img_size, self.env.img_size)
        self.network = EquivariantDeltaNetwork(img_shape).to(device)
        self.target_network = EquivariantDeltaNetwork(img_shape).to(device)
        self.hard_target_update()

        self.optim = torch.optim.Adam(self.network.parameters(),
                                      lr=learning_rate)

        # extra things to pickle
        self.global_step = 1
        self.rewards_data = []
        self.success_data = []
        self.loss_data = []
        self.episode_count = 0
        self.episode_rewards = 0

        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

        # if snapshot exists, load all parameters
        snapshot = os.path.join(os.getcwd(), "snapshot.pt")
        if os.path.exists(snapshot):
            print(f'resuming: {snapshot}')
            payload = torch.load(snapshot)
            for k, v in payload.items():
                self.__dict__[k] = v
            # might not be necessary but might as well
            self.network.load_state_dict(torch.load(os.path.join(os.getcwd(), "recent.pt")))

    def isTerminal(self) -> bool:
        ''' determines if the current state of the agent is a terminal state
        '''
        return self.env.canGrasp()

    def train(self, num_steps: int, plotting_freq: int = 0) -> None:
        '''Train q-function for given number of environment steps using
        q-learning with e-greedy action selection

        Parameters
        ----------
        num_steps
            number of environment steps
        plotting_freq
            interval (in env steps) between plotting of training data, if 0
            then never plots.
        '''

        s = self.env.reset()

        pbar = tqdm(range(self.global_step, num_steps+1))
        for step in pbar:
            self.global_step += 1
            progress_fraction = step/(self.exploration_fraction*num_steps)
            self.epsilon = self.compute_epsilon(progress_fraction)
            a = self.select_action(s, self.epsilon)

            sp, r, done, info = self.env.step(a)
            self.episode_rewards += r

            self.buffer.add_transition(s=s, a=a, r=r, sp=sp, d=done)

            # optimize
            if len(self.buffer) > self.batch_size:
                loss = self.optimize()
                self.loss_data.append(loss)
                if len(self.loss_data) % self.target_network_update_freq == 0:
                    self.hard_target_update()

            s = sp.copy()
            if done:
                s = self.env.reset()
                self.rewards_data.append(self.episode_rewards)
                self.success_data.append(info['success'])

                self.episode_rewards = 0
                self.episode_count += 1

                avg_success = np.mean(self.success_data[-min(self.episode_count, 50):])
                avg_rewards = np.mean(self.rewards_data[-min(self.episode_count, 50):])
                pbar.set_description(f'Success = {avg_success:.1%}, Rewards = {avg_rewards}')

            if step % 10000 == 0:
            # pickle and plot every 10000 steps
                torch.save(self.network.state_dict(), os.path.join(os.getcwd(), "recent.pt"))
                snapshot = os.path.join(os.getcwd(), "snapshot.pt")
                keys_to_save = ['epsilon', 'buffer', 'network', 'target_network', 'global_step', 
                                'rewards_data', 'optim', 'success_data', 'loss_data', 
                                'episode_count', 'episode_rewards']
                payload = {k: self.__dict__[k] for k in keys_to_save}
                torch.save(payload, snapshot)
                plot_curves(self.rewards_data, self.success_data, self.loss_data)
#                 with torch.no_grad():
#                     actions = self.network(imgs)
                # actions = argmax2d(q_map_pred)
                # plot_predictions(imgs, q_map_pred, actions)
#                 plt.show()

        return self.rewards_data, self.success_data, self.loss_data

    def optimize(self) -> float:
        '''Optimizes q-network by minimizing td-loss on a batch sampled from
        replay buffer

        Returns
        -------
        mean squared td-loss across batch
        '''
        batch = self.buffer.sample(self.batch_size)
        s, a, r, sp, d = self.prepare_batch(*batch)

        q_all_pred = self.network(s)

        q_pred = torch.sum(torch.cat([torch.max(q_all_pred[:, i:i+3], dim=1)[0].unsqueeze(1)
                                      for i in range(0, 12, 3)], dim=1), 1)

        if self.update_method == 'standard':
            with torch.no_grad():
                q_all_pred_next = self.target_network(sp)
                q_next = torch.sum(torch.cat([torch.max(q_all_pred_next[:, i:i+3], dim=1)[0].unsqueeze(1)
                                              for i in range(0, 12, 3)], dim=1), 1)
                q_target = r + self.gamma * q_next * (1-d)

        # TODO implement
        # elif self.update_method == 'double':
        #     with torch.no_grad():
        #         q_map_next_est = self.target_network(sp)
        #         pred_act = self.network.predict(sp)
        #         q_next = q_map_next_est[np.arange(len(q_map_next_est)),
        #                                 0,
        #                                 pred_act[:, 0],
        #                                 pred_act[:, 1]]
        #         q_target = r + self.gamma * q_next * (1-d)

        assert q_pred.shape == q_target.shape
        self.optim.zero_grad()
        loss = self.network.compute_loss(q_pred, q_target)
        loss.backward()

        nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optim.step()

        return loss.item()

    def prepare_batch(self, s: np.ndarray, a: np.ndarray,
                      r: np.ndarray, sp: np.ndarray, d: np.ndarray,
                      ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        '''Converts components of transition from numpy arrays to tensors
        that are ready to be passed to q-network.  Make sure you send tensors
        to the right device!

        Parameters
        ----------
        s : array of state images, dtype=np.uint8, shape=(B, H, W, C)
        a : array of actions, dtype=np.int8, shape=(B, 2)
        r : array of rewards, dtype=np.float32, shape=(B,)
        sp : array of next state images, dtype=np.uint8, shape=(B, H, W, C)
        d : array of done flags, dtype=np.bool, shape=(B,)

        Returns
        ----------
        s : tensor of state images, dtype=torch.float32, shape=(B, C, H, W)
        a : tensor of actions, dtype=torch.long, shape=(B, 4)
        r : tensor of rewards, dtype=torch.float32, shape=(B,)
        sp : tensor of next state images, dtype=torch.float32, shape=(B, C, H, W)
        d : tensor of done flags, dtype=torch.float32, shape=(B,)
        '''
        s0 = torch.tensor(s, dtype=torch.float32,
                          device=self.device).permute(0, 3, 1, 2)
        s0 = torch.div(s0, 255)
        a0 = torch.tensor(a, dtype=torch.long, device=self.device)
        r0 = torch.tensor(r, dtype=torch.float32, device=self.device)
        sp0 = torch.tensor(sp, dtype=torch.float32,
                           device=self.device).permute(0, 3, 1, 2)
        sp0 = torch.div(sp0, 255)
        d0 = torch.tensor(d, dtype=torch.float32, device=self.device)

        return s0, a0, r0, sp0, d0

    def select_action(self, state: np.ndarray, epsilon: float = 0.) -> np.ndarray:
        '''Returns action based on e-greedy action selection.  With probability
        of epsilon, choose random action in environment action space, otherwise
        select argmax of q-function at given state

        Returns
        -------
        pixel action (px, py), dtype=int
        '''
        if np.random.random() < epsilon:
            return np.array(self.env.action_space.sample()) - 1
        else:
            return self.policy(state)

    def policy(self, state: np.ndarray) -> np.ndarray:
        '''Policy is the argmax over actions of the q-function at the given
        state. You will need to convert state to tensor on the device (similar
        to `prepare_batch`), then use `network.predict`.  Make sure to convert
        back to cpu before converting to numpy

        Returns
        -------
        pixel action (px, py); shape=(2,); dtype=int
        '''

        t_state = torch.tensor(state, dtype=torch.float32).unsqueeze(
            0)
        t_state = t_state.permute(0, 3, 1, 2)
        t_state = torch.div(t_state, 255)
        t_state = t_state.to(self.device)
        return self.network.predict(t_state).squeeze().cpu().numpy()

    def compute_epsilon(self, fraction: float) -> float:
        '''Calculate epsilon value based on linear annealing schedule

        Parameters
        ----------
        fraction
            fraction of exploration time steps that have been taken
        '''
        fraction = np.clip(fraction, 0., 1.)
        return (1-fraction) * self.initial_epsilon \
            + fraction * self.final_epsilon

    def hard_target_update(self):
        '''Update target network by copying weights from online network'''
        self.target_network.load_state_dict(self.network.state_dict())

    def save_network(self, dest: str = 'q_network.pt'):
        torch.save(self.network.state_dict(), dest)

    def load_network(self, model_path: str, map_location: str = 'cpu'):
        self.network.load_state_dict(torch.load(model_path,
                                                map_location=map_location))
        self.hard_target_update()


if __name__ == "__main__":
    env = HandoverGraspingEnv(render=False, sparse_reward=False)
    # get object to float

    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
    pb.resetDebugVisualizerCamera(cameraDistance=.4,
                                  cameraYaw=65.2,
                                  cameraPitch=-40.6,
                                  cameraTargetPosition=(.5, -0.36, 0.40))
    # TODO change render, device, and uncomment optimize
    agent = DQNAgent(env=env,
                     gamma=0.0,
                     learning_rate=1e-3,
                     buffer_size=6000,
                     batch_size=64,
                     initial_epsilon=0.5,  # TODO change hyperparams
                     final_epsilon=0.2,
                     update_method='standard',
                     exploration_fraction=0.9,
                     target_network_update_freq=500,
                     seed=1,
                     device='cuda')

    # TODO change save frequency, plot_curve, and this train num
    agent.train(1000000, 100)
