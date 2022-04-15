import os
from typing import Optional, Union, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import gym
from tqdm import tqdm
import pybullet as pb
import matplotlib.pyplot as plt

from value_critic import ValueNetwork
from continuous_actor import ContinuousActorNetwork
from conv_nets import CNN, R2EquiCNN
from utils import ReplayBuffer, plot_curves
from grasping_env import HandoverGraspingEnv


class ActorCriticAgent:
    def __init__(self,
                 env: HandoverGraspingEnv,
                 gamma: float,
                 learning_rate: float,
                 buffer_size: int,
                 batch_size: int,
                 update_method: str = 'standard',
                 exploration_fraction: float = 0.9,
                 target_network_update_freq: int = 1000,
                 seed: int = 0,
                 device: Union[str, torch.device] = 'cpu',
                 ) -> None:
        self.env = env

        self.gamma = gamma
        self.batch_size = batch_size
        # self.initial_epsilon = initial_epsilon
        # self.final_epsilon = final_epsilon
        self.exploration_fraction = exploration_fraction

        self.target_network_update_freq = target_network_update_freq
        self.update_method = update_method

        self.buffer = ReplayBuffer(buffer_size,
                                   env.observation_space.shape,
                                   (len(env.action_space),))

        self.device = device
        img_shape = (3, self.env.img_size, self.env.img_size)
        self.image_feature_extractor = CNN(img_shape)

        self.policy_network = ContinuousActorNetwork(self.image_feature_extractor)
        self.value_network = ValueNetwork(self.image_feature_extractor)
        self.policy_target_network = ContinuousActorNetwork(self.image_feature_extractor)
        self.value_target_network = ValueNetwork(self.image_feature_extractor)

        self.hard_target_update()

        self.policy_opt = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_opt = torch.optim.Adam(self.value_network.parameters(), lr=learning_rate)


        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    def isTerminal(self) -> bool:
        ''' determines if the current state of the agent is a terminal state
        '''
        return self.env.canGrasp()

    def train(self, num_steps: int, plotting_freq: int = 0):
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
        rewards_data = []
        success_data = []
        critic_loss_data = []
        policy_loss_data = []

        episode_count = 0
        episode_rewards = 0
        s = self.env.reset()

        pbar = tqdm(range(1, num_steps+1))
        for step in pbar:
            progress_fraction = step/(self.exploration_fraction*num_steps)
            a = self.select_action(s)

            sp, r, done, info = self.env.step(a)
            episode_rewards += r

            # print(r)
            # print(self.env.getEEPosOrn()[0])

            self.buffer.add_transition(s=s, a=a, r=r, sp=sp, d=done)

            # optimize
            if len(self.buffer) > self.batch_size:
                
                batch = self.buffer.sample(self.batch_size)
                batch = self.prepare_batch(*batch)
                critic_loss = self.optimize_critic(*batch)
                policy_loss = self.optimize_policy(*batch)

                critic_loss_data.append(critic_loss)
                policy_loss_data.append(policy_loss)
                if len(policy_loss_data) % self.target_network_update_freq == 0:
                    self.hard_target_update()

            s = sp.copy()
            if done:
                s = self.env.reset()
                torch.save(self.value_network.state_dict(),
                           os.path.join(os.getcwd(), "recent_critic.pt"))
                torch.save(self.policy_network.state_dict(),
                           os.path.join(os.getcwd(), "recent_actor.pt"))
                rewards_data.append(episode_rewards)
                success_data.append(info['success'])

                episode_rewards = 0
                episode_count += 1

                avg_success = np.mean(success_data[-min(episode_count, 50):])
                pbar.set_description(f'Success = {avg_success:.1%}')

            if plotting_freq > 0 and step % plotting_freq == 0:
                plot_curves(rewards_data, success_data, critic_loss_data, policy_loss_data)
                plt.show()

        return rewards_data, success_data, loss_data

    def playout(self, num_steps):
        
        s = self.env.reset()
        step = 0
        done = 0
        while step < num_steps and not done:
            
            a = self.select_action(s)

            sp, r, done, info = self.env.step(a)
            # print(r)
            # print(self.env.getEEPosOrn()[0])

            s = sp.copy()

    def optimize_policy(self, s, a, r, sp, d):
        self.policy_opt.zero_grad()
        score = self.policy_network.compute_score(s, a, self.value_network(sp))
        score.sum().backward()
        self.policy_opt.step()

    def optimize_critic(self, s, a, r, sp, d) -> float:
        '''Optimizes q-network by minimizing td-loss on a batch sampled from
        replay buffer

        Returns
        -------
        mean squared td-loss across batch
        '''
        # def q_axis_sum(network_pred):
        #     '''computes the sum of the optimal action for each dimension of the action space (across a batch)
        #     Input
        #     -----
        #         12-vector of axis-wise q-values
        #     Return
        #     -----
        #         Q-value of the state is the sum of the axis-wise 4 max actions
        #     '''
        #     return torch.sum(torch.cat([torch.max(network_pred[:, i:i+3], dim=1)[0].unsqueeze(1)
        #                               for i in range(0, 12, 3)], dim=1), 1)

        # batch = self.buffer.sample(self.batch_size)
        # s, a, r, sp, d = self.prepare_batch(*batch)

        r = r.unsqueeze(1)
        d = d.unsqueeze(1)

        q_pred = self.value_network(s)

        if self.update_method == 'standard':
            with torch.no_grad():
                q_pred_next = self.value_target_network(sp)
                # q_next = self.value_network(q_pred_next)
                q_target = r + self.gamma * q_pred_next * (1-d)

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

        assert q_pred.shape == q_target.shape, f"r: {r.shape}, q_target: {q_target.shape}, q_pred_next: {q_pred_next.shape}"
        self.value_opt.zero_grad()
        loss = self.value_network.compute_loss(q_pred, q_target)
        loss.backward()

        #NOTE: what is the purpose of this clipping?
        nn.utils.clip_grad_norm_(self.value_network.parameters(), 10)
        self.value_opt.step()

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

    def select_action(self, state: np.ndarray) -> np.ndarray:
        '''Returns action based on stochastic sampling action selection. Given a policy function
        output of a mean and variance, sample an action from the normal distribution for each axis
        in the action space.
        Returns
        -------
        4-vector of discrete actions, each in {-1, 0, 1}
        '''
        action = np.zeros(4)
        action_dist = self.policy(state) 
        mu, var = action_dist[::2], action_dist[1::2]
        for i in range(4):
            action[i] = np.random.normal(mu[i], var[i]**2)


        # print(action)
        return action

    def policy(self, state: np.ndarray) -> np.ndarray:
        '''Policy is the argmax over actions of the q-function at the given
        state. You will need to convert state to tensor on the device (similar
        to `prepare_batch`), then use `network.predict`.  Make sure to convert
        back to cpu before converting to numpy

        Returns
        -------

        '''

        t_state = torch.tensor(state, dtype=torch.float32).unsqueeze(
            0)
        t_state = t_state.permute(0, 3, 1, 2)
        t_state = torch.div(t_state, 255)
        t_state = t_state.to(self.device)
        action = self.policy_network.predict(t_state).squeeze().cpu().numpy()
        return action

    def hard_target_update(self):
        '''Update target network by copying weights from online network'''
        self.policy_target_network.load_state_dict(self.policy_network.state_dict())
        self.value_target_network.load_state_dict(self.value_network.state_dict())

    def save_network(self, dests: Tuple[str, str] =('policy_network.pt', 'q_network.pt')):
        assert len(dests) == 2
        torch.save(self.policy_network.state_dict(), dests[0])
        torch.save(self.value_network.state_dict(), dests[1])

    def load_network(self, model_path: Tuple[str, str]= ('policy_network.pt', 'q_network.pt'), map_location: str = 'cpu'):
        self.policy_network.load_state_dict(torch.load(model_path[0],
                                                map_location=map_location))
        self.value_network.load_state_dict(torch.load(model_path[1],
                                                map_location=map_location))
        self.hard_target_update()


if __name__ == "__main__":
    env = HandoverGraspingEnv(render=False, sparse_reward=False)
    # TODO questions reward logging?
    # why is atol not working for all close? even with 0, still returning true eventually
    # get object to float

    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
    pb.resetDebugVisualizerCamera(cameraDistance=.4,
                                  cameraYaw=65.2,
                                  cameraPitch=-40.6,
                                  cameraTargetPosition=(.5, -0.36, 0.40))

    agent = ActorCriticAgent(env=env,
                     gamma=0.5,
                     learning_rate=1e-3,
                     buffer_size=4000,
                     batch_size=8,
                     update_method='standard',
                     exploration_fraction=0.9,
                     target_network_update_freq=250,
                     seed=1,
                     device='cpu')

    agent.train(10000, 100)
