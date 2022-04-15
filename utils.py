from typing import Tuple, Dict, List
import h5py
import numpy as np
import matplotlib.pyplot as plt


def plot_curves(rewards, success, critic_loss, actor_loss):
    f, axs = plt.subplots(1, 4, figsize=(7,2.5))
    W = 50 # smoothing window

    [a.clear() for a in axs]
    axs[0].plot(np.convolve(rewards, np.ones(W)/W, 'valid'))
    axs[0].set_xlabel('episodes')
    axs[0].set_ylabel('episodic rewards')

    axs[1].plot(np.convolve(success, np.ones(W)/W, 'valid'))
    axs[1].set_xlabel('episodes')
    axs[1].set_ylabel('success rate')
    axs[1].set_ylim(0, 1)

    if len(critic_loss) > 0:
        axs[2].plot(np.convolve(critic_loss, np.ones(W)/W, 'valid'))
    axs[2].set_xlabel('opt steps')
    axs[2].set_ylabel('value est. td-loss')
    plt.tight_layout()

    if len(actor_loss) > 0:
        axs[2].plot(np.convolve(actor_loss, np.ones(W)/W, 'valid'))
    axs[2].set_xlabel('opt steps')
    axs[2].set_ylabel('policy gradient loss score')
    plt.tight_layout()


class ReplayBuffer:
    def __init__(self,
                 size: int,
                 state_shape: Tuple[int],
                 action_shape: Tuple[int],
                ) -> None:
        '''Replay Buffer that stores transitions (s,a,r,sp,d) and can be sampled
        for random batches of transitions

        Parameters
        ----------
        size
            number of transitions that can be stored in buffer at a time (beyond
            this size, new transitions will overwrite old transitions)
        state_shape
            shape of state image (H,W,C), needed to initialize data array
        action_shape
            shape of action (2,) since action is <px, py>, dtype=int
        '''
        self.data = {'state' : np.zeros((size, *state_shape), dtype=np.uint8),
                     'action' : np.zeros((size, *action_shape), dtype=np.int8),
                     'next_state' : np.zeros((size, *state_shape), dtype=np.uint8),
                     'reward' : np.zeros((size), dtype=np.float32),
                     'done' : np.zeros((size), dtype=np.bool8),
                    }
        self.length = 0
        self.size = size
        self._next_idx = 0

    def add_transition(self, s: np.ndarray, a: np.ndarray, r: float,
                       sp: np.ndarray, d: bool) -> None:
        '''Add single transition to replay buffer, overwriting old transitions
        if buffer is full
        '''
        self.data['state'][self._next_idx] = s
        self.data['action'][self._next_idx] = a
        self.data['reward'][self._next_idx] = r
        self.data['next_state'][self._next_idx] = sp
        self.data['done'][self._next_idx] = d

        self.length = min(self.length + 1, self.size)
        self._next_idx = (self._next_idx + 1) % self.size

    def sample(self, batch_size: int) -> Tuple:
        '''Sample a batch from replay buffer.

        Parameters
        ----------
        batch_size
            number of transitions to sample
        '''
        idxs = np.random.randint(self.length, size=batch_size)

        keys = ('state', 'action', 'reward', 'next_state', 'done')
        s, a, r, sp, d = [self.data[k][idxs] for k in keys]

        return s, a, r, sp, d

    def load_transitions(self, hdf5_file: str):
        '''loads pre-collected transitions into buffer. pybullet can be quite
        slow so I am giving you transitions to prepopulate the buffer with
        '''
        with h5py.File(hdf5_file, 'r') as hf:
            states = np.array(hf['states'])
            actions = np.array(hf['actions'])
            rewards = np.array(hf['rewards'])
            next_states = np.array(hf['next_states'])
            dones = np.array(hf['dones'])

        for i in range(len(states)):
            self.add_transition(states[i], actions[i], rewards[i],
                                next_states[i], dones[i])

    def __len__(self):
        return self.length


