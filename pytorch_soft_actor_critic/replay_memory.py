import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0



    def add_noise(self, state):
        # print("state before noise:", state)
        noise_level = np.random.uniform(0.01, 0.15)

        noise = np.random.normal(0, noise_level, size=state.shape)
        sn = state + noise
        # print("state after noise: ", sn)

        return sn
    

    def push(self, state, action, reward, next_state, done, cost):
        # assert state is not None, "State is None"
        # assert action is not None, "Action is None"
        # assert reward is not None, "Reward is None"
        # assert next_state is not None, "Next state is None"
        # assert done is not None, "Done is None"
        # assert cost is not None, "Cost is None"

        state = self.add_noise(state)
        next_state = self.add_noise(next_state)


        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done,
                                      cost)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, get_cost=False):

        if len(self.buffer) < batch_size:
            raise ValueError("Not enough elements in the buffer to sample the requested batch size")
        
        # print(f"Sampling {batch_size} elements from buffer with {len(self.buffer)} elements.")


        batch = random.sample(self.buffer, batch_size)

        # for i, item in enumerate(batch):
        #     if item is None:
        #         print(f"None item found in batch at index {i}")
        #     else:
        #         print(f"Batch item {i}: {item}")


        state, action, reward, next_state, done, cost =  map(np.stack, zip(*batch))
        if get_cost:
            return state, action, reward, next_state, done, cost
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
