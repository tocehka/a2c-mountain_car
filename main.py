from r_utils import GameEnv
from nn_models import Actor, Critic
import torch.optim as optim
from torch import Tensor, distributions
import numpy as np
import matplotlib.pyplot as plt

class GameEmulator:
    def __init__(self, epochs, default_reward, reward_func):
        self.epochs = epochs
        self.max_reward = default_reward
        self.overall_reward = []

        self.game = GameEnv("MountainCar-v0")
        states_dim, actions_dim = self.game.get_init_params()

        self.actor = Actor(actions_dim, states_dim[0])
        self.critic = Critic(states_dim[0])

        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=0.0009)
        self.critic_optim = optim.Adam(params=self.critic.parameters(), lr=0.0009)

        # self.best_model = None

        self.reward_func = reward_func

    def get_distributed_action(self, state):
        softmax_probs = self.actor(Tensor(state))
        d = distributions.Categorical(probs=softmax_probs)
        return d.sample(), d

    def A_func(self, done, state, prev_state, gamma=0.98):
        return (1 - done) * gamma * self.critic(Tensor(state)) - self.critic(Tensor(prev_state))

    def set_best(self, last_step=-3):
        if self.max_reward < np.mean(self.overall_reward[last_step:]):
            self.max_reward = np.mean(self.overall_reward[last_step:])
            # self.best_model = self.actor
    
    def train_model(self, final_reward=None):
        self.actor.train()
        for i in range(self.epochs):
            print(f"----------- Start {i + 1} train epoch -----------")
            
            done = False
            epoch_reward = 0
            prev_state = self.game.env_reset()
            while not done:
                action, distribution = self.get_distributed_action(prev_state)

                state, reward, done, _ = self.game.env_step(action.detach().data.numpy())
                reward = reward + self.reward_func(state, prev_state)
                
                A = reward + self.A_func(done, state, prev_state)
                prev_state = state
                
                critic_loss = A.pow(2).mean()
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                actor_loss = -distribution.log_prob(action) * A.detach()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                epoch_reward += reward

            self.overall_reward.append(epoch_reward)
            self.set_best()
            if final_reward and epoch_reward > final_reward:
                print(f"----------- Model achieve the maximum best result - {epoch_reward} -----------")
                return

            print(f"----------- At the end of {i + 1} epoch reward = {epoch_reward} -----------")

    def play(self):
        self.actor.eval()
        while True:
            state = self.game.env_reset()
            done = False
            while not done:
                action, _ = self.get_distributed_action(state)
                state, _, done, _ = self.game.env_step(action.detach().data.numpy())
                self.game.env_render()

    def plot(self):
        plt.plot([i for i,v in enumerate(self.overall_reward)], self.overall_reward)
        plt.xlabel("Number of epochs")
        plt.ylabel("Reward")
        plt.show()

if __name__ == "__main__":
    def reward_func(s2, s1, accelerator=315):
        return accelerator * (abs(s2[1]) - abs(s1[1]))

    game = GameEmulator(300, -200, reward_func)
    game.train_model()
    game.plot()
    game.play()

