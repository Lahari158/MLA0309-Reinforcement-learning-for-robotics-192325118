import random

# -------------------------------
# Environment (Streaming Service)
# -------------------------------
class RecommenderEnv:
    def __init__(self):
        self.state = 0.0

    def reset(self):
        # user preference (0 to 1)
        self.state = random.random()
        return self.state

    def step(self, action):
        # reward based on how close recommendation is to preference
        reward = 1 - abs(self.state - action)

        # user preference changes after watching
        self.state = random.random()
        return self.state, reward


# -------------------------------
# Actor Network (Policy)
# -------------------------------
class Actor:
    def __init__(self):
        self.weight = random.uniform(0.0, 1.0)

    def act(self, state):
        # generate continuous recommendation
        action = state * self.weight

        # clip action between 0 and 1
        action = max(0, min(1, action))
        return action


# -------------------------------
# Critic Network (Value Function)
# -------------------------------
class Critic:
    def __init__(self):
        self.value = random.uniform(0.0, 1.0)

    def update(self, reward):
        # TD-style update
        self.value = 0.9 * self.value + 0.1 * reward
        return self.value


# -------------------------------
# DDPG-Style Agent
# -------------------------------
class DDPGAgent:
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.learning_rate = 0.05

    def train_step(self, state):
        action = self.actor.act(state)
        next_state, reward = env.step(action)

        critic_value = self.critic.update(reward)

        # policy improvement
        self.actor.weight += self.learning_rate * reward
        self.actor.weight = max(0, min(1, self.actor.weight))

        return next_state, reward, critic_value


# -------------------------------
# Main Training Loop
# -------------------------------
env = RecommenderEnv()
agent = DDPGAgent()

episodes = 10
steps_per_episode = 5

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(steps_per_episode):
        state, reward, value = agent.train_step(state)
        total_reward += reward

    print("Episode:", ep + 1,
          "| Total Reward:", round(total_reward, 2),
          "| Actor Weight:", round(agent.actor.weight, 2),
          "| Critic Value:", round(agent.critic.value, 2))
