import random

# -------------------------------
# Simple Highway Environment
# -------------------------------
class HighwayEnv:
    def _init_(self):
        self.reset()

    def reset(self):
        self.speed = 1
        self.distance = 10
        return (self.speed, self.distance)

    def step(self, action):
        # actions: 0=maintain, 1=accelerate, 2=brake
        if action == 1:
            self.speed += 1
        elif action == 2:
            self.speed = max(0, self.speed - 1)

        self.distance -= self.speed

        reward = 1
        done = False

        if self.distance <= 0:
            reward = -100
            done = True

        return (self.speed, self.distance), reward, done

# -------------------------------
# Q-Learning Agent
# -------------------------------
actions = [0, 1, 2]
Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2

def get_q(state, action):
    return Q.get((state, action), 0)

def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)
    qs = [get_q(state, a) for a in actions]
    return actions[qs.index(max(qs))]

def update_q(state, action, reward, next_state):
    old_q = get_q(state, action)
    future_q = max(get_q(next_state, a) for a in actions)
    Q[(state, action)] = old_q + alpha * (reward + gamma * future_q - old_q)

# -------------------------------
# Training (10 Episodes)
# -------------------------------
env = HighwayEnv()

for episode in range(10):
    state = env.reset()
    total_reward = 0

    for step in range(50):
        action = choose_action(state)
        next_state, reward, done = env.step(action)
        update_q(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        if done:
            break

    print("Episode:", episode + 1, "Reward:", total_reward)

print("Training completed")
