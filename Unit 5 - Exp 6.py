import numpy as np
import random

# Grid
GRID = 4
STATES = [(i, j) for i in range(GRID) for j in range(GRID)]
ACTIONS = ['U', 'D', 'L', 'R']
GOAL = (3, 3)

# Initial belief (uniform)
belief = {s: 1/len(STATES) for s in STATES}

# Transition model
def transition(state, action):
    x, y = state
    if action == 'U': x -= 1
    if action == 'D': x += 1
    if action == 'L': y -= 1
    if action == 'R': y += 1
    if 0 <= x < GRID and 0 <= y < GRID:
        return (x, y)
    return state

# Observation model (noisy sensor)
def observe(true_state):
    if random.random() < 0.7:
        return true_state
    return random.choice(STATES)

# Belief update
def update_belief(belief, action, observation):
    new_belief = {}
    for s in STATES:
        prob = 0
        for prev in STATES:
            if transition(prev, action) == s:
                prob += belief[prev]
        new_belief[s] = prob * (0.7 if s == observation else 0.3)
    total = sum(new_belief.values())
    for s in new_belief:
        new_belief[s] /= total
    return new_belief

# Simulation
true_state = (0, 0)
steps = 0

while true_state != GOAL and steps < 20:
    action = random.choice(ACTIONS)
    true_state = transition(true_state, action)
    obs = observe(true_state)
    belief = update_belief(belief, action, obs)
    steps += 1

print("Reached Goal:", true_state == GOAL)
print("Steps Taken:", steps)
