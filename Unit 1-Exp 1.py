import numpy as np

# 1. Initialize parameters
np.random.seed(0)
n_ads = 5
n_rounds = 10000
true_ctr = [0.05, 0.08, 0.12, 0.04, 0.10]  # true click-through rates

# ------------------ ε-Greedy ------------------
epsilon = 0.1
clicks_e = np.zeros(n_ads)
shows_e = np.zeros(n_ads)

for t in range(n_rounds):
    if np.random.rand() < epsilon:
        ad = np.random.randint(n_ads)
    else:
        ad = np.argmax(clicks_e / (shows_e + 1e-5))
    reward = np.random.rand() < true_ctr[ad]
    shows_e[ad] += 1
    clicks_e[ad] += reward

ctr_e = np.sum(clicks_e) / n_rounds

# ------------------ UCB ------------------
clicks_u = np.zeros(n_ads)
shows_u = np.zeros(n_ads)

for t in range(n_rounds):
    ucb_values = clicks_u / (shows_u + 1e-5) + np.sqrt(2 * np.log(t + 1) / (shows_u + 1e-5))
    ad = np.argmax(ucb_values)
    reward = np.random.rand() < true_ctr[ad]
    shows_u[ad] += 1
    clicks_u[ad] += reward

ctr_u = np.sum(clicks_u) / n_rounds

# ------------------ Thompson Sampling ------------------
success = np.ones(n_ads)
failure = np.ones(n_ads)

for t in range(n_rounds):
    samples = np.random.beta(success, failure)
    ad = np.argmax(samples)
    reward = np.random.rand() < true_ctr[ad]
    if reward:
        success[ad] += 1
    else:
        failure[ad] += 1

ctr_t = (np.sum(success) - n_ads) / n_rounds

# 3. Results
print("Average CTR after", n_rounds, "rounds:")
print("ε-Greedy CTR      :", round(ctr_e, 4))
print("UCB CTR           :", round(ctr_u, 4))
print("Thompson CTR      :", round(ctr_t, 4))
