import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-.5 * ((x - mu) / sigma)**2))

x = np.arange(-4, 6, .1)

P = 0.5 * gaussian(x, -2, 0.6) + 0.5 * gaussian(x, 3, 0.8)

# KL Divergence

# PQ = np.sum(P * np.log(P / (Q + 1e-12))) * (x[1] - x[0])
# QP = np.sum(Q * np.log(Q / (P + 1e-12))) * (x[1] - x[0])

# print(PQ)
# print(QP)

def objective(vars):
    Q = gaussian(x, vars[0], vars[1])
    # return np.sum(P * np.log(P / (Q + 1e-12))) * .1 # forward KL
    return np.sum(Q * np.log(Q / (P + 1e-12))) * .1


start = [.1, 1]
result = minimize(objective, start)

print(result)
  
Q_og = gaussian(x, start[0], start[1]) 
Q = gaussian(x, result.x[0], result.x[1])

plt.plot(x, P, label="P")
plt.plot(x, Q_og, label="Q Original")
plt.plot(x, Q, label="Q Optimized")
# plt.plot(x, P * np.log(P / Q))
# plt.plot(x, Q * np.log(Q / P))
plt.legend()
plt.show()
