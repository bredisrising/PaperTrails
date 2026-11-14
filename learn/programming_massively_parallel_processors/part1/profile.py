import os
import time
import matplotlib.pyplot as plt
import numpy as np
import subprocess


N = 29

orders = [i for i in range(N)]
n_elements = [2**order for order in orders]
times = []

block_dims = [2**i for i in range(4, 9)]

# compile once more
subprocess.run(
    ['nvcc', 'vadd.cu', '-arch=sm_89']
)

with open("results", "w") as f:
    for block_dim in block_dims:
        times.append([])
        for n in n_elements:
            vec_add = subprocess.run(
                ["./a.out", str(n), str(block_dim)],
                capture_output=True,
                text=True
            )

            output = vec_add.stdout.split(",")
            times[-1].append(float(output[1]))

            f.write(f"elements: {n}, block_dim: {block_dim}, time_taken: {float(output[1])}\n")
            f.flush()



for i, time_sub in enumerate(times):
    plt.plot(orders, time_sub, label=str(block_dims[i]))
plt.legend()
plt.show()





