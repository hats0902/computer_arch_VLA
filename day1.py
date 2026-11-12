import numpy as np, time
a = np.random.rand(8000, 8000)
t = time.perf_counter(); a.sum(axis=1); print("行方向:", time.perf_counter() - t)
t = time.perf_counter(); a.sum(axis=0); print("列方向:", time.perf_counter() - t)
