import numpy as np
import jax

def use_single_gpu(DEVICE_ID):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

def print_device_info():
    print("Visible JAX devices:", jax.devices())
    print(f"Default JAX backend: {jax.lib.xla_bridge.get_backend().platform}\n")

def save_sol(U, p, sol_path):
    sol = np.concatenate((np.array([U.shape[0]]), np.array(U), np.array(p)))
    np.save(sol_path, sol)
    print('Saved solution to', sol_path)

def load_sol(sol_path):
    sol = np.load(sol_path)
    U = sol[1:(1 + int(sol[0]))]
    p = sol[(1 + int(sol[0])):]
    print('Loaded solution from', sol_path)
    return U, p

def save_iter_hist(gmres_list, cg_list, steps, path):
    hist = np.stack((gmres_list, cg_list, steps), axis=0)
    np.save(path, hist)
    print('Saved linear solver iteration counts to', path)

def load_iter_hist(path):
    hist = np.load(path)
    gmres_iters = hist[0, :]
    cg_iters = hist[1, :]
    steps = hist[2, :]
    print('Loaded iteration counts from', path)
    return gmres_iters, cg_iters, steps