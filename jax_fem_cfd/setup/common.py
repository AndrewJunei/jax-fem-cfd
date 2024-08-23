import jax.numpy as jnp
import time

def set_face_velocity(U_d, surfnodes, gamma_d, values, nn, var):
    """ Set velocity values for specified surface nodes 
        var = 1: set u, var = 2: set v
    """
    node_indices = jnp.searchsorted(gamma_d, surfnodes + nn*(var-1))

    return U_d.at[node_indices].set(values)

def format_boundaries(gamma_d, gamma_p, nn, ndim):
    """ gamma_d is a 1D array indicating which nodes are dirichlet  
    """
    gamma_d = jnp.sort(jnp.unique(gamma_d)) 
    gamma_d = jnp.concatenate([gamma_d + nn*i for i in range(ndim)]) 
    U_d = jnp.zeros_like(gamma_d, dtype=float) # initialization
    not_gamma_p = jnp.setdiff1d(jnp.arange(nn), gamma_p) # set minus: all nodes - gamma_p

    return gamma_d, U_d, not_gamma_p

def timestep_loop(t_final, dt, save_iter, timestep_func, U, P):
    step, t = 0, 0.0
    gmres_list, cg_list, step_list = [], [], []

    compile_start = time.perf_counter()
    _, _, _, _ = timestep_func(U, U, P)
    compile_time = time.perf_counter() - compile_start
    print(f'Run to compile took {(compile_time * 1e3):.4f} ms')

    total_start = time.perf_counter()
    while t < t_final: # no condition to exit loop if solution diverges
        if step > 0:
            U_n_1 = U_n
        else:
            U_n_1 = U

        U_n, P_n = U, P

        start = time.perf_counter()
        U, P, gmres_iters, cg_iters = timestep_func(U_n, U_n_1, P_n)
        end = time.perf_counter()

        t += dt
        step += 1

        if step % save_iter == 0:
            gmres_list.append(gmres_iters)
            cg_list.append(cg_iters)
            step_list.append(step)

        diff = (jnp.linalg.norm(jnp.concatenate((U, P)) - jnp.concatenate((U_n, P_n)), 2) / 
                    jnp.linalg.norm(jnp.concatenate((U, P)), 2))

        print(f'End step {step}, time {t:.2f}, took {((end - start) * 1000):.2f} ms, diff = {diff:.6f}')

    total_end = time.perf_counter()
    print(f'\nTotal timestepping time: {(total_end - total_start):.4f} s')

    gmres_list = jnp.array(gmres_list)
    cg_list = jnp.array(cg_list)
    step_list = jnp.array(step_list)

    return U, P, gmres_list, cg_list, step_list