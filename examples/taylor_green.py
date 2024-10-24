import jax.numpy as jnp
import jax_fem_cfd.setup.simulation as sim
from jax_fem_cfd.utils.running import *


def analytic_solution(node_coords, t, rho, mu):
    x = node_coords[:, 0]
    y = node_coords[:, 1]

    u = jnp.sin(x) * jnp.cos(y) * jnp.exp((-2*mu/rho) * t)
    v = -jnp.cos(x) * jnp.sin(y) * jnp.exp((-2*mu/rho) * t)
    p = (rho/4) * (jnp.cos(2*x) + jnp.cos(2*y)) * jnp.exp((-4*mu/rho) * t)

    return u, v, p

def expand_periodic(xs, nex, ney, ord):
    """ nex and ney are for the larger mesh (+1 element)
        xs is a nodal vector on the smaller mesh
    """
    if ord == 1:
        nnx = nex + 1
        nny = ney + 1
    elif ord == 2:
        nnx = 2*nex + 1
        nny = 2*ney + 1

    xs2d = xs.reshape(nny - 1, nnx - 1)
    x2d = jnp.zeros((nny, nnx))

    x2d = x2d.at[:-1, :-1].set(xs2d) # copy
    x2d = x2d.at[:-1, -1].set(xs2d[:, 0]) # periodic in x
    x2d = x2d.at[-1, :-1].set(xs2d[0, :]) # periodic in y
    x2d = x2d.at[-1, -1].set(xs2d[0, 0]) # periodic in both x and y

    return x2d.flatten()

def taylor_green_plots(U, p, node_coords, t_final, rho, mu, num_elem, nn, U_iters, P_iters, steps, ord, set_zeroP):
    import jax_fem_cfd.utils.plotting as plot_util
    import matplotlib.pyplot as plt

    u = U[:nn]
    v = U[nn:]

    u = expand_periodic(u, *num_elem, ord)
    v = expand_periodic(v, *num_elem, ord)
    p = expand_periodic(p, *num_elem, ord)

    u_exact, v_exact, p_exact = analytic_solution(node_coords, t_final, rho, mu)

    # compare both solutions with zero mean since P is unique up to a constant
    if set_zeroP:
        p_exact = p_exact - p_exact[0]
    else:
        p_exact = p_exact - jnp.mean(p_exact)

    u_exact = expand_periodic(u_exact, *num_elem, ord)
    v_exact = expand_periodic(v_exact, *num_elem, ord)
    p_exact = expand_periodic(p_exact, *num_elem, ord)

    x = node_coords[:, 0]
    y = node_coords[:, 1]

    x = expand_periodic(x, *num_elem, ord)
    y = expand_periodic(y, *num_elem, ord)

    if ord == 1:
        nnx, nny = num_elem[0] + 1, num_elem[1] + 1
    elif ord == 2:
        nnx, nny = 2*num_elem[0] + 1, 2*num_elem[1] + 1

    tri = plot_util.get_2d_tri(x, y)

    # plot_util.plot_contour(tri, u_exact, x, y, u_exact, v_exact, nnx, nny, 'u exact', quiv=True)
    # plot_util.plot_contour(tri, p_exact, x, y, u_exact, v_exact, nnx, nny, 'p exact', quiv=False)
    # plot_util.plot_image(u_exact, x, y, u_exact, v_exact, nnx, nny, 'u exact', quiv=True)
    plot_util.plot_image(p_exact, x, y, u_exact, v_exact, nnx, nny, 'p exact', quiv=False)

    # plot_util.plot_contour(tri, u, x, y, u, v, nnx, nny, 'u', quiv=True)
    # plot_util.plot_contour(tri, p, x, y, u, v, nnx, nny, 'p', quiv=False)
    # plot_util.plot_image(u, x, y, u, v, nnx, nny, 'u', quiv=True)
    plot_util.plot_image(p, x, y, u, v, nnx, nny, 'p', quiv=False)

    # plot_util.plot_contour(tri, u - u_exact, x, y, u, v, nnx, nny, 'u error', quiv=False)
    # plot_util.plot_contour(tri, p - p_exact, x, y, u, v, nnx, nny, 'p error', quiv=False)
    # plot_util.plot_image(u - u_exact, x, y, u, v, nnx, nny, 'u error', quiv=False)
    plot_util.plot_image(p - p_exact, x, y, u, v, nnx, nny, 'p error', quiv=False)

    # x2d, y2d, u2d_exact, v2d_exact = plot_util.get_2d_arrays(x, y, u_exact, v_exact, nnx, nny)
    # plot_util.plot_streamlines(x2d, y2d, u2d_exact, v2d_exact, 'Streamlines', figsize=(6,6))

    # plot_util.plot_solve_iters(U_iters, P_iters, steps)

    plot_util.show_plots()



if __name__ == "__main__":
    # use_single_gpu(DEVICE_ID=7)
    print_device_info()
    jax.config.update("jax_enable_x64", True)

    config = sim.Config(mesh="simple",
                        solver="standard",
                        mesh_config=sim.SimpleMeshConfig(
                            num_elem=[64, 64], # x elements, y elements
                            domain_size=[2*jnp.pi, 2*jnp.pi], # x length, y length
                            dirichlet_faces=[], 
                            symmetry_faces=[],
                            outlet_faces=[],
                            periodic=True), # all faces periodic, overwrites other conditions
                        solver_config=sim.StandardSolverConfig(
                            ndim=2,
                            shape_func_ord=1,
                            streamline_diff=False,
                            crank_nicolson=True,
                            solver_tol=1e-6,
                            set_zeroP=False,
                            pressure_precond=None,
                            final_multigrid_mesh=[2, 2]))

    precompute, timestep, node_coords, surfnodes, h, nn, gamma_d, U_d = sim.setup(config)
    print('Mesh and solver setup complete')
    print(f'Using a {config.mesh_config.num_elem[0]}x{config.mesh_config.num_elem[1]} mesh')

    # pressure decays to max 0.067 by t=5
    t_final = 1.6 # final time of simulation
    Cmax = 1 # for CFL condition
    rho, mu = 1, 0.1
    # dt = Cmax * h # velocity scale is 1

    dt = 0.01

    Re = int(rho * config.mesh_config.domain_size[0] / mu) # velocity scale is 1
    print('Re =', Re)
    print('dt =', dt)

    u0, v0, P0 = analytic_solution(node_coords, 0, rho, mu)
    U0 = jnp.concatenate((u0, v0))

    if config.solver_config.set_zeroP:
        P0 = P0 - P0[0] # shift such that P=0  at node 0
    else:
        P0 = P0 - jnp.mean(P0) # ensure that the initial guess has zero mean (it does)

    U, p, U_iters, P_iters, steps = sim.timestep_loop(t_final, dt, 20, 20, precompute, timestep, 
                                                      rho, mu, U_d, U0, P0)
    
    taylor_green_plots(U, p, node_coords, t_final, rho, mu, config.mesh_config.num_elem, nn, 
                       U_iters, P_iters, steps, config.solver_config.shape_func_ord, config.solver_config.set_zeroP)
    

