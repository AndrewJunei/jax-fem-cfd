import jax.numpy as jnp
from jax_fem_cfd.setup.regular_mesh import create_mesh_2d
from jax_fem_cfd.setup.common import set_face_velocity, format_boundaries, timestep_loop
from jax_fem_cfd.solvers.standard_2d import setup_solver, four_step_timestep
from jax_fem_cfd.utils.running import *


def step_setup(num_elem, domain_size, Re, Cmax):
    """ 2D flow over a backward facing step. The step takes up the bottom half of the total 
        domain height. In this setup, the left boundary marks the end of the step. For this to 
        work, the x velocity takes on a fully developed parabolic profile on the top half of the 
        left boundary and is 0 below. There are walls on the top and bottom boundaries, and an
        outlet on the right.
    """
    rho = 1
    u_avg = 1
    mu = rho * u_avg * domain_size[1] / Re
    print('Re =', Re)

    h = domain_size[0] / num_elem[0]
    Pe = u_avg * h / (2*mu / rho)
    print('Avg global Pe = ', Pe)

    dt = Cmax * h / u_avg
    print('dt =', dt)

    node_coords, nconn, surfnodes, bconns, nvecs = create_mesh_2d(*num_elem, *domain_size)
    nn = node_coords.shape[0]
    print(f'Using a {num_elem[0]}x{num_elem[1]} mesh')

    bconn = jnp.concatenate([bconns[0], bconns[2], bconns[3]], axis=0) # exclude outlet
    nvec = jnp.concatenate([nvecs[0], nvecs[2], nvecs[3]], axis=0) # exclude outlet
    gamma_d = jnp.concatenate([surfnodes[0], surfnodes[2], surfnodes[3]]) # exclude outlet
    gamma_p = surfnodes[1][1:-1] # outlet (check corners)

    gamma_d, U_d, not_gamma_p = format_boundaries(gamma_d, gamma_p, nn, 2)

    def parabolic_inlet(node_coords, left_nodes):
        """ Set the parabolic x velocity profile for the inlet 
        """
        inlet_velocity = jnp.zeros_like(left_nodes)
        
        y_coords = node_coords[left_nodes, 1]
        mask = (y_coords >= 0.5) & (y_coords <= 1.0)
        inlet_velocity = jnp.where(mask, 24 * (y_coords - 0.5) * (0.5 - (y_coords - 0.5)), 0.0)

        return inlet_velocity
    
    left_nodes = surfnodes[3]
    left_velocity = parabolic_inlet(node_coords, left_nodes) # left face
    U_d = set_face_velocity(U_d, left_nodes, gamma_d, left_velocity, nn, 1) # rest are 0

    U = jnp.zeros(2*nn) # initial condition
    P = jnp.zeros(nn)
    print('Boundary conditions have been set\n')

    wq, N, Nderiv, M, Le, Ge, F, L_diag = setup_solver(node_coords, nconn, bconn, nvec, 
                                                       rho, 
                                                       not_gamma_p, gamma_d, U_d)
    
    mask = F != 0
    F_nonzero = F[mask]
    F_idx = jnp.arange(F.shape[0])[mask]
    F = F_nonzero

    def timestep_func(U_n, U_n_1, P_n):
        return four_step_timestep(U_n, U_n_1, P_n, M, Le, Ge, rho, mu, h, dt, nconn, 
                                  wq, N, Nderiv, 
                                  gamma_d, U_d, not_gamma_p, F, F_idx, L_diag, nn)
    
    return U, P, timestep_func, dt, node_coords


def step_plots(node_coords, num_elem, U, p, Re, gmres_list, cg_list, step_list):
    import jax_fem_cfd.utils.plotting as plot_util
    import matplotlib.pyplot as plt

    nn = node_coords.shape[0]
    nnx, nny = num_elem[0] + 1, num_elem[1] + 1

    x = node_coords[:, 0]
    y = node_coords[:, 1]
    u = U[:nn]
    v = U[nn:]

    x2d, y2d, u2d, v2d = plot_util.get_2d_arrays(x, y, u, v, nnx, nny)
    tri = plot_util.get_2d_tri(x, y)

    def interpolate_at_x(x_unique, u2d, target_x):
        dx = x_unique[1] - x_unique[0]
        index = (target_x - x_unique[0]) / dx
        index_low = jnp.floor(index).astype(int)
        index_high = index_low + 1
        weight_high = index - index_low
        
        u_low = u2d[:, index_low]
        u_high = u2d[:, index_high]
        
        return u_low + weight_high * (u_high - u_low)
    
    ygartling = jnp.linspace(0.0, 1.0, 21)
    ugartling7 = jnp.array([0., 0.232, 0.428, 0.613, 0.792, 0.948, 1.062, 1.118, 1.105, 1.024, 0.885,
                             0.709, 0.522, 0.349, 0.204, 0.092, 0.015, -0.032, -0.049, -0.038, 0.])
    vgartling7 = jnp.array([0., -0.118, -0.504, -1.000, -1.423, -1.748, -1.917, -1.925, -1.778,
                            -1.507, -1.165, -0.823, -0.544, -0.362, -0.268, -0.225, -0.193, -0.147,
                            -0.086, -0.027, 0.]) * 1e-2
    
    x1d = np.linspace(x.min(), x.max(), nnx)
    y1d = np.linspace(y.min(), y.max(), nny)
    u_at_target_x = interpolate_at_x(x1d, u2d, 7.0)
    v_at_target_x = interpolate_at_x(x1d, v2d, 7.0)

    plot_util.plot_contour(tri, u, x, y, u, v, nnx, nny, 'x Velocity', 
                           figsize=(12, 4), c_shrink=0.4, quiv=False)
    
    if Re == 800:
        x1d = np.linspace(x.min(), x.max(), nnx)
        y1d = np.linspace(y.min(), y.max(), nny)
        u_at_target_x = interpolate_at_x(x1d, u2d, 7.0)
        v_at_target_x = interpolate_at_x(x1d, v2d, 7.0)
        reflbl = 'Gartling Re=800'
        lbl = f'{num_elem[0]}x{num_elem[1]} Mesh'
        plot_util.compare_double(ygartling, ugartling7, reflbl, y1d, u_at_target_x, lbl, 
                                ygartling, vgartling7, reflbl, y1d, v_at_target_x, lbl,
                                'x Velocity at x=7.0', 'y Velocity at x=7.0', 'y', 'y')

    plot_util.plot_streamlines(x2d, y2d, u2d, v2d, 'Streamlines', figsize=(12,4))
    plot_util.plot_solve_iters(gmres_list, cg_list, step_list)
    plt.show()


if __name__ == "__main__":

    # use_single_gpu(DEVICE_ID=7)
    print_device_info()
    
    num_elem = [150, 15] # x elements, y elements
    domain_size= [10, 1] # x length, y length
    t_final = 215 # final time of simulation
    Re = 800
    Cmax = 10 # for CFL condition

    U0, p0, timestep_func, dt, node_coords = step_setup(num_elem, domain_size, Re, Cmax)

    U, p, gmres_list, cg_list, step_list = timestep_loop(t_final, dt, 20, timestep_func, U0, p0)

    # save_sol(U, p, 'data/800step.npy')
    # save_iter_hist(gmres_list, cg_list, step_list, 'data/800step_iters.npy')

    # U, p = load_sol('data/800step.npy')
    # gmres_list, cg_list, step_list = load_iter_hist('data/800step_iters.npy')

    step_plots(node_coords, num_elem, U, p, Re, gmres_list, cg_list, step_list)




