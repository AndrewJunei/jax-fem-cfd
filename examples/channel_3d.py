import jax.numpy as jnp
from jax_fem_cfd.setup.regular_mesh import create_mesh_3d
from jax_fem_cfd.setup.common import set_face_velocity, format_boundaries, timestep_loop
from jax_fem_cfd.solvers.standard_3d import setup_solver, four_step_timestep
from jax_fem_cfd.utils.running import *


def channel_setup(num_elem, domain_size, rho, mu, u_inlet, Cmax):
    """ 3D channel flow where the left face is an inlet, right is an outlet, top and bottom are
        walls, and front and back are symmetry boundary conditions.
    """
    Re = rho * u_inlet * domain_size[1] / mu
    print('Re =', Re)

    h = domain_size[0] / num_elem[0]
    Pe = u_inlet * h / (2*mu / rho)
    print('Global Pe = ', Pe)

    dt = Cmax * h / u_inlet
    print('dt =', dt)

    node_coords, nconn, surfnodes, bconns, nvecs = create_mesh_3d(*num_elem, *domain_size)
    nn = node_coords.shape[0] # faces = [D, R, T, L, F, B]
    print(f'Using a {num_elem[0]}x{num_elem[1]}x{num_elem[2]} mesh')

    gamma_d = jnp.concatenate([surfnodes[0], surfnodes[2], surfnodes[3]]) # inlet and walls
    bconn = bconns[3] # inlet
    nvec = nvecs[3] # inlet
    gamma_p = surfnodes[1] # outlet (check corners)
    
    gamma_d, U_d, not_gamma_p = format_boundaries(gamma_d, gamma_p, nn, 3) # 3D

    left_nodes = surfnodes[3]
    U_d = set_face_velocity(U_d, left_nodes, gamma_d, u_inlet, nn, 1) # rest are 0

    U = jnp.zeros(3*nn) # initial condition
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


def channel_plots(node_coords, num_elem, domain_size, U, p, gmres_list, cg_list, step_list):
    import jax_fem_cfd.utils.plotting as plot_util
    import matplotlib.pyplot as plt

    nnx, nny, nnz = num_elem[0] + 1, num_elem[1] + 1, num_elem[2] + 1

    def plot_xy(z_node):
        z_coord = domain_size[2] * (z_node / num_elem[2])

        idx = z_node * nnx * nny
        u = U[idx : (idx + nnx*nny)]
        v = U[(nnz + z_node)*nnx*nny : (nnz + z_node + 1)*nnx*nny]
        w = U[(2*nnz + z_node)*nnx*nny : (2*nnz + z_node + 1)*nnx*nny]
        p1 = p[idx : (idx + nnx*nny)]

        x = node_coords[idx : (idx + nnx*nny), 0]
        y = node_coords[idx : (idx + nnx*nny), 1]
        tri = plot_util.get_2d_tri(x, y)

        plot_util.plot_contour(tri, u, x, y, u, v, nnx, nny, f'x Velocity at z={z_coord:.2f}', 
                               figsize=(8, 3), c_shrink=0.8, quiv=False, xlbl='x', ylbl='y')
        plot_util.plot_contour(tri, v, x, y, u, v, nnx, nny, f'y Velocity at z={z_coord:.2f}', 
                               figsize=(8, 3), c_shrink=0.8, quiv=False, xlbl='x', ylbl='y')
        plot_util.plot_contour(tri, w, x, y, u, v, nnx, nny, f'z Velocity at z={z_coord:.2f}', 
                               figsize=(8, 3), c_shrink=0.8, quiv=False, xlbl='x', ylbl='y')
        plot_util.plot_contour(tri, p1, x, y, u, v, nnx, nny, f'Pressure at z={z_coord:.2f}', 
                               figsize=(8, 3), c_shrink=0.8, quiv=False, xlbl='x', ylbl='y')

    def plot_yz(x_node):
        x_coord = domain_size[0] * (x_node / num_elem[0])
        idxs = jnp.arange(x_node, nnx * nny * nnz, nnx)
        
        u = U[idxs]
        v = U[nnz * nnx * nny + idxs]
        w = U[2 * nnz * nnx * nny + idxs]
        p1 = p[idxs]
        
        y = node_coords[idxs, 1]
        z = node_coords[idxs, 2]
        tri = plot_util.get_2d_tri(z ,y)

        plot_util.plot_contour(tri, u, z, y, u, v, nnx, nny, f'x Velocity at x={x_coord:.2f}', 
                               figsize=(8, 3), c_shrink=0.8, quiv=False, xlbl='z', ylbl='y')
        plot_util.plot_contour(tri, v, z, y, u, v, nnx, nny, f'y Velocity at x={x_coord:.2f}', 
                               figsize=(8, 3), c_shrink=0.8, quiv=False, xlbl='z', ylbl='y')
        plot_util.plot_contour(tri, w, z, y, u, v, nnx, nny, f'z Velocity at x={x_coord:.2f}', 
                               figsize=(8, 3), c_shrink=0.8, quiv=False, xlbl='z', ylbl='y')
        plot_util.plot_contour(tri, p1, z, y, u, v, nnx, nny, f'Pressure at x={x_coord:.2f}', 
                               figsize=(8, 3), c_shrink=0.8, quiv=False, xlbl='z', ylbl='y')
    
    plot_xy(z_node=10)
    # plot_yz(x_node=55)
    plot_util.plot_solve_iters(gmres_list, cg_list, step_list)
    plt.show()


if __name__ == "__main__":

    # use_single_gpu(DEVICE_ID=7)
    print_device_info()

    num_elem = [60, 20, 20] # x elements, y elements, z elements
    domain_size= [3, 1, 1] # x length, y length, z length
    rho, mu, u_inlet = 1, 0.01, 1
    t_final = 3
    Cmax = 5

    U0, p0, timestep_func, dt, node_coords = channel_setup(num_elem, domain_size, rho, mu, u_inlet, Cmax)

    U, p, gmres_list, cg_list, step_list = timestep_loop(t_final, dt, 1, timestep_func, U0, p0)

    # save_sol(U, p, 'data/channel_3d.npy')
    # save_iter_hist(gmres_list, cg_list, step_list, 'data/channel_3d_iters.npy')

    # U, p = load_sol('data/channel_3d.npy')
    # gmres_list, cg_list, step_list = load_iter_hist('data/channel_3d_iters.npy')

    channel_plots(node_coords, num_elem, domain_size, U, p, gmres_list, cg_list, step_list)


