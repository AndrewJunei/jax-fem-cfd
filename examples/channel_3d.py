import jax.numpy as jnp
import jax_fem_cfd.setup.simulation as sim
from jax_fem_cfd.utils.running import *


def channel_setup(U_d, gamma_d, surfnodes, nn, u_inlet):
    """ 3D channel flow where the left face is an inlet, right is an outlet, top and bottom are
        walls, and front and back are symmetry boundary conditions.
    """
    left_nodes = surfnodes[3]
    U_d = sim.set_face_velocity(U_d, left_nodes, gamma_d, u_inlet, nn, var=1) # rest are 0

    U0 = jnp.zeros(3*nn) # initial condition
    P0 = jnp.zeros(nn)
    
    return U_d, U0, P0


def channel_plots(node_coords, num_elem, domain_size, U, p, U_iters, P_iters, steps):
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
    plot_util.plot_solve_iters(U_iters, P_iters, steps)
    plt.show()


if __name__ == "__main__":

    # use_single_gpu(DEVICE_ID=7)
    print_device_info()

    config = sim.Config(mesh="simple",
                        solver="standard",
                        mesh_config=sim.SimpleMeshConfig(
                            num_elem=[60, 20, 20], # x elements, y elements, z elements
                            domain_size=[3, 1, 1], # x length, y length, z length
                            inlet_faces=[3], # L
                            wall_faces=[0, 2], # DT
                            outlet_faces=[1]), # R
                        solver_config=sim.StandardSolverConfig(
                            ndim=3,
                            shape_func_ord=1,
                            streamline_diff=True,
                            pressure_precond=None))
    
    precompute, timestep, node_coords, surfnodes, h, nn, gamma_d, U_d = sim.setup(config)
    print('Mesh and solver setup complete')
    num_elem = config.mesh_config.num_elem
    print(f'Using a {num_elem[0]}x{num_elem[1]}x{num_elem[2]} mesh')

    # FIX: 3D boundary integral

    t_final = 3
    Cmax = 5
    rho, mu, u_inlet = 1, 0.01, 1
    dt = Cmax * h / u_inlet
    Re = int(rho * u_inlet * config.mesh_config.domain_size[1] / mu)
    print('Re =', Re)
    print('dt =', dt)

    U_d, U0, P0 = channel_setup(U_d, gamma_d, surfnodes, nn, u_inlet)
    print('Boundary conditions have been set\n')

    U, p, U_iters, P_iters, steps = sim.timestep_loop(t_final, dt, 1, 1, precompute, timestep, 
                                                      rho, mu, U_d, U0, P0) 

    # save_sol(U, p, 'data/channel_3d.npy')
    # save_iter_hist(gmres_list, cg_list, step_list, 'data/channel_3d_iters.npy')

    # U, p = load_sol('data/channel_3d.npy')
    # gmres_list, cg_list, step_list = load_iter_hist('data/channel_3d_iters.npy')

    channel_plots(node_coords, config.mesh_config.num_elem, config.mesh_config.domain_size, 
                  U, p, U_iters, P_iters, steps)


