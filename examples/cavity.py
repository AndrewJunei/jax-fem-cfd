import jax.numpy as jnp
import jax_fem_cfd.setup.simulation as sim
from jax_fem_cfd.utils.running import *


def cavity_plots(node_coords, num_elem, domain_size, U, p, Re, U_iters, P_iters, steps, ord):
    import jax_fem_cfd.utils.plotting as plot_util
    import matplotlib.pyplot as plt

    nn = node_coords.shape[0]
    if ord == 1:
        nnx, nny = num_elem[0] + 1, num_elem[1] + 1
    elif ord == 2:
        nnx, nny = 2*num_elem[0] + 1, 2*num_elem[1] + 1

    x = node_coords[:, 0]
    y = node_coords[:, 1]
    u = U[:nn]
    v = U[nn:]

    yghia = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172,
         0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]

    xghia = [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000, 0.8047, 
            0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000]

    ughia100 = [0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
                -0.20581, -0.13641, -0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000]

    ughia400 = [0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477,
                0.02135,  0.16256,  0.29093,  0.55892,  0.61756, 0.68439,  0.75837,  1.00000]

    ughia1000 = [0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648, -0.06080,
                0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.00000]

    ughia5000 = [0.00000, -0.41165, -0.42901, -0.43643, -0.40435, -0.33050, -0.22855, -0.07404, -0.03039,
                0.08183, 0.20087, 0.33556, 0.46036, 0.45992, 0.46120, 0.48223, 1.00000]

    vghia100 = [0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533,
                -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000]

    vghia400 = [0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174, 0.05186, -0.38958,
                -0.44993, -0.23827, -0.22847, -0.19254, -0.15663, -0.12146, 0.00000]

    vghia1000 = [0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02526, -0.31966,
                -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21388, 0.00000]

    vghia5000 = [0.00000, 0.42447, 0.43329, 0.43648, 0.42951, 0.35368, 0.28066, 0.27280, 0.00945, -0.30018,
                -0.36214, -0.41442, -0.52876, -0.55408, -0.55069, -0.49774, 0.00000]
    
    ghia_map = { 5000: (ughia5000, vghia5000), 1000: (ughia1000, vghia1000),
                400: (ughia400, vghia400), 100: (ughia100, vghia100) }

    # tri = plot_util.get_2d_tri(x, y)
    # plot_util.plot_contour(tri, u, x, y, u, v, nnx, nny, 'x Velocity', quiv=True)
    # plot_util.plot_surface(x, y, u, 'x Velocity', figsize=(8,5))
    # plot_util.plot_surface(x, y, p, 'Pressure', figsize=(8,5))

    x2d, y2d, u2d, v2d = plot_util.get_2d_arrays(x, y, u, v, nnx, nny)
    plot_util.plot_streamlines(x2d, y2d, u2d, v2d, 'Streamlines')

    if Re in [5000, 1000, 400, 100]:
        ughia, vghia = ghia_map.get(Re, (None, None))
        ux = u.reshape(nny, nnx)
        vy = v.reshape(nny, nnx)
        x1d = jnp.linspace(0, domain_size[0], vy.shape[1])
        y1d = jnp.linspace(0, domain_size[1], ux.shape[0])
        reflbl = f'Ghia Re={Re}'
        lbl = f'{num_elem[0]}x{num_elem[1]} Mesh'
        plot_util.compare_double(yghia, ughia, reflbl, y1d, ux[:, (nnx - 1) // 2], lbl, 
                                xghia, vghia, reflbl, x1d, vy[(nny - 1) // 2, :], lbl,
                                'x Velocity at x=0.5', 'y Velocity at y=0.5', 'y', 'x')
        
    plot_util.plot_solve_iters(U_iters, P_iters, steps)

    plot_util.show_plots()


if __name__ == "__main__":
    # use_single_gpu(DEVICE_ID=7)
    print_device_info()
    # jax.config.update("jax_enable_x64", True)

    config = sim.Config(mesh="simple",
                        solver="standard",
                        mesh_config=sim.SimpleMeshConfig(
                            num_elem=[128, 128], # x elements, y elements
                            domain_size=[1, 1], # x length, y length
                            dirichlet_faces=[0, 1, 2, 3], # DRTL, 2 = top
                            outlet_faces=[], # no outlet
                            symmetry_faces=[],
                            periodic=False), 
                        solver_config=sim.StandardSolverConfig(
                            ndim=2,
                            shape_func_ord=1,
                            streamline_diff=False,
                            crank_nicolson=True, # why is it faster if True?
                            solver_tol=1e-6,
                            set_zeroP=True,
                            pressure_precond='multigrid',
                            final_multigrid_mesh=[4, 4]))

    compute, timestep, node_coords, surfnodes, h, nn, gamma_d, U_d = sim.setup(config)
    print('Mesh and solver setup complete')
    print(f'Using a {config.mesh_config.num_elem[0]}x{config.mesh_config.num_elem[1]} mesh')

    rho, mu, u_top = 1, 0.001, 1
    t_final = 35 # final time of simulation
    Cmax = 10 # for CFL condition
    dt = Cmax * h / u_top
    Re = int(rho * u_top * config.mesh_config.domain_size[0] / mu)
    print('Re =', Re)
    print('dt =', dt)

    # look into better way of abstracting the boundary condition setup and dt calculation

    # no leaky lid condition!
    top_nodes = surfnodes[2][1:-1] # set lid x velocity
    U_d = sim.set_face_velocity(U_d, top_nodes, gamma_d, u_top, nn, var=1) # rest are 0

    U0 = jnp.zeros(2*nn) # initial condition
    P0 = jnp.zeros(nn)
    print('Boundary conditions have been set\n')

    U, p, U_iters, P_iters, steps = sim.timestep_loop(t_final, dt, 100, 100, compute, timestep, 
                                                      rho, mu, U_d, U0, P0) 

    # save_sol(U, p, 'testing/data/noleak_tol8_64bitincludeF_32mesh_Re1000cavity.npy')
    # save_iter_hist(gmres_list, cg_list, step_list, 'data/400cavity_iters.npy')

    # U, p = load_sol('data/400cavity.npy')
    # gmres_list, cg_list, step_list = load_iter_hist('data/400cavity_iters.npy')

    cavity_plots(node_coords, config.mesh_config.num_elem, config.mesh_config.domain_size,
                 U, p, Re, U_iters, P_iters, steps, config.solver_config.shape_func_ord)




