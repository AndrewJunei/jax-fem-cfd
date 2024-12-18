import jax.numpy as jnp
import jax_fem_cfd.setup.simulation as sim
from jax_fem_cfd.utils.running import *


def step_setup(U_d, gamma_d, surfnodes, nn):
    """ 2D flow over a backward facing step. The step takes up the bottom half of the total 
        domain height. In this setup, the left boundary marks the end of the step. For this to 
        work, the x velocity takes on a fully developed parabolic profile on the top half of the 
        left boundary and is 0 below. There are walls on the top and bottom boundaries, and an
        outlet on the right.
    """
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
    U_d = sim.set_face_velocity(U_d, left_nodes, gamma_d, left_velocity, nn, var=1) # rest are 0

    U0 = jnp.zeros(2*nn) # initial condition
    P0 = jnp.zeros(nn)

    return U_d, U0, P0

def step_plots(node_coords, num_elem, U, p, Re, U_iters, P_iters, steps, ord):
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

    x2d, y2d, u2d, v2d = plot_util.get_2d_arrays(x, y, u, v, nnx, nny)

    def interpolate_at_x(x, u2d, target_x):
        distances = jnp.abs(x - target_x)
        closest_index = jnp.argmin(distances)
        
        if distances[closest_index] == 0:
            return u2d[:, closest_index]
        
        closest_indices = jnp.argsort(distances)[:2]
        i1, i2 = jnp.sort(closest_indices)
        
        x1, x2 = x[i1], x[i2]
        u1, u2 = u2d[:, i1], u2d[:, i2]
        
        weight = (target_x - x1) / (x2 - x1)
        return u1 + weight * (u2 - u1)
    
    ygartling = jnp.linspace(0.0, 1.0, 21)
    ugartling7 = jnp.array([0., 0.232, 0.428, 0.613, 0.792, 0.948, 1.062, 1.118, 1.105, 1.024, 0.885,
                             0.709, 0.522, 0.349, 0.204, 0.092, 0.015, -0.032, -0.049, -0.038, 0.])
    vgartling7 = jnp.array([0., -0.118, -0.504, -1.000, -1.423, -1.748, -1.917, -1.925, -1.778,
                            -1.507, -1.165, -0.823, -0.544, -0.362, -0.268, -0.225, -0.193, -0.147,
                            -0.086, -0.027, 0.]) * 1e-2

    tri = plot_util.get_2d_tri(x, y)
    plot_util.plot_contour(tri, u, x, y, u, v, nnx, nny, 'x Velocity', 
                           figsize=(12, 4), c_shrink=0.4, quiv=False)
    
    if Re == 800:
        x1d = x2d[0, :] 
        y1d = y2d[:, 0]  
        u_at_target_x = interpolate_at_x(x1d, u2d, 7.0)
        v_at_target_x = interpolate_at_x(x1d, v2d, 7.0)
        reflbl = 'Gartling Re=800'
        lbl = f'{num_elem[0]}x{num_elem[1]} Mesh'
        plot_util.compare_double(ygartling, ugartling7, reflbl, y1d, u_at_target_x, lbl, 
                                ygartling, vgartling7, reflbl, y1d, v_at_target_x, lbl,
                                'x Velocity at x=7.0', 'y Velocity at x=7.0', 'y', 'y')

    # plot_util.plot_streamlines(x2d, y2d, u2d, v2d, 'Streamlines', figsize=(12,4))
    # plot_util.plot_solve_iters(U_iters, P_iters, steps)
    
    plot_util.show_plots()


if __name__ == "__main__":
    # use_single_gpu(DEVICE_ID=7)
    print_device_info()
    jax.config.update("jax_enable_x64", True)

    config = sim.Config(mesh="simple",
                        solver="standard",
                        mesh_config=sim.SimpleMeshConfig(
                            num_elem=[16*20, 16], # x elements, y elements
                            domain_size=[20, 1], # x length, y length
                            dirichlet_faces=[0, 2, 3], # down, top, left
                            symmetry_faces=[], 
                            outlet_faces=[1], # right
                            periodic=False), 
                        solver_config=sim.StandardSolverConfig(
                            ndim=2,
                            shape_func_ord=1,
                            streamline_diff=False,
                            crank_nicolson=True,
                            solver_tol=1e-6,
                            set_zeroP=True, # necessary when there is an outlet
                            pressure_precond='multigrid',
                            final_multigrid_mesh=[20, 1]))
    
    precompute, timestep, node_coords, surfnodes, h, nn, gamma_d, U_d = sim.setup(config)
    print('Mesh and solver setup complete')
    print(f'Using a {config.mesh_config.num_elem[0]}x{config.mesh_config.num_elem[1]} mesh')
    
    t_final = 250 # final time of simulation
    Cmax = 5 # for CFL condition
    rho, u_avg = 1, 1
    Re = 800
    mu = rho * u_avg * config.mesh_config.domain_size[1] / Re
    dt = Cmax * h / u_avg
    print('Re =', Re)
    print('dt =', dt)

    U_d, U0, P0 = step_setup(U_d, gamma_d, surfnodes, nn)
    print('Boundary conditions have been set\n')

    U, p, U_iters, P_iters, steps = sim.timestep_loop(t_final, dt, 100, 100, precompute, timestep, 
                                                      rho, mu, U_d, U0, P0) 

    # save_sol(U, p, 'data/800step.npy')
    # save_iter_hist(gmres_list, cg_list, step_list, 'data/800step_iters.npy')

    # U, p = load_sol('data/800step.npy')
    # gmres_list, cg_list, step_list = load_iter_hist('data/800step_iters.npy')

    step_plots(node_coords, config.mesh_config.num_elem, U, p, Re, U_iters, P_iters, steps, 
               config.solver_config.shape_func_ord)




