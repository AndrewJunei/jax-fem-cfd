import jax.numpy as jnp
import jax_fem_cfd.setup.simulation as sim
from jax_fem_cfd.utils.running import *


def vortex(node_coords, alpha):
    x = node_coords[:, 0]
    y = node_coords[:, 1]

    u = -alpha * (y - 0.5)
    v = alpha * (x - 0.5)

    return jnp.concatenate((u, v))

def create_rect(node_coords, edge_start_x, edge_end_x, edge_start_y, edge_end_y, mag=1.0):
    x = node_coords[:, 0]
    y = node_coords[:, 1]

    square = (x >= edge_start_x) & (x <= edge_end_x) & (y >= edge_start_y) & (y <= edge_end_y)

    phi = jnp.where(square, mag, 0.0)
    return phi

def create_x(node_coords, size, thickness):
    x = node_coords[:, 0]
    y = node_coords[:, 1]

    half_size = size / 2
    lower_x = 0.5 - half_size
    upper_x = 0.5 + half_size
    lower_y = 0.5 - half_size
    upper_y = 0.5 + half_size

    in_bounds = (x >= lower_x) & (x <= upper_x) & (y >= lower_y) & (y <= upper_y)
    diagonal1 = jnp.abs((y - lower_y) - (x - lower_x)) <= thickness / 2
    diagonal2 = jnp.abs((y - lower_y) + (x - lower_x) - size) <= thickness / 2

    phi = jnp.where(in_bounds & (diagonal1 | diagonal2), 1.0, 0.0)
    return phi

def create_asymmetric_push(node_coords, u_inlet):
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    
    # Base flow with y-dependent magnitude
    u0 = create_rect(node_coords, 0.1, 0.7, 0.3, 0.7, u_inlet)
    amplitude = 0.75 + 0.25*jnp.sin(7*jnp.pi*y)  # Varies from 0.5 to 1
    u0 = u0 * amplitude
    
    # Add small vertical component
    v0 = 0.25 * jnp.sin(7*jnp.pi*x) * u0
    
    return jnp.concatenate((u0, v0))


if __name__ == "__main__":
    # use_single_gpu(DEVICE_ID=7)
    print_device_info()
    jax.config.update("jax_enable_x64", True)

    config = sim.Config(
        mesh="simple",
        solver="passive_transport",
        mesh_config=sim.SimpleMeshConfig(
            num_elem=[64, 64],
            domain_size=[1, 1],
            shape_func_ord=1,
            periodic=False
        ),
        solver_config=sim.PassiveTransportConfig(
            sim.StandardSolverConfig(
                dirichlet_faces=[0, 1, 2, 3],
                symmetry_faces=[],
                outlet_faces=[],
                streamline_diff=False,
                solver_tol=1e-6,
                set_zeroP=True,
                pressure_precond="multigrid",
                final_multigrid_mesh=[4, 4]
            ),
            sim.ADEConfig(
                phi_dirichlet_faces=[],
                phi_SUPG=True,
            )
        )
    )

    compute, timestep, node_coords, surfnodes, h, nn, nnx, nny, gamma_d, U_d, gamma_d_phi = sim.setup(config)
    print('Mesh and solver setup complete')
    print(f'Using a {config.mesh_config.num_elem[0]}x{config.mesh_config.num_elem[1]} mesh')

    option = "rotating_x"

    if option == "rotating_x":
        rho, mu, D = 1, 0.01, 1e-4
        t_final = 3

        U0 = vortex(node_coords, 5)
        P0 = jnp.zeros(nn)
        phi0 = create_x(node_coords, 0.5, 0.1)
        phi_d = None
        u0 = U0[:nn]
        v0 = U0[nn:]
        u_scale = jnp.mean(jnp.sqrt(u0**2 + v0**2))
    
    elif option == "asymm_push":
        rho, mu, D = 1, 0.0025, 1e-4
        u_scale = 25
        t_final = 1

        U0 = create_asymmetric_push(node_coords, u_scale)
        P0 = jnp.zeros(nn)
        phi0 = create_rect(node_coords, 0.2, 0.8, 0.3, 0.7)
        phi_d = None

    Cmax = 1 # recommended for SUPG 
    dt = Cmax * h / u_scale

    Re = round(rho * u_scale * config.mesh_config.domain_size[0] / mu)
    NS_elem_Pe = rho * u_scale * h / (2*mu)
    ADE_elem_Pe = u_scale * h / (2*D)
    print('Re =', Re)
    print('dt =', dt)
    print(f'Element Pe for Navier-Stokes = {NS_elem_Pe:.2f}')
    print(f'Element Pe for ADE = {ADE_elem_Pe:.2f}\n')

    sim_data = sim.timestep_loop(t_final, dt, 1, 100, compute, timestep, 
                                 [rho, mu, D], [U_d, phi_d], [U0, P0, phi0])
    U, P, phi, U_iters, P_iters, phi_iters, steps, U_list, phi_list = sim_data

    import jax_fem_cfd.utils.plotting as plot_util
    plot_util.save_animation(phi_list, 'examples/data/test', dt, nnx, nny, 5, 0, True)
