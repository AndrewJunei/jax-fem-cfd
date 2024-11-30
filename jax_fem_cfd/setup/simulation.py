import jax
import jax.numpy as jnp
import numpy as np
import time
from dataclasses import dataclass
from typing import Literal, Optional, List, Dict, Union
import h5py


@dataclass
class SimpleMeshConfig:
    # note that specifying types in python doesn't really do anything
    # these variables can be set to any type when the class is initialized
    num_elem: List[int]
    domain_size: List[float]
    shape_func_ord: int
    periodic: bool
    partial_dirichlet: Dict = None

@dataclass
class StandardSolverConfig:
    dirichlet_faces: List[int]
    symmetry_faces: List[int]
    outlet_faces: List[int]
    streamline_diff: bool
    crank_nicolson: bool
    solver_tol: float
    set_zeroP: bool
    has_source: bool
    pressure_precond: Optional[str] = None
    final_multigrid_mesh: Optional[List[int]] = None

@dataclass
class ADEConfig:
    phi_dirichlet_faces: List[int]
    phi_SUPG: bool
    has_phi_source: bool

@dataclass
class PassiveTransportConfig:
    NS_config: StandardSolverConfig
    ADE_config: ADEConfig

@dataclass
class Config:
    mesh: Literal["simple"]
    solver: Literal["standard", "passive_transport"]
    mesh_config: SimpleMeshConfig
    solver_config: Union[StandardSolverConfig, PassiveTransportConfig]

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.mesh == "simple":
            if not isinstance(self.mesh_config, SimpleMeshConfig):
                raise ValueError("Mesh type and config type must match") 
        else:
            raise ValueError(f"Unknown mesh type: {self.mesh}")
        
        if self.solver == "standard":
            if not isinstance(self.solver_config, StandardSolverConfig):
                raise ValueError("Solver type and config type must match")
        elif self.solver == "passive_transport":
            if not isinstance(self.solver_config, PassiveTransportConfig):
                raise ValueError("Solver type and config type must match")
        else:
            raise ValueError(f"Unknown solver type: {self.solver}")
        
        # additional checks
        if self.solver == "standard":
            if self.mesh != "simple":
                raise ValueError("Mesh type is not implemented for the current solver")
            
            if len(self.mesh_config.num_elem) == 3:
                if self.mesh_config.shape_func_ord == 2:
                    raise ValueError("3D quadratic shape functions not yet implemented")
                if self.solver_config.pressure_precond == "multigrid":
                    raise ValueError("3D multigrid not yet implemented")
                if self.mesh_config.periodic:
                    raise ValueError("3D periodic boundary conditions not yet implemented")
            
            if self.solver_config.pressure_precond == "multigrid":
                num_elemf = self.solver_config.final_multigrid_mesh
                num_elem = self.mesh_config.num_elem

                if num_elemf is None:
                    raise ValueError("Must specify final_multigrid_mesh to use multigrid")
                
                for n, f in zip(num_elem, num_elemf):
                    if f > n:
                        raise ValueError(f"final_multigrid_mesh is too large")
                    ratio = n // f
                    log2_ratio = jnp.log2(ratio)
                    if not jnp.isclose(log2_ratio, jnp.round(log2_ratio)):
                        raise ValueError(f"Invalid final_multigrid_mesh: log2({n}/{f}) is not an integer")


def simple_mesh(mesh_config: SimpleMeshConfig, NS_config: StandardSolverConfig):
    from ..setup import regular_mesh as rm

    domain_size = mesh_config.domain_size
    num_elem = mesh_config.num_elem
    shape_func_ord = mesh_config.shape_func_ord
    ndim = len(num_elem)

    if shape_func_ord == 1:
        if ndim == 2:
            if mesh_config.periodic:
                mesh_func = rm.create_periodic_mesh_2d
            else:
                mesh_func = rm.create_mesh_2d
        
        elif ndim == 3:
            mesh_func = rm.create_mesh_3d

    elif shape_func_ord == 2:
        if ndim == 2:
            if mesh_config.periodic:
                mesh_func = rm.create_periodic_mesh_2d_quad9
            else:
                mesh_func = rm.create_mesh_2d_quad9
    
    node_coords, nconn, surfnodes, bconns, nvecs = mesh_func(*num_elem, *domain_size)
    nn = node_coords.shape[0]
    h = mesh_config.domain_size[0] / mesh_config.num_elem[0]

    # default
    bconn, nvec, gamma_d, U_d, gamma_p = None, None, None, None, None

    dirichlet_nodes = []
    outlet_nodes = []
    bconn_list = []
    nvec_list = []
    
    if NS_config.dirichlet_faces:
        for face in NS_config.dirichlet_faces:
            dirichlet_nodes.append(surfnodes[face])
            bconn_list.append(bconns[face])
            nvec_list.append(nvecs[face])

    if NS_config.outlet_faces:
        for face in NS_config.outlet_faces:
            outlet_nodes.append(surfnodes[face])
    
    if mesh_config.partial_dirichlet:
        for face, (start_frac, end_frac, remainder_type) in mesh_config.partial_dirichlet.items():
            face_nodes = surfnodes[face]
            face_bconn = bconns[face]
            face_nvec = nvecs[face]

            num_face_elem = face_bconn.shape[0]
            
            start_idx = round(start_frac * num_face_elem)
            end_idx = round(end_frac * num_face_elem)
            
            bconn_list.append(face_bconn[start_idx:end_idx])
            nvec_list.append(face_nvec[start_idx:end_idx])

            if shape_func_ord == 2:
                start_idx = 2 * start_idx
                end_idx = 2 * end_idx

            surfnodes[face] = face_nodes[start_idx:end_idx+1]

            dirichlet_nodes.append(face_nodes[start_idx:end_idx+1])
            
            if remainder_type == "outlet":
                outlet_nodes.append(face_nodes[:start_idx])
                outlet_nodes.append(face_nodes[end_idx+1:])
    
    if dirichlet_nodes:
        gamma_d = jnp.concatenate(dirichlet_nodes)
        gamma_d = jnp.sort(jnp.unique(gamma_d))
        gamma_d = jnp.concatenate([gamma_d + nn*i for i in range(ndim)])
        U_d = jnp.zeros_like(gamma_d, dtype=float)
        
        bconn = jnp.concatenate(bconn_list, axis=0)
        nvec = jnp.concatenate(nvec_list, axis=0)

    if outlet_nodes: # True if outlet_nodes is non empty
        gamma_p = jnp.unique(jnp.concatenate(outlet_nodes)) # check corners
    else:
        gamma_p = 0

    return node_coords, nconn, surfnodes, nn, h, bconn, nvec, gamma_d, U_d, gamma_p, mesh_func

def setup(config: Config):

    if config.mesh == "simple":
        if config.solver == "standard":
            mesh_data = simple_mesh(config.mesh_config, config.solver_config)
        elif config.solver == "passive_transport":
            mesh_data = simple_mesh(config.mesh_config, config.solver_config.NS_config)

        node_coords, nconn, surfnodes, nn, h, bconn, nvec, gamma_d, U_d, gamma_p, mesh_func = mesh_data

    if config.solver == "standard":
        from ..solvers.standard import setup_solver

        compute, timestep = setup_solver(node_coords, nconn, bconn, nvec, gamma_d, gamma_p, nn, 
                                         len(config.mesh_config.num_elem), h, 
                                         config.solver_config.streamline_diff,
                                         config.solver_config.pressure_precond,
                                         config.mesh_config.shape_func_ord,
                                         config.mesh_config.num_elem,
                                         config.solver_config.final_multigrid_mesh,
                                         config.mesh_config.domain_size,
                                         config.solver_config.outlet_faces,
                                         config.solver_config.solver_tol,
                                         config.solver_config.crank_nicolson,
                                         mesh_func, 
                                         config.mesh_config.periodic,
                                         config.solver_config.set_zeroP,
                                         config.mesh_config.partial_dirichlet,
                                         config.solver_config.has_source)
        
        return compute, timestep, node_coords, surfnodes, h, nn, gamma_d, U_d
        
    elif config.solver == "passive_transport":
        from ..solvers.passive_transport import setup_solver

        if config.mesh_config.shape_func_ord == 1:
            nnx, nny = config.mesh_config.num_elem[0] + 1, config.mesh_config.num_elem[1] + 1
        elif config.mesh_config.shape_func_ord == 2:
            nnx, nny = 2*config.mesh_config.num_elem[0] + 1, 2*config.mesh_config.num_elem[1] + 1

        if not config.solver_config.ADE_config.phi_dirichlet_faces:
            gamma_d_phi = None
        else:
            gamma_d_phi = jnp.sort(jnp.unique(jnp.concatenate(surfnodes[i] for i in config.solver_config.ADE_config.phi_dirichlet_faces)))

        compute, timestep = setup_solver(node_coords, nconn, bconn, nvec, gamma_d, gamma_p, nn, gamma_d_phi, mesh_func, config)

        return compute, timestep, node_coords, surfnodes, h, nn, nnx, nny, gamma_d, U_d, gamma_d_phi

def set_face_velocity(U_d, nodes, gamma_d, values, nn, var):
    """ Set velocity values for specified node list
        gamma_d is a 1D array indicating which nodes are dirichlet for velocity
        var = 1: set u, var = 2: set v
    """
    node_indices = jnp.searchsorted(gamma_d, nodes + nn*(var-1))

    return U_d.at[node_indices].set(values)

def timestep_loop(t_final, dt, print_iter, save_iter, compute_func, timestep_func, 
                  rho, mu, U_d, U, P, S_func=None):
    
    U_iter_list, P_iter_list, step_list = [], [], []

    F_p, F = compute_func(U_d, S_func, 0, 0)

    compile_start = time.perf_counter()
    _, _, _, _ = timestep_func(U, P, dt, rho, mu, U_d, F_p, F)
    compile_time = time.perf_counter() - compile_start
    print(f'Run to compile took {(compile_time):.2f} s\n')

    num_steps = int(t_final / dt)
    if num_steps * dt < t_final:
        num_steps += 1

    total_start = time.perf_counter()
    for step in range(num_steps):
        U_n, P_n = jnp.array(U), jnp.array(P)

        t_n = step * dt
        t = (step + 1) * dt

        start = time.perf_counter()
        F_p, F = compute_func(U_d, S_func, t_n, t) # should be optional
        U, P, U_iters, P_iters = timestep_func(U_n, P_n, dt, rho, mu, U_d, F_p, F)
        end = time.perf_counter()

        if step < 5 or (step + 1) % save_iter == 0 or step == num_steps - 1:
            U_iter_list.append(U_iters)
            P_iter_list.append(P_iters)
            step_list.append(step + 1)

        if step < 5 or (step + 1) % print_iter == 0 or step == num_steps - 1:
            diff = (jnp.linalg.norm(jnp.concatenate((U, P)) - jnp.concatenate((U_n, P_n)), 2) / 
                    jnp.linalg.norm(jnp.concatenate((U, P)), 2))
            # diff = jnp.max(jnp.abs(U - U_n))
            print(f'End step {step + 1}, time {t:.3f}, took {((end - start)*1000):.2f} ms, diff = {diff:.6f}')

    total_time = time.perf_counter() - total_start
    print(f'\nTotal timestepping time: {total_time:.2f} s')
    print(f'Avg of {((total_time / num_steps)*1000):.2f} ms per timestep')
    print(f'Avg of {(((total_time + compile_time) / num_steps)*1000):.2f} ms per timestep including compile time\n')

    U_iter_list = jnp.array(U_iter_list)
    P_iter_list = jnp.array(P_iter_list)
    step_list = jnp.array(step_list)

    return U, P, U_iter_list, P_iter_list, step_list

""" Reorganize Later. """
def timestep_loop_transport(t_final, dt, save_interval, print_interval, compute_func, timestep_func, 
                  rho, mu, D, U_d, phi_d, U, P, phi, S_func=None, phi_S_func=None, save2file=False, h5name=None):
    
    U_iter_list, P_iter_list, phi_iter_list, step_list = [], [], [], []

    F_p, F = compute_func(U_d, S_func, 0, 0)

    compile_start = time.perf_counter()
    # always recompiles second step no matter what we do...
    _ = timestep_func(U, P, phi, rho, mu, D, dt, U_d, F_p, F, phi_d, phi_S_func)
    compile_time = time.perf_counter() - compile_start
    print(f'Run to compile took {(compile_time):.2f} s\n')

    num_steps = int(t_final / dt)
    if num_steps * dt < t_final:
        num_steps += 1

    print('num steps:', num_steps)

    if save2file:
        U_list, phi_list = [], []

        with h5py.File(f'{h5name}.h5', 'w') as f:
            # Calculate optimal chunk size (typically similar to buffer size)
            chunk_size = min(save_interval, 100)  # Don't make chunks too large
            
            # Create datasets with optimized settings
            f.create_dataset('U',
                           shape=(0, U.shape[0]),
                           maxshape=(None, U.shape[0]),
                           chunks=(chunk_size, U.shape[0]),
                           compression='gzip',
                           compression_opts=4,  # Balance between speed and compression
                           fletcher32=True)     # Checksum for data integrity
            
            f.create_dataset('phi',
                           shape=(0, phi.shape[0]),
                           maxshape=(None, phi.shape[0]),
                           chunks=(chunk_size, phi.shape[0]),
                           compression='gzip',
                           compression_opts=4,
                           fletcher32=True)
            
            # Save initial condition
            f['U'].resize(1, axis=0)
            f['phi'].resize(1, axis=0)
            f['U'][0] = jax.device_get(U) # initial condition
            f['phi'][0] = jax.device_get(phi) # initial condition
    
    else:
        U_list, phi_list = [jax.device_get(U)], [jax.device_get(phi)]

    def save_buffer():
        if not save2file:
            return
            
        with h5py.File(f'{h5name}.h5', 'a') as f:
            current_size = f['U'].shape[0]
            batch_size = len(U_list)
            
            # Resize once for the batch
            f['U'].resize(current_size + batch_size, axis=0)
            f['phi'].resize(current_size + batch_size, axis=0)
            
            # Convert and save in one operation
            f['U'][current_size:current_size + batch_size] = np.stack(U_list)
            f['phi'][current_size:current_size + batch_size] = np.stack(phi_list)
        
        U_list.clear()
        phi_list.clear()

    total_start = time.perf_counter()
    for step in range(1, num_steps + 1):
        U_n, P_n, phi_n = jnp.copy(U), jnp.copy(P), jnp.copy(phi)

        t_n = (step - 1)*dt
        t = step * dt

        start = time.perf_counter()
        F_p, F = compute_func(U_d, S_func, t_n, t) # should be optional
        U, P, phi, U_iters, P_iters, phi_iters = timestep_func(U_n, P_n, phi_n, rho, mu, D, dt, U_d,
                                                               F_p, F, phi_d, phi_S_func)
        end = time.perf_counter()

        if step % save_interval == 0 or step == num_steps:
            U_list.append(jax.device_get(U))
            phi_list.append(jax.device_get(phi))
            U_iter_list.append(U_iters)
            P_iter_list.append(P_iters)
            phi_iter_list.append(phi_iters)
            step_list.append(step)
            save_buffer()

        if step <= 5 or step % print_interval == 0 or step == num_steps:
            diff = jnp.linalg.norm(U - U_n, 2) / jnp.linalg.norm(U, 2)
            print(f'End step {step}, time {t:.2f}, took {((end - start)*1000):.2f} ms, {U_iters} U iters, {P_iters} P iters, {phi_iters} phi iters, diff = {diff:.6f}')
    
    total_time = time.perf_counter() - total_start
    print(f'\nTotal timestepping time: {total_time:.2f} s')
    print(f'Average of {((total_time / step)*1000):.2f} ms per timestep\n')

    U_iter_list = jnp.array(U_iter_list)
    P_iter_list = jnp.array(P_iter_list)
    phi_iter_list = jnp.array(phi_iter_list)
    step_list = jnp.array(step_list)

    if save2file:
        U_list.clear()
        phi_list.clear()
    else:
        U_list = np.stack(U_list)
        phi_list = np.stack(phi_list)

    return U, P, phi, U_iter_list, P_iter_list, phi_iter_list, step_list, U_list, phi_list

def timestep_loop_scan(dt, num_steps, compute_func, timestep_func, rho, mu, D, U_d, phi_d, U0, P0, phi0, S_func):
    F_p, F = compute_func(U_d, S_func, 0, 0)

    def scan_body(carry, _):
        U_n, P_n, phi_n = carry
        U, P, phi, _, _, _ = timestep_func(U_n, P_n, phi_n, dt, rho, mu, D, U_d, F_p, phi_d, F, None)
        return (U, P, phi), None
    
    initial_carry = (U0, P0, phi0)

    (U, P, phi), _ = jax.lax.scan(scan_body, initial_carry, None, num_steps)

    return U, P, phi