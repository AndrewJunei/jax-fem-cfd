import jax.numpy as jnp
import time
from dataclasses import dataclass
from typing import Literal, Optional, List


@dataclass
class SimpleMeshConfig:
    # note that specifying types in python doesn't really do anything
    # these variables can be set to any type when the class is initialized
    num_elem: List[int]
    domain_size: List[float]
    dirichlet_faces: List[int]
    symmetry_faces: List[int]
    outlet_faces: List[int]
    periodic: bool

@dataclass
class StandardSolverConfig:
    ndim: int
    shape_func_ord: int
    streamline_diff: bool
    crank_nicolson: bool
    solver_tol: float
    set_zeroP: bool
    pressure_precond: Optional[str] = None
    final_multigrid_mesh: Optional[List[int]] = None

@dataclass
class Config:
    mesh: Literal["simple"]
    solver: Literal["standard", "passive_transport"] 
    mesh_config: SimpleMeshConfig
    solver_config: StandardSolverConfig # ex: Union[StandardSolverConfig, TwophaseSolverConfig]

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
        else:
            raise ValueError(f"Unknown solver type: {self.solver}")
        
        # additional checks
        if self.solver == "standard":
            if self.mesh != "simple":
                raise ValueError("Mesh type is not implemented for the current solver")
            
            if self.solver_config.ndim != len(self.mesh_config.num_elem):
                raise ValueError("Solver and mesh dimensions must match")
            
            if self.solver_config.ndim == 3:
                if self.solver_config.shape_func_ord == 2:
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


def simple_mesh(mesh_config: SimpleMeshConfig, shape_func_ord):
    from ..setup import regular_mesh as rm

    domain_size = mesh_config.domain_size
    num_elem = mesh_config.num_elem
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

    if mesh_config.dirichlet_faces: # dirichlet faces is non empty
        bconn = jnp.concatenate([bconns[i] for i in mesh_config.dirichlet_faces], axis=0)
        nvec = jnp.concatenate([nvecs[i] for i in mesh_config.dirichlet_faces], axis=0)
        gamma_d = jnp.concatenate([surfnodes[i] for i in mesh_config.dirichlet_faces]) # check corners
        gamma_d = jnp.sort(jnp.unique(gamma_d))  # unique is not jitable
        gamma_d = jnp.concatenate([gamma_d + nn*i for i in range(ndim)]) 
        U_d = jnp.zeros_like(gamma_d, dtype=float) # initialization

    if not mesh_config.outlet_faces: # True if outlet_faces is empty
        gamma_p = 0 # set pressure = 0 at node 0, technically optional with cg
    else:
        gamma_p = jnp.concatenate([surfnodes[i] for i in mesh_config.outlet_faces]) # check corners
        gamma_p = jnp.unique(gamma_p)

    return node_coords, nconn, surfnodes, nn, h, bconn, nvec, gamma_d, U_d, gamma_p, mesh_func

def setup(config: Config):

    if config.mesh == "simple":
        mesh_data = simple_mesh(config.mesh_config, config.solver_config.shape_func_ord)
        node_coords, nconn, surfnodes, nn, h, bconn, nvec, gamma_d, U_d, gamma_p, mesh_func = mesh_data

    if config.solver == "standard":
        from ..solvers.standard import setup_solver

        compute, timestep = setup_solver(node_coords, nconn, bconn, nvec, gamma_d, gamma_p, nn, 
                                         config.solver_config.ndim, h, 
                                         config.solver_config.streamline_diff,
                                         config.solver_config.pressure_precond,
                                         config.solver_config.shape_func_ord,
                                         config.mesh_config.num_elem,
                                         config.solver_config.final_multigrid_mesh,
                                         config.mesh_config.domain_size,
                                         config.mesh_config.outlet_faces,
                                         config.solver_config.solver_tol,
                                         config.solver_config.crank_nicolson,
                                         mesh_func, 
                                         config.mesh_config.periodic,
                                         config.solver_config.set_zeroP)
        
    return compute, timestep, node_coords, surfnodes, h, nn, gamma_d, U_d

def set_face_velocity(U_d, nodes, gamma_d, values, nn, var):
    """ Set velocity values for specified node list
        gamma_d is a 1D array indicating which nodes are dirichlet for velocity
        var = 1: set u, var = 2: set v
    """
    node_indices = jnp.searchsorted(gamma_d, nodes + nn*(var-1))

    return U_d.at[node_indices].set(values)

def timestep_loop(t_final, dt, print_iter, save_iter, compute_func, timestep_func, 
                  rho, mu, U_d, U, P):
    
    U_iter_list, P_iter_list, step_list = [], [], []

    F = compute_func(U_d)

    compile_start = time.perf_counter()
    _, _, _, _ = timestep_func(U, U, P, dt, rho, mu, U_d, F)
    compile_time = time.perf_counter() - compile_start
    print(f'Run to compile took {(compile_time):.2f} s\n')

    num_steps = int(t_final / dt)
    if num_steps * dt < t_final:
        num_steps += 1

    total_start = time.perf_counter()
    for step in range(num_steps):
        if step > 0:
            U_n_1 = U_n
        else:
            U_n_1 = U

        U_n, P_n = U, P

        start = time.perf_counter()
        U, P, U_iters, P_iters = timestep_func(U_n, U_n_1, P_n, dt, rho, mu, U_d, F)
        end = time.perf_counter()

        t = (step + 1) * dt

        diff = (jnp.linalg.norm(jnp.concatenate((U, P)) - jnp.concatenate((U_n, P_n)), 2) / 
                jnp.linalg.norm(jnp.concatenate((U, P)), 2))
        # diff = jnp.max(jnp.abs(U - U_n))

        if step < 5 or (step + 1) % save_iter == 0 or step == num_steps - 1:
            U_iter_list.append(U_iters)
            P_iter_list.append(P_iters)
            step_list.append(step + 1)

        if step < 5 or (step + 1) % print_iter == 0 or step == num_steps - 1:
            print(f'End step {step + 1}, time {t:.3f}, took {((end - start)*1000):.2f} ms, diff = {diff:.6f}')

    total_time = time.perf_counter() - total_start
    print(f'\nTotal timestepping time: {total_time:.2f} s')
    print(f'Avg of {((total_time / num_steps)*1000):.2f} ms per timestep')
    print(f'Avg of {(((total_time + compile_time) / num_steps)*1000):.2f} ms per timestep including compile time\n')

    U_iter_list = jnp.array(U_iter_list)
    P_iter_list = jnp.array(P_iter_list)
    step_list = jnp.array(step_list)

    return U, P, U_iter_list, P_iter_list, step_list