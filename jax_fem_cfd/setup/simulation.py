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
    inlet_faces: List[int]
    wall_faces: List[int]
    outlet_faces: List[int]

@dataclass
class StandardSolverConfig:
    ndim: int
    shape_func_ord: int
    streamline_diff: bool
    pressure_precond: Optional[str]
    final_multigrid_mesh: Optional[List[int]]

@dataclass
class Config:
    mesh: Literal["simple"]
    solver: Literal["standard"] # ex: Literal["standard", "twophase"]
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
            
            if self.solver_config.ndim == 3 and self.solver_config.shape_func_ord == 2:
                raise ValueError("3D quadratic shape functions not yet implemented")
            
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

    ndim = len(mesh_config.num_elem)

    if shape_func_ord == 1:
        if ndim == 2:
            mesh_data = rm.create_mesh_2d(*mesh_config.num_elem, *mesh_config.domain_size)
        
        elif ndim == 3:
            mesh_data = rm.create_mesh_3d(*mesh_config.num_elem, *mesh_config.domain_size)

    elif shape_func_ord == 2:
        if ndim == 2:
            mesh_data = rm.create_mesh_2d_quad9(*mesh_config.num_elem, *mesh_config.domain_size)
    
    node_coords, nconn, surfnodes, bconns, nvecs = mesh_data
    nn = node_coords.shape[0]
    h = mesh_config.domain_size[0] / mesh_config.num_elem[0]

    bconn = jnp.concatenate([bconns[i] for i in mesh_config.inlet_faces], axis=0)
    nvec = jnp.concatenate([nvecs[i] for i in mesh_config.inlet_faces], axis=0)

    dirichlet_faces = mesh_config.inlet_faces + mesh_config.wall_faces
    gamma_d = jnp.concatenate([surfnodes[i] for i in dirichlet_faces]) # check corners

    if not mesh_config.outlet_faces: # True if outlet_faces is empty
        gamma_p = 0 # set pressure = 0 at node 0
    else:
        gamma_p = jnp.concatenate([surfnodes[i] for i in mesh_config.outlet_faces]) # check corners

    gamma_d = jnp.sort(jnp.unique(gamma_d))  # unique is not jitable
    gamma_d = jnp.concatenate([gamma_d + nn*i for i in range(ndim)]) 
    U_d = jnp.zeros_like(gamma_d, dtype=float) # initialization
    gamma_p = jnp.unique(gamma_p)

    return node_coords, nconn, surfnodes, nn, h, bconn, nvec, gamma_d, U_d, gamma_p

def setup(config: Config):

    if config.mesh == "simple":
        mesh_data = simple_mesh(config.mesh_config, config.solver_config.shape_func_ord)
        node_coords, nconn, surfnodes, nn, h, bconn, nvec, gamma_d, U_d, gamma_p = mesh_data

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
                                         config.mesh_config.outlet_faces)
        
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
    
    step, t = 0, 0.0
    U_iter_list, P_iter_list, step_list = [], [], []

    F = compute_func(U_d)

    compile_start = time.perf_counter()
    _, _, _, _ = timestep_func(U, U, P, dt, rho, mu, U_d, F)
    compile_time = time.perf_counter() - compile_start
    print(f'Run to compile took {(compile_time):.2f} s\n')

    total_start = time.perf_counter()
    while t < t_final: 
        if step > 0:
            U_n_1 = U_n
        else:
            U_n_1 = U

        U_n, P_n = U, P

        start = time.perf_counter()
        U, P, U_iters, P_iters = timestep_func(U_n, U_n_1, P_n, dt, rho, mu, U_d, F)
        end = time.perf_counter()

        t += dt
        step += 1

        if step == 1 or step % save_iter == 0 or t >= t_final:
            U_iter_list.append(U_iters)
            P_iter_list.append(P_iters)
            step_list.append(step)

        if step == 1 or step % print_iter == 0 or t >= t_final:
            diff = (jnp.linalg.norm(jnp.concatenate((U, P)) - jnp.concatenate((U_n, P_n)), 2) / 
                        jnp.linalg.norm(jnp.concatenate((U, P)), 2))

            print(f'End step {step}, time {t:.2f}, took {((end - start)*1000):.2f} ms, diff = {diff:.6f}')

    total_time = time.perf_counter() - total_start
    print(f'\nTotal timestepping time: {total_time:.2f} s')
    print(f'Average of {((total_time / step)*1000):.2f} ms per timestep')

    U_iter_list = jnp.array(U_iter_list)
    P_iter_list = jnp.array(P_iter_list)
    step_list = jnp.array(step_list)

    return U, P, U_iter_list, P_iter_list, step_list