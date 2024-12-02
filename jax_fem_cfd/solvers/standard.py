""" Incompressible, Newtonian Navier Stokes
"""
import jax
import jax.numpy as jnp
from functools import partial
from..core import shape_functions as shapefunc
from ..core import element_calc as ecalc
from ..core import element_mult as emult
from ..utils.jax_iterative_solvers import bicgstab, cg
from ..core import multigrid as mg
from ..setup.simulation import Config


def solve_momentum(U_n, P_n, Me, Le, Ge, Ce_all_n, rho, mu, dt, nconn, nn, ndim, gamma_d, U_d, F, tol, theta, gamma):
    """ General momentum equation solver valid for various pressure projection schemes.
        theta controls the temporal discretization, gamma controls if P_n appears on the rhs.
        The boundary conditions are: U = U_d on Gamma_d, (Dirichlet)
                                     grad(U) * n = 0 on Gamma \ Gamma_d (Neumann)          
    """
    def lhs_operator(U):
        result = ((rho/dt) * emult.consistent_mass_mult(U, Me, nconn, nn, ndim) 
                  + theta * emult.visc_conv_mult(U, Ce_all_n, Le, mu, rho, nconn, nn, ndim))
        
        if gamma_d is not None:
            U_mask = jnp.zeros_like(U, dtype=bool).at[gamma_d].set(True) # maybe move outside
            result = jnp.where(U_mask, U, result)

        return result
    
    # theta = 0.5 and gamma = 1 typically, so no need for if statements
    rhs = (((rho/dt) * emult.consistent_mass_mult(U_n, Me, nconn, nn, ndim)) 
        - gamma * emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim) 
        - (1-theta) * emult.visc_conv_mult(U_n, Ce_all_n, Le, mu, rho, nconn, nn, ndim) + F)

    if gamma_d is not None:
        rhs = rhs.at[gamma_d].set(U_d)
        x0 = U_n.at[gamma_d].set(U_d) # better initial guess if BC not applied at last step
    else:
        x0 = U_n

    U, U_iters = bicgstab(lhs_operator, rhs, x0=x0, tol=tol)

    return U, U_iters

""" Not updated yet """
def solve_momentum_stab(U_n, P_n, M, Le, Ge, Ce_all, Se_all, rho, mu, dt, nconn, gamma_d, U_d, nn, ndim, 
                        tol, crank_nicol, has_dirichlet, F):
    """ Same as solve_momentum but with an articial streamline diffusion
    """
    if crank_nicol:
        def lhs_operator(U):
            result = ((rho/dt) * M * U) + emult.stabilized_visc_conv_mult(U, Ce_all, Se_all, Le, mu, 
                                                                          rho, nconn, nn, ndim) / 2
            if has_dirichlet:
                mask = jnp.zeros_like(U, dtype=bool).at[gamma_d].set(True)
                result = jnp.where(mask, U, result)

            return result
        
        rhs = (((rho/dt) * M * U_n) - emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim)
               - emult.stabilized_visc_conv_mult(U_n, Ce_all, Se_all, Le, mu, rho, nconn, nn, ndim) / 2 + F)

    else:
        def lhs_operator(U):
            result = ((rho/dt) * M * U) + emult.stabilized_visc_conv_mult(U, Ce_all, Se_all, Le, mu, 
                                                                          rho, nconn, nn, ndim)
            if has_dirichlet:
                mask = jnp.zeros_like(U, dtype=bool).at[gamma_d].set(True)
                result = jnp.where(mask, U, result)
            
            return result
        
        rhs = ((rho/dt) * M * U_n) - emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim) + F

    if has_dirichlet:
        rhs = rhs.at[gamma_d].set(U_d)
        x0 = U_n.at[gamma_d].set(U_d) # better initial guess if BC not applied at last step
    else:
        x0 = U_n

    U_hat, U_iters = bicgstab(lhs_operator, rhs, x0=x0, tol=tol)

    return U_hat, U_iters

def solve_pressure(U_star, P_n, Le, Ge, rho, dt, nconn, nn, ndim, gamma_p, F_p, set_zeroP, precond, M_op, tol, gamma):
    """ Solves a Poisson equation with conjugate gradient preconditioner given by M_op.
        gamma controls if a P_n term appears on the rhs.
        The boundary conditions are: P = 0 on Gamma_p
    """
    def lhs_operator(P):
        result = emult.laplacian_element_mult(P, Le, nconn, nn)
        if set_zeroP:
            result = result.at[gamma_p].set(0)
        return result
    
    if gamma == 1:
        rhs = ((rho/dt) * (emult.divergence_element_mult(U_star, Ge, nconn, nn, ndim) - F_p) 
            + emult.laplacian_element_mult(P_n, Le, nconn, nn))
    elif gamma == 0:
        rhs = (rho/dt) * (emult.divergence_element_mult(U_star, Ge, nconn, nn, ndim) - F_p) 

    if set_zeroP:
        rhs = rhs.at[gamma_p].set(0)
        x0 = P_n.at[gamma_p].set(0) # important that P always = 0 on gamma_p
    else:
        rhs = rhs - jnp.mean(rhs) # ensure orthogonal to Ker(L)
        x0 = P_n

    if precond == "approx_inv":
        M_op = partial(M_op, nconn=nconn) # ensure nconn is not treated as static
    # elif precond == "multigrid":
    #     M_op = partial(M_op, nconn_lvls=nconn_lvls) # seems to halve compile time, but iterations are much slower

    if M_op is None:
        P, P_iters = cg(lhs_operator, rhs, x0=x0, tol=tol, maxiter=1000)
    else:
        P, P_iters = cg(lhs_operator, rhs, x0=x0, tol=tol, M=M_op, maxiter=1000)

    # final modification - most likely unnecessary in both cases
    if set_zeroP:
        P = P.at[gamma_p].set(0)
    else:
        P = P - jnp.mean(P)

    return P, P_iters 

def three_step_timestep(U_n, P_n, M, Me, Le, Ge, rho, mu, dt, nconn, nn, ndim, wq, N, Nderiv, gamma_d, U_d, 
                        gamma_p, F_p, F, set_zeroP, p_precond, p_M_op, tol, theta, gamma):
    """ theta = 1/2: Crank-Nicolson, theta=1: backward Euler
        gamma = 1: P_n term in the momentum equation, gamma = 0: no P_n term 
    """
    Ce_all_n = ecalc.conservative_convec_elem_calc(nconn, U_n, wq, N, Nderiv, nn, ndim, 0.5)

    U_star, U_iters = solve_momentum(U_n, P_n, Me, Le, Ge, Ce_all_n, rho, mu, dt, nconn, nn, ndim, gamma_d, U_d, F, tol, theta, gamma)
    
    P, P_iters = solve_pressure(U_star, P_n, Le, Ge, rho, dt, nconn, nn, ndim, gamma_p, F_p, set_zeroP, p_precond, p_M_op, tol, gamma)

    if gamma == 0:
        U =  U_star - ((dt/rho) * emult.gradient_element_mult(P, Ge, nconn, nn, ndim) / M)
    elif gamma == 1:
        U =  U_star - ((dt/rho) * emult.gradient_element_mult(P - P_n, Ge, nconn, nn, ndim) / M)
    
    if gamma_d is not None:
        U = U.at[gamma_d].set(U_d)

    return U, P, U_iters, P_iters

def four_step_timestep(U_n, P_n, M, Me, Le, Ge, rho, mu, dt, nconn, nn, ndim, wq, N, Nderiv, gamma_d, U_d, 
                        gamma_p, F_p, F, set_zeroP, p_precond, p_M_op, tol, theta):
    """ Adds an extra step to the 3-step method, first seen in Choi et al. 1997.
    """
    Ce_all_n = ecalc.conservative_convec_elem_calc(nconn, U_n, wq, N, Nderiv, nn, ndim, 0.5)

    U_hat, U_iters = solve_momentum(U_n, P_n, Me, Le, Ge, Ce_all_n, rho, mu, dt, nconn, nn, ndim, gamma_d, U_d, F, tol, theta, gamma=1)

    U_star =  U_hat + ((dt/rho) * emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim) / M)
    
    P, P_iters = solve_pressure(U_star, P_n, Le, Ge, rho, dt, nconn, nn, ndim, gamma_p, F_p, set_zeroP, p_precond, p_M_op, tol, gamma=0)

    U =  U_star - ((dt/rho) * emult.gradient_element_mult(P, Ge, nconn, nn, ndim) / M)

    if gamma_d is not None:
        U = U.at[gamma_d].set(U_d)

    return U, P, U_iters, P_iters

@partial(jax.jit, static_argnames=("boundary_shapefunc", "nn", "ndim", "S_func"))
def compute(node_coords, bconn, nvec, boundary_shapefunc, gamma_d, U_d, nn, ndim, S_func, Me, nconn, t_n, t):
    """ Compute the boundary integral that depends on U_d and the momentum source term
    """
    F, F_p = 0, 0

    if gamma_d is not None:
        if ndim == 2:
            F_p = ecalc.u_dot_n_boundary_integral_1d(bconn, nvec, gamma_d, U_d, nn, node_coords, 
                                                     boundary_shapefunc)
        elif ndim == 3:
            F_p = ecalc.u_dot_n_boundary_integral_2d(bconn, nvec, gamma_d, U_d, nn, node_coords,
                                                     boundary_shapefunc)

    if S_func is not None:
        S_n = S_func(t_n)
        S = S_func(t)
        S = (S_n + S) / 2 # assuming crank-nicolson

        F = emult.consistent_mass_mult(S, Me, nconn, nn, ndim)

    return F_p, F


def setup_solver(node_coords, nconn, bconn, nvec, gamma_d, gamma_p, nn, mesh_func, config: Config):

    mesh_config = config.mesh_config
    NS_config = config.solver_config
    ndim = len(mesh_config.num_elem)

    if ndim == 2:
        if mesh_config.shape_func_ord == 1:
            shape_func = shapefunc.quad4_shape_functions
            boundary_shapefunc = shapefunc.lin2_shape_functions
        
        elif mesh_config.shape_func_ord == 2:
            shape_func = shapefunc.quad9_shape_functions
            boundary_shapefunc = shapefunc.lin3_shape_functions

    elif ndim == 3:
        if mesh_config.shape_func_ord == 1: # add 2 case in future
            shape_func = shapefunc.hex8_shape_functions
            boundary_shapefunc = shapefunc.quad4_shape_functions

    wq, N, Nderiv = shape_func(node_coords[nconn[0]])
    M, Me = ecalc.lumped_mass_matrix(nconn, wq, N, nn, ndim)
    Le = ecalc.laplacian_element_calc(wq, Nderiv)
    Ge = ecalc.gradient_element_calc(wq, N, Nderiv, ndim)

    nconn_lvls = None
    pressure_M_op = None

    if NS_config.pressure_precond == "jacobi":
        L_diag = ecalc.laplacian_diag(Le, nconn, nn)

        def pressure_M_op(r):
            return r / L_diag
            
    elif NS_config.pressure_precond == "approx_inv":
        L_diag = ecalc.laplacian_diag(Le, nconn, nn)

        def pressure_M_op(r, nconn):
            D1x = r / L_diag
            D1L = emult.laplacian_element_mult(D1x, Le, nconn, nn) / L_diag
            D1L = D1L.at[gamma_p].set(0)
            M1x = (3*D1x - 3*D1L + emult.laplacian_element_mult(D1L, Le, nconn, nn
                    ).at[gamma_p].set(0) / L_diag)
            return M1x
        
    elif NS_config.pressure_precond == "multigrid":
        mg_data = mg.multigrid_setup(mesh_config.num_elem, NS_config.final_multigrid_mesh, mesh_config.domain_size, NS_config.outlet_faces,
                                     mesh_func, shape_func, mesh_config.shape_func_ord, mesh_config.partial_dirichlet)
        lvl_nums, Le_lvls, nconn_lvls, L_diag_lvls, gamma_p_lvls = mg_data
        
        # @jax.jit
        # def pressure_M_op(r, nconn_lvls):
        #     return mg.v_cycle(r, lvl_nums, Le_lvls, nconn_lvls, L_diag_lvls, gamma_p_lvls, tol)
        @jax.jit
        def pressure_M_op(r):
            return mg.v_cycle(r, lvl_nums, Le_lvls, nconn_lvls, L_diag_lvls, gamma_p_lvls, NS_config.solver_tol, 
                              mesh_config.periodic, NS_config.set_zeroP)

    def compute_loaded(U_d, S_func, t_n, t):
        return compute(node_coords, bconn, nvec, boundary_shapefunc, gamma_d, U_d, nn, ndim, S_func, Me, nconn, t_n, t)

    theta = 0.5
    gamma = 1

    # treat large arrays like nconn as dynamic to lower compile time. if not, constant folding results in very long compile times
    @partial(jax.jit, static_argnames=("pressure_M_op"))
    def timestep_jitted(U_n, P_n, dt, rho, mu, U_d, F_p, nconn, pressure_M_op, F):
        return four_step_timestep(U_n, P_n, M, Me, Le, Ge, rho, mu, dt, nconn, nn, ndim, wq, N, Nderiv, gamma_d, U_d, 
                                  gamma_p, F_p, F, NS_config.set_zeroP, NS_config.pressure_precond, pressure_M_op, NS_config.solver_tol, theta)
    
        # return three_step_timestep(U_n, P_n, M, Me, Le, Ge, rho, mu, dt, nconn, nn, ndim, wq, N, Nderiv, gamma_d, U_d, 
                                #    gamma_p, F_p, F, set_zeroP, p_precond, pressure_M_op, tol, theta, gamma)

    def timestep_loaded(U_n, P_n, dt, rho, mu, U_d, F_p, F):
        return timestep_jitted(U_n, P_n, dt, rho, mu, U_d, F_p, nconn, pressure_M_op, F)

    return compute_loaded, timestep_loaded

