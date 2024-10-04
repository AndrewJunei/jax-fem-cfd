""" Incompressible, Newtonian Navier Stokes with no body force
"""
import jax
import jax.numpy as jnp
from..core import shape_functions as shapefunc
from ..core import element_calc as ecalc
from ..core import element_mult as emult
from ..core.jax_iterative_solvers import bicgstab, cg


def precompute(wq, N, Nderiv, bwq, bN, nconn, nn, ndim, p_precond, bconn, nvec, 
               rho, gamma_d, U_d):
    # update later with new preconditioning
    M = ecalc.lumped_mass_matrix(nconn, rho, wq, N, nn)

    Le = ecalc.laplacian_element_calc(wq, Nderiv)
    Ge = ecalc.gradient_element_calc(wq, N, Nderiv, ndim)

    if ndim == 2:
        M = jnp.concatenate([M, M])
        F = ecalc.u_dot_n_boundary_integral_1d(bconn, nvec, gamma_d, U_d, bwq, bN, nn)

    elif ndim == 3:
        M = jnp.concatenate([M, M, M])
        F = ecalc.u_dot_n_boundary_integral_2d(bconn, nvec, gamma_d, U_d, bwq, bN, nn)

    if p_precond == "jacobi" or "approx_inv":
        L_diag = ecalc.laplacian_diag(Le, nconn, nn)
    
    else: 
        L_diag = None

    return M, Le, Ge, F, L_diag

def solve_momentum(U_n, P_n, M, Le, Ge, Ce_all, mu, dt, nconn, gamma_d, U_d, nn, ndim):
    """ Solves the momentum equation with pressure from the previous timestep using the BiCGSTAB
        iterative solver. 
        The boundary conditions are: U_hat = U_d on Gamma_d,
                                     grad(u) * n = 0,
                                     grad(v) * n = 0 on Gamma \ Gamma_d
    """
    def lhs_operator(U):
        result = (M * U / dt) + emult.visc_conv_mult(U, Ce_all, Le, mu, nconn, nn, ndim)

        mask = jnp.zeros_like(U, dtype=bool).at[gamma_d].set(True)
        result = jnp.where(mask, U, result)
        return result
    
    rhs = (M * U_n / dt) - emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim)
    rhs = rhs.at[gamma_d].set(U_d)

    x0 = U_n.at[gamma_d].set(U_d) # better initial guess if BC not applied at last step

    U_hat, U_iters = bicgstab(lhs_operator, rhs, x0=x0, tol=1e-6)

    return U_hat, U_iters

def solve_momentum_stab(U_n, P_n, M, Le, Ge, Ce_all, Se_all, mu, dt, nconn, gamma_d, U_d, nn, ndim):
    """ Same as solve_momentum but with an articial streamline diffusion
    """
    def lhs_operator(U):
        result = (M * U / dt) + emult.stabilized_visc_conv_mult(U, Ce_all, Se_all, Le, mu, nconn, 
                                                                nn, ndim)

        mask = jnp.zeros_like(U, dtype=bool).at[gamma_d].set(True)
        result = jnp.where(mask, U, result)
        return result
    
    rhs = (M * U_n / dt) - emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim)
    rhs = rhs.at[gamma_d].set(U_d)

    x0 = U_n.at[gamma_d].set(U_d) # better initial guess if BC not applied at last step

    U_hat, U_iters = bicgstab(lhs_operator, rhs, x0=x0, tol=1e-6)

    return U_hat, U_iters

def solve_pressure(U_star, P_n, Le, Ge, rho, dt, nconn, gamma_p, F, nn, ndim):
    """ Solves a pressure poisson equation for the future pressure using the conjugate gradient
        iterative solver. 
        The boundary conditions are: P = 0 on Gamma_p,
                                     U = U_d on Gamma_d,
                                     U * n = 0 on (Gamma \ Gamma_d) \ Gamma_p
    """
    def lhs_operator(P):
        result = emult.laplacian_element_mult(P, Le, nconn, nn)
        return result.at[gamma_p].set(P[gamma_p]) 
    
    rhs = (rho/dt) * (emult.divergence_element_mult(U_star, Ge, nconn, nn, ndim) - F)
    rhs = rhs.at[gamma_p].set(0)
    
    P, P_iters = cg(lhs_operator, rhs, x0=P_n, tol=1e-6)
    P = P.at[gamma_p].set(0) # deal with potential round off errors (could be unnecessary)

    return P, P_iters

def solve_pressure_jacobi(U_star, P_n, Le, Ge, rho, dt, nconn, gamma_p, F, nn, ndim, L_diag):
    """ Same as solve_pressure but with the Jacobi preconditioner
    """
    def lhs_operator(P):
        result = emult.laplacian_element_mult(P, Le, nconn, nn)
        return result.at[gamma_p].set(P[gamma_p]) 
    
    def M_operator(x):
        return x / L_diag
    
    rhs = (rho/dt) * (emult.divergence_element_mult(U_star, Ge, nconn, nn, ndim) - F)
    rhs = rhs.at[gamma_p].set(0)
    
    P, P_iters = cg(lhs_operator, rhs, x0=P_n, tol=1e-6, M=M_operator)
    P = P.at[gamma_p].set(0) # deal with potential round off errors (could be unnecessary)

    return P, P_iters

def solve_pressure_approx_inv(U_star, P_n, Le, Ge, rho, dt, nconn, gamma_p, F, nn, ndim, L_diag):
    """ Same as solve_pressure but with a series approximate inverse preconditioner
    """
    def lhs_operator(P):
        result = emult.laplacian_element_mult(P, Le, nconn, nn)
        return result.at[gamma_p].set(P[gamma_p]) 
    
    def M_operator(x):
        D1x = x / L_diag
        D1L = emult.laplacian_element_mult(D1x, Le, nconn, nn) / L_diag
        D1L = D1L.at[gamma_p].set(D1x[gamma_p])
        M1x = (3*D1x - 3*D1L + 
               emult.laplacian_element_mult(D1L, Le, nconn, nn).at[gamma_p].set(D1L[gamma_p]) / L_diag)
        return M1x
    
    rhs = (rho/dt) * (emult.divergence_element_mult(U_star, Ge, nconn, nn, ndim) - F)
    rhs = rhs.at[gamma_p].set(0)
    
    P, P_iters = cg(lhs_operator, rhs, x0=P_n, tol=1e-6, M=M_operator)
    P = P.at[gamma_p].set(0) # deal with potential round off errors (could be unnecessary)

    return P, P_iters

def four_step_timestep(U_n, U_n_1, P_n, M, Le, Ge, rho, mu, h, dt, nconn, wq, N, Nderiv, 
                       gamma_d, U_d, gamma_p, F, nn, ndim, stab, pressure_precond,
                       L_diag=None):
    """ Uses the algorithm of Choi et al. 1997.
        Some notable things: to deal with advection oscillations, an artificial streamline diffusion 
        is used, but the full, consistent SUPG terms are not. The convection and streamline 
        diffusion matrices are computed using the linearly extrapolated previous velocity.
        To make this algorithm more accurate but also more compuationally expensive, one can 
        perform a nonlinear solve in the provisional velocity step (eg, Newton's method) and
        iterate the entire process multiple times each timestep until the solution satisfies both
        the momentum equation and is incompressible.
    """
    U_double_star = 2*U_n - U_n_1

    Ce_all = ecalc.convection_element_calc(nconn, rho, U_double_star, wq, N, Nderiv, nn, ndim) 

    if stab:
        Se_all = ecalc.streamline_diff_element_calc(nconn, h, mu, rho, U_double_star, wq, N, Nderiv, 
                                                    nn, ndim)

        U_hat, U_iters = solve_momentum_stab(U_n, P_n, M, Le, Ge, Ce_all, Se_all, mu, dt, nconn,
                                             gamma_d, U_d, nn, ndim)
    else:
        U_hat, U_iters = solve_momentum(U_n, P_n, M, Le, Ge, Ce_all, mu, dt, nconn,
                                        gamma_d, U_d, nn, ndim)
    
    U_star =  U_hat + (dt * emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim) / M)
    # U_star = U_star.at[gamma_d].set(U_d) # check

    if pressure_precond is None:
        P, P_iters = solve_pressure(U_star, P_n, Le, Ge, rho, dt, nconn, gamma_p, F, nn, ndim)

    elif pressure_precond == "jacobi":
        P, P_iters = solve_pressure_jacobi(U_star, P_n, Le, Ge, rho, dt, nconn, gamma_p, F, 
                                           nn, ndim, L_diag)
        
    elif pressure_precond == "approx_inv":
        P, P_iters = solve_pressure_approx_inv(U_star, P_n, Le, Ge, rho, dt, nconn, gamma_p, F, 
                                               nn, ndim, L_diag)

    U =  U_star - (dt * emult.gradient_element_mult(P, Ge, nconn, nn, ndim) / M)
    # U = U.at[gamma_d].set(U_d) # check

    return U, P, U_iters, P_iters


def setup_solver(node_coords, nconn, bconn, nvec, gamma_d, gamma_p, nn, ndim, h,
                 stab, p_precond, shape_func_ord):

    if shape_func_ord == 1:
        if ndim == 2:
            wq, N, Nderiv = shapefunc.quad4_shape_functions(node_coords[nconn[0]])
            bwq, bN, _ = shapefunc.lin2_shape_functions(node_coords[bconn[0]])

        elif ndim == 3:
            wq, N, Nderiv = shapefunc.hex8_shape_functions(node_coords[nconn[0]])
            # (z, y) coords, check for more general case
            bwq, bN, _ = shapefunc.quad4_shape_functions(node_coords[bconn[0]][:, [2, 1]]) 
    
    elif shape_func_ord == 2:
        if ndim == 2:
            wq, N, Nderiv = shapefunc.quad9_shape_functions(node_coords[nconn[0]])
            bwq, bN, _ = shapefunc.lin3_shape_functions(node_coords[bconn[0]])

    @jax.jit
    def precompute_loaded(rho, U_d):
        return precompute(wq, N, Nderiv, bwq, bN, nconn, nn, ndim, p_precond, bconn, nvec, 
                          rho, gamma_d, U_d)
    
    @jax.jit
    def timestep_loaded(U_n, U_n_1, P_n, M, Le, Ge, F, L_diag, dt, rho, mu, U_d):
        return four_step_timestep(U_n, U_n_1, P_n, M, Le, Ge, rho, mu, h, dt, nconn, wq, N, Nderiv, 
                                  gamma_d, U_d, gamma_p, F, nn, ndim, stab, p_precond, L_diag)
    
    return precompute_loaded, timestep_loaded
    


