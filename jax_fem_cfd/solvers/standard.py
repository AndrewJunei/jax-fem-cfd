""" Incompressible, Newtonian Navier Stokes with no body force
"""
import jax
import jax.numpy as jnp
from functools import partial
from..core import shape_functions as shapefunc
from ..core import element_calc as ecalc
from ..core import element_mult as emult
from ..utils.jax_iterative_solvers import bicgstab, cg
from ..core import multigrid as mg


def solve_momentum(U_n, P_n, M, Le, Ge, Ce_all, rho, mu, dt, nconn, gamma_d, U_d, nn, ndim, tol, 
                   crank_nicol, has_dirichlet):
    """ Solves the momentum equation with pressure from the previous timestep using the BiCGSTAB
        iterative solver. 
        The boundary conditions are: U_hat = U_d on Gamma_d,
                                     grad(u) * n = 0 on Gamma \ Gamma_d,
                                     grad(v) * n = 0 on Gamma \ Gamma_d
    """
    if crank_nicol:
        def lhs_operator(U):
            result = ((rho/dt) * M * U) + emult.visc_conv_mult(U, Ce_all, Le, mu, rho, nconn, nn, ndim) / 2
            
            if has_dirichlet:
                mask = jnp.zeros_like(U, dtype=bool).at[gamma_d].set(True)
                result = jnp.where(mask, U, result)
            
            return result
        
        rhs = (((rho/dt) * M * U_n) - emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim) 
               - emult.visc_conv_mult(U_n, Ce_all, Le, mu, rho, nconn, nn, ndim) / 2)
    
    else:
        def lhs_operator(U):
            result = ((rho/dt) * M * U) + emult.visc_conv_mult(U, Ce_all, Le, mu, rho, nconn, nn, ndim)
            
            if has_dirichlet:
                mask = jnp.zeros_like(U, dtype=bool).at[gamma_d].set(True)
                result = jnp.where(mask, U, result)

            return result
    
        rhs = ((rho/dt) * M * U_n) - emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim)
    
    if has_dirichlet:
        rhs = rhs.at[gamma_d].set(U_d)
        x0 = U_n.at[gamma_d].set(U_d) # better initial guess if BC not applied at last step
    else:
        x0 = U_n

    U_hat, U_iters = bicgstab(lhs_operator, rhs, x0=x0, tol=tol)

    return U_hat, U_iters

def solve_momentum_stab(U_n, P_n, M, Le, Ge, Ce_all, Se_all, rho, mu, dt, nconn, gamma_d, U_d, nn, ndim, 
                        tol, crank_nicol, has_dirichlet):
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
               - emult.stabilized_visc_conv_mult(U_n, Ce_all, Se_all, Le, mu, rho, nconn, nn, ndim) / 2)

    else:
        def lhs_operator(U):
            result = ((rho/dt) * M * U) + emult.stabilized_visc_conv_mult(U, Ce_all, Se_all, Le, mu, 
                                                                          rho, nconn, nn, ndim)
            if has_dirichlet:
                mask = jnp.zeros_like(U, dtype=bool).at[gamma_d].set(True)
                result = jnp.where(mask, U, result)
            
            return result
        
        rhs = ((rho/dt) * M * U_n) - emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim)

    if has_dirichlet:
        rhs = rhs.at[gamma_d].set(U_d)
        x0 = U_n.at[gamma_d].set(U_d) # better initial guess if BC not applied at last step
    else:
        x0 = U_n

    U_hat, U_iters = bicgstab(lhs_operator, rhs, x0=x0, tol=tol)

    return U_hat, U_iters

def solve_pressure(U_star, P_n, Le, Ge, rho, dt, nconn, gamma_p, F, nn, ndim, precond, M_op, tol, set_zeroP):
    """ Solves a pressure poisson equation for the future pressure using the conjugate gradient
        iterative solver. The preconditioner is given by M_op.
        The boundary conditions are: P = 0 on Gamma_p,
                                     U = U_d on Gamma_d,
                                     U * n = 0 on (Gamma \ Gamma_d) \ Gamma_p
    """
    if set_zeroP:
        def lhs_operator(P):
            return emult.laplacian_element_mult(P, Le, nconn, nn).at[gamma_p].set(0)

        rhs = (rho/dt) * (emult.divergence_element_mult(U_star, Ge, nconn, nn, ndim) - F)
        rhs = rhs.at[gamma_p].set(0)

        x0 = P_n.at[gamma_p].set(0) # important that P always = 0 on gamma_p
    
    else:
        def lhs_operator(P):
            return emult.laplacian_element_mult(P, Le, nconn, nn)

        rhs = (rho/dt) * (emult.divergence_element_mult(U_star, Ge, nconn, nn, ndim) - F)
        rhs = rhs - jnp.mean(rhs) # ensure orthogonal to Ker(L)

        x0 = P_n

    if precond == "approx_inv":
        M_op = partial(M_op, nconn=nconn) # ensure nconn is not treated as static
    # elif precond == "multigrid":
    #     M_op = partial(M_op, nconn_lvls=nconn_lvls) # CHECK
        # seems to halve compile time and not give warning, but iterations are much slower
    
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

def four_step_timestep(U_n, U_n_1, P_n, M, Le, Ge, rho, mu, h, dt, nconn, wq, N, Nderiv, 
                       gamma_d, U_d, gamma_p, F, nn, ndim, stab, p_precond, pressure_M_op, tol,
                       crank_nicol, has_dirichlet, set_zeroP):
    """ Uses the algorithm of Choi et al. 1997.
        Some notable things: to deal with advection oscillations, an artificial streamline diffusion 
        is used, but the full, consistent SUPG terms are not. The convection and streamline 
        diffusion matrices are computed using the linearly extrapolated previous velocity.
        To make this algorithm more accurate but also more compuationally expensive, one can 
        perform a nonlinear solve in the provisional velocity step (eg, Newton's method) and
        iterate the entire process multiple times each timestep until the solution satisfies both
        the momentum equation and is incompressible.
    """
    # if crank_nicol:
    #     U_double_star = U_n
    # else:
    #     U_double_star = 2*U_n - U_n_1
    U_double_star = U_n

    Ce_all = ecalc.convection_element_calc(nconn, U_double_star, wq, N, Nderiv, nn, ndim) 

    if stab:
        Se_all = ecalc.streamline_diff_element_calc(nconn, h, mu, rho, U_double_star, wq, N, Nderiv, 
                                                    nn, ndim)

        U_hat, U_iters = solve_momentum_stab(U_n, P_n, M, Le, Ge, Ce_all, Se_all, rho, mu, dt, nconn,
                                             gamma_d, U_d, nn, ndim, tol, crank_nicol, has_dirichlet)
    else:
        U_hat, U_iters = solve_momentum(U_n, P_n, M, Le, Ge, Ce_all, rho, mu, dt, nconn, 
                                        gamma_d, U_d, nn, ndim, tol, crank_nicol, has_dirichlet)

    U_star =  U_hat + ((dt/rho) * emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim) / M)
    # if has_dirichlet:
        # U_star = U_star.at[gamma_d].set(U_d) # questionable

    P, P_iters = solve_pressure(U_star, P_n, Le, Ge, rho, dt, nconn, gamma_p, F, nn, ndim, 
                                p_precond, pressure_M_op, tol, set_zeroP)

    U =  U_star - ((dt/rho) * emult.gradient_element_mult(P, Ge, nconn, nn, ndim) / M)
    if has_dirichlet:
        U = U.at[gamma_d].set(U_d) # questionable

    return U, P, U_iters, P_iters

@partial(jax.jit, static_argnames=("boundary_shapefunc", "nn", "ndim", "has_dirichlet"))
def compute(node_coords, bconn, nvec, boundary_shapefunc, gamma_d, U_d, nn, ndim, has_dirichlet):
    """ Compute the boundary integral that depends on U_d.
        In future, could include source terms, changing material properties, and time dependent
        boundary conditions.
    """
    if has_dirichlet:
        if ndim == 2:
            F = ecalc.u_dot_n_boundary_integral_1d(bconn, nvec, gamma_d, U_d, nn, node_coords, 
                                                boundary_shapefunc)
        elif ndim == 3:
            F = ecalc.u_dot_n_boundary_integral_2d(bconn, nvec, gamma_d, U_d, nn, node_coords,
                                                boundary_shapefunc)
    else:
        F = 0

    return F


def setup_solver(node_coords, nconn, bconn, nvec, gamma_d, gamma_p, nn, ndim, h, stab, p_precond, 
                 shape_func_ord, num_elem, num_elemf, domain_size, outlet_faces, tol, crank_nicol,
                 mesh_func, periodic, set_zeroP):

    if ndim == 2:
        if shape_func_ord == 1:
            shape_func = shapefunc.quad4_shape_functions
            boundary_shapefunc = shapefunc.lin2_shape_functions
        
        elif shape_func_ord == 2:
            shape_func = shapefunc.quad9_shape_functions
            boundary_shapefunc = shapefunc.lin3_shape_functions

    elif ndim == 3:
        if shape_func_ord == 1: # add 2 case in future
            shape_func = shapefunc.hex8_shape_functions
            boundary_shapefunc = shapefunc.quad4_shape_functions

    has_dirichlet = False if gamma_d is None else True

    wq, N, Nderiv = shape_func(node_coords[nconn[0]])

    M = ecalc.lumped_mass_matrix(nconn, wq, N, nn, ndim)
    Le = ecalc.laplacian_element_calc(wq, Nderiv)
    Ge = ecalc.gradient_element_calc(wq, N, Nderiv, ndim)

    nconn_lvls = None

    if p_precond == "jacobi":
        L_diag = ecalc.laplacian_diag(Le, nconn, nn)

        def pressure_M_op(r):
            return r / L_diag
            
    elif p_precond == "approx_inv":
        L_diag = ecalc.laplacian_diag(Le, nconn, nn)

        def pressure_M_op(r, nconn):
            D1x = r / L_diag
            D1L = emult.laplacian_element_mult(D1x, Le, nconn, nn) / L_diag
            D1L = D1L.at[gamma_p].set(0)
            M1x = (3*D1x - 3*D1L + emult.laplacian_element_mult(D1L, Le, nconn, nn
                    ).at[gamma_p].set(0) / L_diag)
            return M1x
        
    elif p_precond == "multigrid":
        mg_data = mg.multigrid_setup(num_elem, num_elemf, domain_size, outlet_faces,
                                     mesh_func, shape_func, shape_func_ord)

        lvl_nums, Le_lvls, nconn_lvls, L_diag_lvls, gamma_p_lvls = mg_data
        
        # TODO: look more into issue with constant folding, still gives warning sometimes
        # @jax.jit
        # def pressure_M_op(r, nconn_lvls):
        #     return mg.v_cycle(r, lvl_nums, Le_lvls, nconn_lvls, L_diag_lvls, gamma_p_lvls, tol)
        @jax.jit
        def pressure_M_op(r):
            return mg.v_cycle(r, lvl_nums, Le_lvls, nconn_lvls, L_diag_lvls, gamma_p_lvls, tol, 
                              periodic, set_zeroP)
    
    else:
        pressure_M_op = None

    def compute_loaded(U_d):
        return compute(node_coords, bconn, nvec, boundary_shapefunc, gamma_d, U_d, nn, ndim, has_dirichlet)
    
    # important! treat large arrays like nconn as dynamic to lower compile time. if not, 
    # constant folding results in very long compile times
    @partial(jax.jit, static_argnames=("pressure_M_op"))
    def timestep_jitted(U_n, U_n_1, P_n, dt, rho, mu, U_d, F, nconn, pressure_M_op):
        return four_step_timestep(U_n, U_n_1, P_n, M, Le, Ge, rho, mu, h, dt, nconn, wq, N, Nderiv, 
                                  gamma_d, U_d, gamma_p, F, nn, ndim, stab, 
                                  p_precond, pressure_M_op, tol, crank_nicol, has_dirichlet, set_zeroP)
    
    def timestep_loaded(U_n, U_n_1, P_n, dt, rho, mu, U_d, F):
        return timestep_jitted(U_n, U_n_1, P_n, dt, rho, mu, U_d, F, nconn, pressure_M_op)

    return compute_loaded, timestep_loaded

