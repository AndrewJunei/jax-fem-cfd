""" 3D incompressible, Newtonian Navier Stokes with no body force
"""
import jax
import jax.numpy as jnp
from ..core.linear_shape_functions import hex_3d_shape_functions, quad_2d_shape_functions
from ..core.element_calc import *
from ..core.element_mult import stabilized_visc_conv_mult_3d, gradient_element_mult_3d, \
                                    divergence_element_mult_3d, laplacian_element_mult
from ..core.jax_iterative_solvers import gmres, cg

@jax.jit
def setup_solver(node_coords, nconn, bconn, normal_vec, rho, not_gamma_p, gamma_d, U_d):
    """ Pre computes everything needed in the timestepping loop that remains constant. 
        Note that we assume a regular mesh is used. However, the routines here can be easily 
        modified to accomodate an unstructured mesh (with some extra expense).
    """
    nn = node_coords.shape[0]
    wq, N, Nderiv = hex_3d_shape_functions(node_coords[nconn[0]])

    wq_2d, N_2d, _ = quad_2d_shape_functions(node_coords[bconn[0]][:, [2, 1]]) # (z, y) coords

    M = lumped_mass_matrix(nconn, rho, wq, N, nn)
    M = jnp.concatenate([M, M, M])
    Le = laplacian_element_calc(wq, Nderiv)
    Ge = gradient_element_calc(wq, N, Nderiv, ndim=3)

    F = u_dot_n_boundary_integral_2d(bconn, normal_vec, gamma_d, U_d, wq_2d, N_2d, nn)

    L_diag = laplacian_diag(Le, nconn, not_gamma_p, nn)

    return wq, N, Nderiv, M, Le, Ge, F, L_diag


def provisional_velocity(U_n, P_n, M, Le, Ge, Ce_all, Se_all, mu, dt, nconn, gamma_d, U_d, nn):
    """ Solves the momentum equation with pressure from the previous timestep using the GMRES
        iterative solver. 
        The boundary conditions are: U_hat = U_d on Gamma_d,
                                     grad(u) * n = 0,
                                     grad(v) * n = 0,
                                     grad(w) * n = 0 on Gamma \ Gamma_d
    """
    def lhs_operator(U):
        result = M * U / dt + stabilized_visc_conv_mult_3d(U, Ce_all, Se_all, Le, mu, nconn, nn)

        mask = jnp.zeros_like(U, dtype=bool).at[gamma_d].set(True)
        result = jnp.where(mask, U, result)
        return result
    
    rhs = M * U_n / dt - gradient_element_mult_3d(P_n, Ge, nconn, nn)
    rhs = rhs.at[gamma_d].set(U_d)

    x0 = U_n.at[gamma_d].set(U_d) # better initial guess if BC not applied at last step

    U_hat, gmres_iters = gmres(lhs_operator, rhs, x0=x0, tol=1e-7)

    return U_hat, gmres_iters


def expand(P_condensed, P_n, not_gamma_p):
    return jnp.zeros_like(P_n).at[not_gamma_p].set(P_condensed) # zero at gamma_p

def condensed_laplacian_mult(P_condensed, P_n, not_gamma_p, Le, nconn, nn):
    """ Here we use static condensation to keep the "matrix" symmetric after applying boundary 
        conditions. Since the values are always 0, the process is simplified. 
    """
    P_full = expand(P_condensed, P_n, not_gamma_p) # use full P for matrix mult
    result = laplacian_element_mult(P_full, Le, nconn, nn)
    return result[not_gamma_p] # back to condensed


def update_pressure(U_star, P_n, Le, Ge, rho, dt, nconn, not_gamma_p, F, F_idx, nn):
    """ Solves a pressure poisson equation for the future pressure using the conjugate gradient
        iterative solver. 
        The boundary conditions are: P = 0 on Gamma_p,
                                     U = U_d on Gamma_d,
                                     U * n = 0 on (Gamma \ Gamma_d) \ Gamma_p
    """
    def lhs_operator(P_condensed):
        return condensed_laplacian_mult(P_condensed, P_n, not_gamma_p, Le, nconn, nn)
    
    rhs = (rho/dt) * divergence_element_mult_3d(U_star, Ge, nconn, nn)
    rhs = rhs.at[F_idx].add((-rho/dt) * F)
    rhs = rhs[not_gamma_p]
    
    P_condensed, cg_iters = cg(lhs_operator, rhs, x0=P_n[not_gamma_p], tol=1e-7)

    return expand(P_condensed, P_n, not_gamma_p), cg_iters

""" Still under development """
def precond_update_P(U_star, P_n, Le, Ge, rho, dt, nconn, not_gamma_p, F, F_idx, L_diag, nn):
    """ Since the laplacian matrix is poorly conditioned, the conjugate gradient step ends up 
        taking longer than the gmres step typically. Preconditioning is usually a simple fix
        for this. However, I have tried many options and have not come up with a preconditioner
        that is 1) cheap to compute, 2) parallelizable, 3) matrix free, and 4) effective. 
        For example, jacobi satisfies 1, 2, and 3 but is basically useless.
        In the future, I will try multigrid methods.
    """
    def lhs_operator(P_condensed):
        return condensed_laplacian_mult(P_condensed, P_n, not_gamma_p, Le, nconn, nn)
    
    """ Series approx preconditioner """
    def M_operator(x):
        D1x = x / L_diag
        D1L = condensed_laplacian_mult(D1x, P_n, not_gamma_p, Le, nconn, nn) / L_diag
        M1x = (3*D1x - 3*D1L + 
               (condensed_laplacian_mult(D1L, P_n, not_gamma_p, Le, nconn, nn) / L_diag))
        # M1x = 2*D1x - D1L
        return M1x
    
    """ Jacobi preconditioner """
    # def M_operator(x):
    #     # return x / L_diag
    
    rhs = (rho/dt) * divergence_element_mult_3d(U_star, Ge, nconn, nn)
    rhs = rhs.at[F_idx].add((-rho/dt) * F)
    rhs = rhs[not_gamma_p]
    
    P_condensed, cg_iters = cg(lhs_operator, rhs, x0=P_n[not_gamma_p], tol=1e-7, M=M_operator)

    return expand(P_condensed, P_n, not_gamma_p), cg_iters


@partial(jax.jit, static_argnames=("nn"))
def four_step_timestep(U_n, U_n_1, P_n, M, Le, Ge, rho, mu, h, dt, nconn, wq, N, Nderiv, 
                       gamma_d, U_d, not_gamma_p, F, F_idx, L_diag, nn):
    """ Uses the algorithm of Choi et al. 1997.
        Some notable things: to deal with advection oscillations, an artificial streamline is
        used, but the full, consistent SUPG terms are not. The convection and streamline 
        diffusion matrices are computed using the linearly extrapolated previous velocity.
        To make this algorithm more accurate but also more compuationally expensive, one can 
        perform a nonlinear solve in the provisional velocity step (eg, Newton's method) and
        iterate the entire process multiple times each timestep until the solution satisfies both
        the momentum equation and is incompressible.
    """
    U_double_star = 2*U_n - U_n_1

    Ce_all = convection_element_calc(nconn, rho, U_double_star, wq, N, Nderiv, nn, ndim=3) 

    Se_all = streamline_diff_element_calc(nconn, h, mu, rho, U_double_star, wq, N, Nderiv, 
                                          nn, ndim=3)

    U_hat, gmres_iters = provisional_velocity(U_n, P_n, M, Le, Ge, Ce_all, Se_all, mu, dt, nconn, 
                                              gamma_d, U_d, nn)
    
    U_star =  U_hat + dt * gradient_element_mult_3d(P_n, Ge, nconn, nn) / M
    # U_star = U_star.at[gamma_d].set(U_d) # check

    P, cg_iters = update_pressure(U_star, P_n, Le, Ge, rho, dt, nconn, not_gamma_p, F, F_idx, nn)
    # P, cg_iters = precond_update_P(U_star, P_n, Le, Ge, rho, dt, nconn, not_gamma_p, F, F_idx, L_diag, nn)

    U =  U_star - dt * gradient_element_mult_3d(P, Ge, nconn, nn) / M
    # U = U.at[gamma_d].set(U_d) # check

    return U, P, gmres_iters, cg_iters