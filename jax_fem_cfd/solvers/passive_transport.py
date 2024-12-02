import jax
import jax.numpy as jnp
from functools import partial
from..core import shape_functions as shapefunc
from ..setup import regular_mesh as rm
from ..core import element_calc as ecalc
from ..core import element_mult as emult
from ..utils.jax_iterative_solvers import bicgstab
from ..core import multigrid as mg
from . import standard
from ..setup.simulation import Config


def update_phi_SUPG(phi_n, nconn, Mtau_all_n, Ctau_all_n, Me, Ce_all, Ce_all_n, Le, D, dt, nn, 
                    gamma_d_phi, phi_d, tol):
    
    def lhs_operator(phi):
        result = ((1/dt) * emult.scalar_consistent_mass_SUPG_mult(phi, Mtau_all_n, Me, nconn, nn)
                + 0.5*emult.scalar_stabilized_visc_conv_mult(phi, Ce_all, Ctau_all_n, Le, D, nconn, nn))
        
        if gamma_d_phi is not None:
            mask = jnp.zeros_like(phi, dtype=bool).at[gamma_d_phi].set(True)
            result = jnp.where(mask, phi, result)
        return result
    
    rhs = ((1/dt) * emult.scalar_consistent_mass_SUPG_mult(phi_n, Mtau_all_n, Me, nconn, nn)
        - 0.5*emult.scalar_stabilized_visc_conv_mult(phi_n, Ce_all_n, Ctau_all_n, Le, D, nconn, nn))

    if gamma_d_phi is not None:
        rhs = rhs.at[gamma_d_phi].set(phi_d)
        x0 = phi_n.at[gamma_d_phi].set(phi_d) 
    else:
        x0 = phi_n

    phi, phi_iters = bicgstab(lhs_operator, rhs, x0=x0, tol=tol)

    return phi, phi_iters

def update_phi(phi_n, nconn, Me, Ce_all, Ce_all_n, Le, D, dt, nn, gamma_d_phi, phi_d, tol):
    
    def lhs_operator(phi):
        result = ((1/dt) * emult.consistent_mass_mult(phi, Me, nconn, nn, 1)
                + 0.5*emult.scalar_visc_conv_mult(phi, Ce_all, Le, D, nconn, nn))
        
        if gamma_d_phi is not None:
            mask = jnp.zeros_like(phi, dtype=bool).at[gamma_d_phi].set(True)
            result = jnp.where(mask, phi, result)
        return result
    
    rhs = ((1/dt) * emult.consistent_mass_mult(phi_n, Me, nconn, nn, 1)
        - 0.5*emult.scalar_visc_conv_mult(phi_n, Ce_all_n, Le, D, nconn, nn))

    if gamma_d_phi is not None:
        rhs = rhs.at[gamma_d_phi].set(phi_d)
        x0 = phi_n.at[gamma_d_phi].set(phi_d) 
    else:
        x0 = phi_n

    phi, phi_iters = bicgstab(lhs_operator, rhs, x0=x0, tol=tol)

    return phi, phi_iters


def timestep(U_n, P_n, M, Le, Ge, rho, mu, dt, nconn, wq, N, Nderiv, 
             gamma_d, U_d, gamma_p, F_p, nn, ndim, stab, p_precond, p_M_op, tol,
             set_zeroP, F, phi_n, gamma_d_phi, phi_d, D, Me, phi_S_func, phi_SUPG):

    Ce_all_n = ecalc.conservative_convec_elem_calc(nconn, U_n, wq, N, Nderiv, nn, ndim, 0.5) 

    U_hat, U_iters = standard.solve_momentum(U_n, P_n, Me, Le, Ge, Ce_all_n, rho, mu, dt, nconn, nn, ndim, gamma_d, U_d, F, tol, 1/2, 1)

    U_star =  U_hat + ((dt/rho) * emult.gradient_element_mult(P_n, Ge, nconn, nn, ndim) / M)

    P, P_iters = standard.solve_pressure(U_star, P_n, Le, Ge, rho, dt, nconn, nn, ndim, gamma_p, F_p, set_zeroP, p_precond, p_M_op, tol, gamma=0)

    U =  U_star - ((dt/rho) * emult.gradient_element_mult(P, Ge, nconn, nn, ndim) / M)
    if gamma_d is not None:
        U = U.at[gamma_d].set(U_d) 

    Ce_all_n = ecalc.conservative_convec_elem_calc(nconn, U_n, wq, N, Nderiv, nn, ndim, 1) 
    Ce_all = ecalc.conservative_convec_elem_calc(nconn, U, wq, N, Nderiv, nn, ndim, 1)

    if phi_SUPG:
        Ctau_tilde_all_n = ecalc.Ctau_tilde_elem_calc(nconn, U_n, wq, N, Nderiv, nn, ndim)
        tau_all_n = ecalc.calculate_tau(nconn, U_n, Ce_all, Ctau_tilde_all_n, N, nn, ndim, D, dt, r=2)

        Ctau_all_n = tau_all_n[:, jnp.newaxis, jnp.newaxis] * Ctau_tilde_all_n
        Mtau_all_n = tau_all_n[:, jnp.newaxis, jnp.newaxis] * Ce_all_n.transpose(0, 2, 1)

        phi, phi_iters = update_phi_SUPG(phi_n, nconn, Mtau_all_n, Ctau_all_n, Me, Ce_all, Ce_all_n, Le, D, dt, nn, 
                                        gamma_d_phi, phi_d, tol)
        
    else:
        phi, phi_iters = update_phi(phi_n, nconn, Me, Ce_all, Ce_all_n, Le, D, dt, nn, gamma_d_phi, 
                                    phi_d, tol)
    
    return U, P, phi, U_iters, P_iters, phi_iters


def setup_solver(node_coords, nconn, bconn, nvec, gamma_d, gamma_p, nn, gamma_d_phi, mesh_func, config: Config):
    
    mesh_config = config.mesh_config
    NS_config = config.solver_config.NS_config
    ADE_config = config.solver_config.ADE_config
    ndim = len(mesh_config.num_elem)

    if ndim == 2:
        if mesh_config.shape_func_ord == 1:
            shape_func = shapefunc.quad4_shape_functions
            boundary_shapefunc = shapefunc.lin2_shape_functions
            mesh_func = rm.create_mesh_2d
        
        elif mesh_config.shape_func_ord == 2:
            shape_func = shapefunc.quad9_shape_functions
            boundary_shapefunc = shapefunc.lin3_shape_functions
            mesh_func = rm.create_mesh_2d_quad9

    elif ndim == 3:
        if mesh_config.shape_func_ord == 1: # add 2 case in future
            shape_func = shapefunc.hex8_shape_functions
            boundary_shapefunc = shapefunc.quad4_shape_functions
            mesh_func = rm.create_mesh_3d

    has_dirichlet = False if gamma_d is None else True
    has_phi_dirichlet = False if gamma_d_phi is None else True

    wq, N, Nderiv = shape_func(node_coords[nconn[0]])

    M, Me = ecalc.lumped_mass_matrix(nconn, wq, N, nn, ndim)
    Le = ecalc.laplacian_element_calc(wq, Nderiv)
    Ge = ecalc.gradient_element_calc(wq, N, Nderiv, ndim)

    nconn_lvls = None

    if NS_config.pressure_precond == "jacobi":
        L_diag = ecalc.laplacian_diag(Le, nconn, nn)

        def pressure_M_op(r):
            return r / L_diag
            
    elif NS_config.pressure_precond == "approx_inv":
        L_diag = ecalc.laplacian_diag(Le, nconn, nn)

        def pressure_M_op(r, nconn):
            D1x = r / L_diag
            D1L = emult.laplacian_element_mult(D1x, Le, nconn, nn) / L_diag
            D1L = D1L.at[gamma_p].set(D1x[gamma_p])
            M1x = (3*D1x - 3*D1L + emult.laplacian_element_mult(D1L, Le, nconn, nn
                                    ).at[gamma_p].set(D1L[gamma_p]) / L_diag)
            return M1x
        
    elif NS_config.pressure_precond == "multigrid":
        mg_data = mg.multigrid_setup(mesh_config.num_elem, NS_config.final_multigrid_mesh, mesh_config.domain_size, NS_config.outlet_faces,
                                     mesh_func, shape_func, mesh_config.shape_func_ord, mesh_config.partial_dirichlet)

        lvl_nums, Le_lvls, nconn_lvls, L_diag_lvls, gamma_p_lvls = mg_data

        # constant fold nconn_lvls (slower compile, faster execution)
        @jax.jit
        def pressure_M_op(r):
            return mg.v_cycle(r, lvl_nums, Le_lvls, nconn_lvls, L_diag_lvls, gamma_p_lvls, NS_config.solver_tol,
                              mesh_config.periodic, NS_config.set_zeroP)
        
    else:
        pressure_M_op = None

    def compute_loaded(U_d, S_func, t_n, t):
        return standard.compute(node_coords, bconn, nvec, boundary_shapefunc, gamma_d, U_d, nn, ndim, S_func, Me, nconn, t_n, t)
    
    # no constant folding of nconn (faster compile, slower execution)
    @partial(jax.jit, static_argnames=("p_M_op"))
    def timestep_jitted(U_n, P_n, phi_n, rho, mu, D, dt, U_d, F_p, F, phi_d, nconn, p_M_op, phi_S_func):
        return timestep(U_n, P_n, M, Le, Ge, rho, mu, dt, nconn, wq, N, Nderiv, 
                        gamma_d, U_d, gamma_p, F_p, nn, ndim, NS_config.streamline_diff, NS_config.pressure_precond, p_M_op, NS_config.solver_tol,
                        NS_config.set_zeroP, F, phi_n, gamma_d_phi, phi_d, D, Me, phi_S_func, ADE_config.phi_SUPG)
    
    def timestep_loaded(U_n, P_n, phi_n, dt, rho, mu, D, U_d, phi_d, F_p, F, phi_S_func):
        return timestep_jitted(U_n, P_n, phi_n, rho, mu, D, dt, U_d, F_p, F, phi_d, nconn, pressure_M_op, phi_S_func)

    return compute_loaded, timestep_loaded
    