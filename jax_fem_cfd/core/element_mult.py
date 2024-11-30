import jax
import jax.numpy as jnp

""" Currently we assume that the element matrices Le and Ge are the same for each element and that
    the viscosity is a constant over each node. 
    If we no longer want to assume this, the element matrices should be modified in element_calc.py
    and the functions here can be modified such that the vmap treats these matrices like the 
    convection in visc_conv_mult
"""

def consistent_mass_mult(S, Me, nconn, nn, ndim):
    """ Computes M @ S without mass lumping
    """
    def compute_MSe(s):
        return Me @ s

    v_compute_MSe = jax.vmap(compute_MSe)
    indices = nconn.flatten()

    x_data = v_compute_MSe(S[nconn])
    x_result = jax.ops.segment_sum(x_data.flatten(), indices, nn)

    if ndim == 3:
        z_data = v_compute_MSe(S[nconn + 2*nn])
        z_result = jax.ops.segment_sum(z_data.flatten(), indices, nn)
        result = jnp.concatenate((x_result, y_result, z_result))

    elif ndim == 2:
        y_data = v_compute_MSe(S[nconn + nn])
        y_result = jax.ops.segment_sum(y_data.flatten(), indices, nn)
        result = jnp.concatenate((x_result, y_result))

    elif ndim == 1:
        result = x_result

    return result

def laplacian_element_mult(P, Le, nconn, nn):
    """ Computes the result of L @ P
    """
    def compute_LPe(p):
        return Le @ p

    v_compute_LPe = jax.vmap(compute_LPe)
    LP_data = v_compute_LPe(P[nconn])

    LP = jax.ops.segment_sum(LP_data.flatten(), nconn.flatten(), nn)

    return LP

def gradient_element_mult(P, Ge, nconn, nn, ndim):
    """ Computes the result of G @ P = [[G1 @ P], [G2 @ P]] if ndim=2
                                       [[G1 @ P], [G2 @ P], [G3 @ P]] if ndim=3
    """
    P_elem = P[nconn]

    def compute_GPe(Gie, p):
        return Gie @ p
    
    v_compute_GPe = jax.vmap(compute_GPe, in_axes=(None, 0))
    indices = nconn.flatten()

    G1P_data = v_compute_GPe(Ge[0], P_elem)
    G2P_data = v_compute_GPe(Ge[1], P_elem)

    G1P = jax.ops.segment_sum(G1P_data.flatten(), indices, nn)
    G2P = jax.ops.segment_sum(G2P_data.flatten(), indices, nn)

    if ndim == 3:
        G3P_data = v_compute_GPe(Ge[2], P_elem)
        G3P = jax.ops.segment_sum(G3P_data.flatten(), indices, nn)
        GP = jnp.concatenate((G1P, G2P, G3P))

    elif ndim == 2:
        GP = jnp.concatenate((G1P, G2P))

    return GP

def visc_conv_mult(U, Ce_all, Le, mu, rho, nconn, nn, ndim):
    """ Computes the combined (C + K) @ U 
    """
    Ke = mu * Le # viscous matrix is viscosity * laplacian

    def combined_mult(Ce, Ke, u):
        return (rho*Ce + Ke) @ u
    
    v_combined_mult = jax.vmap(combined_mult, in_axes=(0, None, 0))
    indices = nconn.flatten()

    u_data = v_combined_mult(Ce_all, Ke, U[nconn])
    v_data = v_combined_mult(Ce_all, Ke, U[nconn + nn])

    u_result = jax.ops.segment_sum(u_data.flatten(), indices, nn)
    v_result = jax.ops.segment_sum(v_data.flatten(), indices, nn)

    if ndim == 3:
        w_data = v_combined_mult(Ce_all, Ke, U[nconn + 2*nn])
        w_result = jax.ops.segment_sum(w_data.flatten(), indices, nn)
        result = jnp.concatenate((u_result, v_result, w_result))

    elif ndim == 2:
        result = jnp.concatenate((u_result, v_result))

    return result

def stabilized_visc_conv_mult(U, Ce_all, Se_all, Le, mu, rho, nconn, nn, ndim):
    """ Computes the combined (C + K + S) @ U 
    """
    Ke = mu * Le # viscous matrix is viscosity * laplacian

    def combined_mult(Ce, Ke, Se, u):
        return (rho*Ce + Ke + Se) @ u
    
    v_combined_mult = jax.vmap(combined_mult, in_axes=(0, None, 0, 0))
    indices = nconn.flatten()

    u_data = v_combined_mult(Ce_all, Ke, Se_all, U[nconn])
    v_data = v_combined_mult(Ce_all, Ke, Se_all, U[nconn + nn])

    u_result = jax.ops.segment_sum(u_data.flatten(), indices, nn)
    v_result = jax.ops.segment_sum(v_data.flatten(), indices, nn)

    if ndim == 3:
        w_data = v_combined_mult(Ce_all, Ke, Se_all, U[nconn + 2*nn])
        w_result = jax.ops.segment_sum(w_data.flatten(), indices, nn)
        result = jnp.concatenate((u_result, v_result, w_result))

    elif ndim == 2:
        result = jnp.concatenate((u_result, v_result))

    return result

def divergence_element_mult(U, Ge, nconn, nn, ndim):
    """ Computes D @ U = (D1 @ u) + (D2 @ v) where U = [[u], [v]] if ndim=2
                         (D1 @ u) + (D2 @ v) + (D3 @ w) where U = [[u], [v], [w]] if ndim=3 
    """
    def compute_div_mult(De, u):
        return De @ u
    
    v_compute_div_mult = jax.vmap(compute_div_mult, in_axes=(None, 0))
    indices = nconn.flatten()

    D1u_data = v_compute_div_mult(Ge[0].T, U[nconn])
    D2v_data = v_compute_div_mult(Ge[1].T, U[nconn + nn])

    D1u = jax.ops.segment_sum(D1u_data.flatten(), indices, nn)
    D2v = jax.ops.segment_sum(D2v_data.flatten(), indices, nn)

    if ndim == 3:
        D3w_data = v_compute_div_mult(Ge[2].T, U[nconn + 2*nn])
        D3w = jax.ops.segment_sum(D3w_data.flatten(), indices, nn)
        DU = D1u + D2v + D3w

    elif ndim == 2:
        DU = D1u + D2v

    return DU

def scalar_visc_conv_mult(phi, Ce_all, Le, D, nconn, nn):
    """ Computes (C + K) @ phi
    """
    Ke = D * Le # D is the diffusion constant

    def combined_mult(Ce, Ke, phie):
        return (Ce + Ke) @ phie
    
    v_combined_mult = jax.vmap(combined_mult, in_axes=(0, None, 0))
    data = v_combined_mult(Ce_all, Ke, phi[nconn])
    
    result = jax.ops.segment_sum(data.flatten(), nconn.flatten(), nn)

    return result

def scalar_stabilized_visc_conv_mult(phi, Ce_all, Se_all, Le, D, nconn, nn):
    """ Computes (C + K + S) @ phi where S is the streamline diffusion matrix
    """
    Ke = D * Le # D is the diffusion constant

    def combined_mult(Ce, Ke, Se, phie):
        return (Ce + Ke + Se) @ phie
    
    v_combined_mult = jax.vmap(combined_mult, in_axes=(0, None, 0, 0))
    data = v_combined_mult(Ce_all, Ke, Se_all, phi[nconn])
    
    result = jax.ops.segment_sum(data.flatten(), nconn.flatten(), nn)

    return result


def scalar_consistent_mass_SUPG_mult(phi, Me_tau_all, Me, nconn, nn):
    """ Computes (M + M^tau) @ phi where M is not lumped and M^tau comes from SUPG
    """
    def combined_mult(Me_tau, phie):
        return (Me + Me_tau) @ phie

    v_combined_mult = jax.vmap(combined_mult)
    data = v_combined_mult(Me_tau_all, phi[nconn])
    
    result = jax.ops.segment_sum(data.flatten(), nconn.flatten(), nn)

    return result
