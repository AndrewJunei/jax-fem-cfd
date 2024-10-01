import jax
import jax.numpy as jnp
from ..core.element_calc import jit_segment_sum

""" Currently we assume that the element matrices Le and Ge are the same for each element and that
    the viscosity is a constant over each node. 
    If we no longer want to assume this, the element matrices should be modified in element_calc.py
    and the functions here can be modified such that the vmap treats these matrices like the 
    convection in visc_conv_mult
"""

def laplacian_element_mult(P, Le, nconn, nn):
    """ Computes the result of L @ P
    """
    def compute_LPe(p):
        return Le @ p

    v_compute_LPe = jax.vmap(compute_LPe)
    LP_data = v_compute_LPe(P[nconn])

    LP = jit_segment_sum(LP_data.flatten(), nconn.flatten(), nn)

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

    G1P = jit_segment_sum(G1P_data.flatten(), indices, nn)
    G2P = jit_segment_sum(G2P_data.flatten(), indices, nn)

    if ndim == 3:
        G3P_data = v_compute_GPe(Ge[2], P_elem)
        G3P = jit_segment_sum(G3P_data.flatten(), indices, nn)
        GP = jnp.concatenate((G1P, G2P, G3P))

    elif ndim == 2:
        GP = jnp.concatenate((G1P, G2P))

    return GP

def visc_conv_mult(U, Ce_all, Le, mu, nconn, nn, ndim):
    """ Computes the combined (C + K) @ U 
    """
    Ke = mu * Le # viscous matrix is viscosity * laplacian

    def combined_mult(Ce, Ke, u):
        return (Ce + Ke) @ u
    
    v_combined_mult = jax.vmap(combined_mult, in_axes=(0, None, 0))
    indices = nconn.flatten()

    u_data = v_combined_mult(Ce_all, Ke, U[nconn])
    v_data = v_combined_mult(Ce_all, Ke, U[nconn + nn])

    u_result = jit_segment_sum(u_data.flatten(), indices, nn)
    v_result = jit_segment_sum(v_data.flatten(), indices, nn)

    if ndim == 3:
        w_data = v_combined_mult(Ce_all, Ke, U[nconn + 2*nn])
        w_result = jit_segment_sum(w_data.flatten(), indices, nn)
        result = jnp.concatenate((u_result, v_result, w_result))

    elif ndim == 2:
        result = jnp.concatenate((u_result, v_result))

    return result

def stabilized_visc_conv_mult(U, Ce_all, Se_all, Le, mu, nconn, nn, ndim):
    """ Computes the combined (C + K + S) @ U 
    """
    Ke = mu * Le # viscous matrix is viscosity * laplacian

    def combined_mult(Ce, Ke, Se, u):
        return (Ce + Ke + Se) @ u
    
    v_combined_mult = jax.vmap(combined_mult, in_axes=(0, None, 0, 0))
    indices = nconn.flatten()

    u_data = v_combined_mult(Ce_all, Ke, Se_all, U[nconn])
    v_data = v_combined_mult(Ce_all, Ke, Se_all, U[nconn + nn])

    u_result = jit_segment_sum(u_data.flatten(), indices, nn)
    v_result = jit_segment_sum(v_data.flatten(), indices, nn)

    if ndim == 3:
        w_data = v_combined_mult(Ce_all, Ke, Se_all, U[nconn + 2*nn])
        w_result = jit_segment_sum(w_data.flatten(), indices, nn)
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

    D1u = jit_segment_sum(D1u_data.flatten(), indices, nn)
    D2v = jit_segment_sum(D2v_data.flatten(), indices, nn)

    if ndim == 3:
        D3w_data = v_compute_div_mult(Ge[2].T, U[nconn + 2*nn])
        D3w = jit_segment_sum(D3w_data.flatten(), indices, nn)
        DU = D1u + D2v + D3w

    elif ndim == 2:
        DU = D1u + D2v

    return DU
