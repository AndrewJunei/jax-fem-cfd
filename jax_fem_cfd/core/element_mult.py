import jax
import jax.numpy as jnp
from ..core.element_calc import jit_segment_sum

def laplacian_element_mult(P, Le, nconn, nn):
    """ Computes the result of L @ P
    """
    def compute_LPe(p):
        return Le @ p

    v_compute_LPe = jax.vmap(compute_LPe)
    LP_data = v_compute_LPe(P[nconn])

    LP = jit_segment_sum(LP_data.flatten(), nconn.flatten(), nn)

    return LP

def gradient_element_mult_2d(P, Ge, nconn, nn):
    """ Computes the result of G @ P = [[G1 @ P], [G2 @ P]]
    """
    P_elem = P[nconn]

    def compute_GPe(Gie, p):
        return Gie @ p
    
    v_compute_GPe = jax.vmap(compute_GPe, in_axes=(None, 0))

    G1P_data = v_compute_GPe(Ge[0], P_elem)
    G2P_data = v_compute_GPe(Ge[1], P_elem)

    indices = nconn.flatten()
    G1P = jit_segment_sum(G1P_data.flatten(), indices, nn)
    G2P = jit_segment_sum(G2P_data.flatten(), indices, nn)

    return jnp.concatenate((G1P, G2P))

def stabilized_visc_conv_mult_2d(U, Ce_all, Se_all, Le, mu, nconn, nn):
    """ Computes the combined (C + K + S) @ U 
    """
    Ke = mu * Le # viscous matrix is viscosity * laplacian

    def combined_mult(Ce, Se, Ke, u):
        return (Ce + Se + Ke) @ u
    
    v_combined_mult = jax.vmap(combined_mult, in_axes=(0, 0, None, 0))

    u_data = v_combined_mult(Ce_all, Se_all, Ke, U[nconn])
    v_data = v_combined_mult(Ce_all, Se_all, Ke, U[nconn + nn])

    indices = nconn.flatten()
    u_result = jit_segment_sum(u_data.flatten(), indices, nn)
    v_result = jit_segment_sum(v_data.flatten(), indices, nn)

    return jnp.concatenate((u_result, v_result))

def divergence_element_mult_2d(U, Ge, nconn, nn):
    """ Computes D @ U = (D1 @ u) + (D2 @ v), where U = [[u], [v]]
    """
    def compute_div_mult(De, u):
        return De @ u
    
    v_compute_div_mult = jax.vmap(compute_div_mult, in_axes=(None, 0))

    D1u_data = v_compute_div_mult(Ge[0].T, U[nconn])
    D2v_data = v_compute_div_mult(Ge[1].T, U[nconn + nn])

    indices = nconn.flatten()
    D1u = jit_segment_sum(D1u_data.flatten(), indices, nn)
    D2v = jit_segment_sum(D2v_data.flatten(), indices, nn)

    return D1u + D2v


def gradient_element_mult_3d(P, Ge, nconn, nn):
    """ Computes the result of G @ P = [[G1 @ P], [G2 @ P], [G3 @ P]]
    """
    P_elem = P[nconn]

    def compute_GPe(Gie, p):
        return Gie @ p
    
    v_compute_GPe = jax.vmap(compute_GPe, in_axes=(None, 0))

    G1P_data = v_compute_GPe(Ge[0], P_elem)
    G2P_data = v_compute_GPe(Ge[1], P_elem)
    G3P_data = v_compute_GPe(Ge[2], P_elem)

    indices = nconn.flatten()
    G1P = jit_segment_sum(G1P_data.flatten(), indices, nn)
    G2P = jit_segment_sum(G2P_data.flatten(), indices, nn)
    G3P = jit_segment_sum(G3P_data.flatten(), indices, nn)

    return jnp.concatenate((G1P, G2P, G3P))

def stabilized_visc_conv_mult_3d(U, Ce_all, Se_all, Le, mu, nconn, nn):
    """ Computes the combined (C + K + S) @ U 
    """
    Ke = mu * Le # viscous matrix is viscosity * laplacian

    def combined_mult(Ce, Se, Ke, u):
        return (Ce + Se + Ke) @ u
    
    v_combined_mult = jax.vmap(combined_mult, in_axes=(0, 0, None, 0))

    u_data = v_combined_mult(Ce_all, Se_all, Ke, U[nconn])
    v_data = v_combined_mult(Ce_all, Se_all, Ke, U[nconn + nn])
    w_data = v_combined_mult(Ce_all, Se_all, Ke, U[nconn + 2*nn])

    indices = nconn.flatten()
    u_result = jit_segment_sum(u_data.flatten(), indices, nn)
    v_result = jit_segment_sum(v_data.flatten(), indices, nn)
    w_result = jit_segment_sum(w_data.flatten(), indices, nn)

    return jnp.concatenate((u_result, v_result, w_result))

def divergence_element_mult_3d(U, Ge, nconn, nn):
    """ Computes D @ U = (D1 @ u) + (D2 @ v) + (D3 @ w), where U = [[u], [v], [w]]
    """
    def compute_div_mult(De, u):
        return De @ u
    
    v_compute_div_mult = jax.vmap(compute_div_mult, in_axes=(None, 0))

    D1u_data = v_compute_div_mult(Ge[0].T, U[nconn])
    D2v_data = v_compute_div_mult(Ge[1].T, U[nconn + nn])

    indices = nconn.flatten()
    D1u = jit_segment_sum(D1u_data.flatten(), indices, nn)
    D2v = jit_segment_sum(D2v_data.flatten(), indices, nn)

    return D1u + D2v