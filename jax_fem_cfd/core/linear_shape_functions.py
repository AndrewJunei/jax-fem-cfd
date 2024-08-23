import jax
import jax.numpy as jnp

""" I, q, and dim index nodes, quadrature points, and spatial dimenstions respectively.
    - coords: nodal coordinates for a given element (I, dim)
    - wq: quadrature weights (q, 1)
    - N: shape function for node I at point q (q, I)
    - Nderiv: derivative wrt dim of shape function for node I at point q (q, dim, I) 
"""

def linear_1d_shape_functions(coords):
    """ Linear 2 node element
    """
    xi_q = jnp.array([-1, 1]) / jnp.sqrt(3) # (2,)
    N = jnp.stack([(1 - xi_q) / 2, (1 + xi_q) / 2], axis=1) # (2,2)
    dNdxi = jnp.array([[-0.5, 0.5], [-0.5, 0.5]]) 

    dx = coords[1, 0] - coords[0, 0]  
    dy = coords[1, 1] - coords[0, 1]  
    h = jnp.sqrt(dx**2 + dy**2) # element length

    Nderiv = dNdxi * (2 / h) # (2,2)
    wq = (jnp.array([1, 1]) * (h / 2))[:, jnp.newaxis] # (2,1)
    
    return wq, N, Nderiv

def quad_2d_shape_functions(coords):
    """ Bilinear 4 node quadrilateral element
    """
    xi_q = jnp.array([-1, 1, 1, -1]) / jnp.sqrt(3) # (4,)
    eta_q = jnp.array([-1, -1, 1, 1]) / jnp.sqrt(3) # (4,)

    N = jnp.stack([(1-xi_q)*(1-eta_q), (1+xi_q)*(1-eta_q), 
                   (1+xi_q)*(1+eta_q), (1-xi_q)*(1+eta_q)], axis=1) / 4 # (4,4)

    dNdxi = jnp.stack([-(1-eta_q), 1-eta_q, 1+eta_q, -(1+eta_q)], axis=1) / 4 # (4,4)
    dNdeta = jnp.stack([-(1-xi_q), -(1+xi_q), 1+xi_q, 1-xi_q], axis=1) / 4 # (4,4)
    J = jnp.concatenate([(dNdxi @ coords)[:, jnp.newaxis, :], 
                         (dNdeta @ coords)[:, jnp.newaxis, :]], axis=1) # (4,2,2)
    
    wq = jnp.linalg.det(J)[:, jnp.newaxis] # (4,1)
    Nderiv = jnp.linalg.inv(J) @ jnp.concatenate([dNdxi[:, jnp.newaxis, :], 
                                                  dNdeta[:, jnp.newaxis, :]], axis=1) # (4,2,4)
    
    return wq, N, Nderiv

def hex_3d_shape_functions(coords):
    """ Trilinear 8 node hexahedral element
    """
    xi_q = jnp.array([-1, 1, 1, -1, -1, 1, 1, -1]) / jnp.sqrt(3) # (8,)
    eta_q = jnp.array([-1, -1, 1, 1, -1, -1, 1, 1]) / jnp.sqrt(3) # (8,)
    zeta_q = jnp.array([-1, -1, -1, -1, 1, 1, 1, 1]) / jnp.sqrt(3) # (8,)

    N = jnp.stack([
        (1-xi_q)*(1-eta_q)*(1-zeta_q), (1+xi_q)*(1-eta_q)*(1-zeta_q), 
        (1+xi_q)*(1+eta_q)*(1-zeta_q), (1-xi_q)*(1+eta_q)*(1-zeta_q), 
        (1-xi_q)*(1-eta_q)*(1+zeta_q), (1+xi_q)*(1-eta_q)*(1+zeta_q),
        (1+xi_q)*(1+eta_q)*(1+zeta_q), (1-xi_q)*(1+eta_q)*(1+zeta_q)], axis=1) / 8 # (8,8)

    dNdxi = jnp.stack([
        -(1-eta_q)*(1-zeta_q), (1-eta_q)*(1-zeta_q), (1+eta_q)*(1-zeta_q), -(1+eta_q)*(1-zeta_q), 
        -(1-eta_q)*(1+zeta_q), (1-eta_q)*(1+zeta_q), (1+eta_q)*(1+zeta_q), -(1+eta_q)*(1+zeta_q)
        ], axis=1) / 8 # (8,8)
    
    dNdeta = jnp.stack([
        -(1-xi_q)*(1-zeta_q), -(1+xi_q)*(1-zeta_q), (1+xi_q)*(1-zeta_q), (1-xi_q)*(1-zeta_q),
        -(1-xi_q)*(1+zeta_q), -(1+xi_q)*(1+zeta_q), (1+xi_q)*(1+zeta_q), (1-xi_q)*(1+zeta_q)
        ], axis=1) / 8 # (8,8)
    
    dNdzeta = jnp.stack([
        -(1-xi_q)*(1-eta_q), -(1+xi_q)*(1-eta_q), -(1+xi_q)*(1+eta_q), -(1-xi_q)*(1+eta_q),
        (1-xi_q)*(1-eta_q), (1+xi_q)*(1-eta_q), (1+xi_q)*(1+eta_q), (1-xi_q)*(1+eta_q)
        ], axis=1) / 8 # (8,8)

    J = jnp.concatenate([(dNdxi @ coords)[:, jnp.newaxis, :], 
                         (dNdeta @ coords)[:, jnp.newaxis, :],
                         (dNdzeta @ coords)[:, jnp.newaxis, :]], axis=1) # (8,3,3)

    wq = jnp.linalg.det(J)[:, jnp.newaxis] # (8,1)

    Nderiv = (jnp.linalg.inv(J) @ 
                jnp.concatenate([dNdxi[:, jnp.newaxis, :],
                                 dNdeta[:, jnp.newaxis, :],
                                 dNdzeta[:, jnp.newaxis, :]], axis=1)) # (8,3,8)
    
    return wq, N, Nderiv